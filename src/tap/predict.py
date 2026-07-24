"""变压器时序属性预测推理脚本（可独立执行，零业务依赖）。

自包含地实现：R-GCN 静态链接预测模型 / TGN 属性预测模型 / 预测打印，全部集中在本文件，
不依赖 demo/fault_prediction.py。推理所需的异构图由 train.py 训练时持久化到模型目录，
本脚本直接加载，不再读取原始数据集。

注意：本脚本只做预测，不训练、不读数据集。需先由 train.py --mode prediction 训练并保存
模型权重与知识图谱到模型目录（默认 ./trained_models/trend）后，再运行本脚本加载推理。

运行方式：
    python trend_predict.py
    python trend_predict.py --model-dir /path/to/trained_models/trend
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, Linear

# 允许以脚本方式直接运行（`python trend_predict.py`）
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model import TGNModel  # noqa: E402

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 健康状态映射
HEALTH_MAPPING = {
    0: "正常运行",
    1: "轻微过热",
    2: "严重过热",
    3: "过载故障",
}
HEALTH_NUM = 4
# 特征列表
FEATURE_LIST = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
FEATURE_NUM = len(FEATURE_LIST)
# 模型配置
HIDDEN_DIM = 32


# ===================== 1. 模型1：R-GCN 静态链接预测模型 =====================
class RGCNFaultDiagnosis(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # 二分图卷积：health_state(源,4维) → time_slice(目标,FEATURE_NUM维)，输出 per-time_slice
        self.conv1 = RGCNConv((4, FEATURE_NUM), hidden_dim, num_relations=1)
        self.conv2 = RGCNConv((hidden_dim, hidden_dim), hidden_dim, num_relations=1)
        self.out_linear = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_slice = x_dict["time_slice"]
        x_health = x_dict["health_state"]
        edge = edge_index_dict["time_slice", "has_health_state", "health_state"]
        # 翻转边方向：health_state → time_slice，使输出为每个 time_slice 的嵌入
        edge_rev = edge.flip(0)
        edge_type = torch.zeros(edge_rev.shape[1], dtype=torch.long, device=edge_rev.device)
        # source=health_state, target=time_slice → 输出 shape: [num_time_slices, hidden_dim]
        h1 = self.conv1((x_health, x_slice), edge_rev, edge_type=edge_type)
        h1 = F.relu(h1)
        h2 = self.conv2((h1, h1), edge_rev, edge_type=edge_type)
        logits = self.out_linear(h2)
        return logits, h2


# ===================== 2. 模型2：TGN 时序属性预测模型 =====================
# TGN 模型核心已抽取到 src/model/tgn.py，这里封装一个便捷工厂函数。
def build_tgn_oil_temperature_predict(hidden_dim, num_time_slices):
    """构建一个用于变压器时序异构图的 TGN 油温预测模型。

    配置与原脚本保持完全一致，包括节点输入维度、边类型与预测头结构。
    """
    return TGNModel(
        in_channels={
            "transformer": 3,
            "time_slice": FEATURE_NUM,
            "health_state": 4,
            "feature_indicator": 2,
        },
        edge_types=[
            ("transformer", "has_time_slice", "time_slice"),
            ("time_slice", "next", "time_slice"),
            ("time_slice", "has_health_state", "health_state"),
            ("time_slice", "has_feature", "feature_indicator"),
            ("health_state", "state_has_symbol", "feature_indicator"),
        ],
        temporal_edge_types=[
            ("transformer", "has_time_slice", "time_slice"),
            ("time_slice", "next", "time_slice"),
        ],
        hidden_dim=hidden_dim,
        num_time_slices=num_time_slices,
        num_layers=2,
        ot_feature_index=-1,
    )


# ===================== 3. 预测打印函数（仅输出预测结果，不依赖真实标签） =====================
def predict_print(kg_data, diag_model, tgn_model, slice_idx):
    """对指定时序切片执行静态链接预测+属性预测，完整打印预测结果（无需真实标签）"""
    print("=" * 100)
    print(f"【变压器时序推理】切片ID：{slice_idx} | 时间：{kg_data['time_slice'].date_str[slice_idx]}")
    print("=" * 100)

    # 1. 提取切片基础信息（使用 x_raw 即原始未标准化值）
    slice_feat = kg_data["time_slice"].x_raw[slice_idx].cpu().numpy()
    print(f"\n[1] 切片基础运行信息：")
    print(f"  核心运行特征：")
    for i, feat_name in enumerate(FEATURE_LIST):
        print(f"    {feat_name}: {slice_feat[i]:.4f}")

    # 2. R-GCN静态链接预测
    with torch.no_grad():
        health_logits, _ = diag_model(kg_data.x_dict, kg_data.edge_index_dict)
        pred_health_logits = health_logits[slice_idx]
        pred_health = torch.argmax(pred_health_logits).item()
        pred_health_prob = F.softmax(pred_health_logits, dim=0).cpu().numpy()

    print(f"\n[2] R-GCN知识图谱静态链接预测结果：")
    print(f"  预测健康状态：{HEALTH_MAPPING[pred_health]}")
    print(f"  各类别预测概率：")
    for label_id, state_name in HEALTH_MAPPING.items():
        print(f"    {state_name}: {pred_health_prob[label_id]:.2%}")

    # 3. 知识图谱行业规则匹配解释
    print(f"\n[3] 知识图谱行业规则匹配解释：")
    rules = []
    ot_val = slice_feat[6]
    hufl_val = slice_feat[0]
    if pred_health == 3 and hufl_val > 20:
        rules.append("规则1：主负载HUFL超标 → 图谱关联过载故障典型特征")
    if pred_health == 2 and ot_val >= 50:
        rules.append("规则2：油温OT≥50℃ → 图谱关联严重过热典型特征")
    if pred_health == 1 and 40 <= ot_val < 50:
        rules.append("规则3：油温40℃≤OT<50℃ → 图谱关联轻微过热典型特征")
    if not rules:
        rules.append("无异常特征，符合正常运行状态")
    for rule in rules:
        print(f"  {rule}")

    # 4. TGN时序属性预测（模型输出是 z-score，需反标准化成摄氏度）
    with torch.no_grad():
        future_ot_pred, fault_risk_pred, _ = tgn_model(kg_data)
        ot_mean_val = kg_data["time_slice"].ot_mean.item()
        ot_std_val = kg_data["time_slice"].ot_std.item()
        # 反标准化：z-score → 摄氏度
        pred_future_ot = future_ot_pred[slice_idx].item() * ot_std_val + ot_mean_val
        pred_fault_risk = fault_risk_pred[slice_idx].item()

    risk_level = "低风险" if pred_fault_risk < 0.3 else ("中风险" if pred_fault_risk < 0.7 else "高风险")
    print(f"\n[4] TGN时序图模型属性预测结果：")
    print(f"  未来3步预测油温：{pred_future_ot:.4f}℃")
    print(f"  未来故障发生概率：{pred_fault_risk:.2%}，风险等级：{risk_level}")

    # 5. 最终诊断结论
    print(f"\n[5] 最终运维建议：")
    if pred_health == 0 and pred_fault_risk < 0.3:
        print("  ✅ 设备运行正常，维持常规巡检周期")
    elif pred_health == 1 or pred_fault_risk >= 0.3:
        print("  ⚠️  设备轻微异常，缩短巡检周期，密切关注油温与负载变化")
    elif pred_health == 2 or pred_fault_risk >= 0.7:
        print("  ⚠️  设备严重过热，立即安排停电检修，检查绝缘与散热系统")
    elif pred_health == 3:
        print("  ❌  设备过载故障，立即降低负载，紧急停机检查")

    print("\n" + "=" * 100 + " 推理结束 " + "=" * 100 + "\n")


# ===================== 主运行入口（纯预测，不训练、不读数据集） =====================
if __name__ == "__main__":
    import argparse

    # 模型/图谱目录，默认与 train.py 的 prediction.save_model_path 一致
    parser = argparse.ArgumentParser(description="变压器时序属性预测推理（不读取数据集）")
    parser.add_argument(
        "--model-dir", type=str,
        default="/mnt/work/code/python_workspace/kg-geometric/trained_models/trend",
        help="train.py 保存的模型与知识图谱目录（含 temporal_*.pth 与 temporal_kg_graph.pt）",
    )
    args = parser.parse_args()
    MODEL_DIR = args.model_dir

    # 1. 加载训练时持久化的知识图谱（含节点特征/边/时序边/反归一化统计量）
    #    推理所需的整图已由 train.py 保存，这里直接加载，不再读取原始数据集。
    graph_path = f"{MODEL_DIR}/temporal_kg_graph.pt"
    diag_path = f"{MODEL_DIR}/temporal_diag_rgcn.pth"
    tgn_path = f"{MODEL_DIR}/temporal_tgn.pth"
    if not (os.path.exists(graph_path) and os.path.exists(diag_path) and os.path.exists(tgn_path)):
        raise FileNotFoundError(
            f"未找到训练好的模型/图谱文件：\n  {diag_path}\n  {tgn_path}\n  {graph_path}\n"
            f"请先运行 train.py --mode prediction 完成训练并保存。"
        )

    kg_graph = torch.load(graph_path, map_location=DEVICE, weights_only=False)
    kg_graph = kg_graph.to(DEVICE)

    # 2. 加载已训练好的模型权重（预测脚本不负责训练）
    total_num = kg_graph["time_slice"].x.shape[0]
    diag_model = RGCNFaultDiagnosis(HIDDEN_DIM, HEALTH_NUM).to(DEVICE)
    tgn_model = build_tgn_oil_temperature_predict(HIDDEN_DIM, total_num).to(DEVICE)
    diag_model.load_state_dict(torch.load(diag_path, map_location=DEVICE, weights_only=True))
    diag_model.eval()
    tgn_model.load_state_dict(torch.load(tgn_path, map_location=DEVICE, weights_only=True))
    tgn_model.eval()
    print(f"已加载模型与知识图谱：\n  {diag_path}\n  {tgn_path}\n  {graph_path}")

    # 3. 对最新的若干时序切片执行属性预测（模拟对实时运行数据的预测）
    HOLD_OUT_N = 10
    pred_idx = torch.tensor(
        np.arange(max(0, total_num - HOLD_OUT_N), total_num), dtype=torch.long
    ).to(DEVICE)
    print("\n" + "=" * 100)
    print(f"【属性预测】对最新 {pred_idx.shape[0]} 个时序切片执行静态链接预测 + 油温属性预测")
    print("=" * 100)
    for i in range(pred_idx.shape[0]):
        slice_id = pred_idx[i].item()
        predict_print(kg_graph, diag_model, tgn_model, slice_id)

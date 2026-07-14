"""统一的知识图谱 + 时序数据模型训练脚本。

本脚本同时集成两套完整的训练流程：

1. **故障诊断**（`--mode diagnosis`）
   - 基于 ES 中的三元组知识图谱，使用 R-GCN 训练节点嵌入，支持自然语言查询 → 故障推理。
   - 参考原 `train.py` 的逻辑。

2. **故障预测**（`--mode prediction`）
   - 基于变压器时序 CSV 数据，构建 transformer / time_slice / health_state /
     feature_indicator 的异构图知识图谱，联合训练 R-GCN（健康状态分类）+
     TGN（油温趋势 + 故障风险预测）。
   - 参考 `demo/fault_prediction.py` 的逻辑。

3. **一键运行**（`--mode both`，默认）
   - 顺序执行上述两个流程，便于一次性验证整个模型管线。

用法:
    # 故障诊断 (默认使用 ES 中的 "车辆故障诊断" 图谱)
    python train.py --mode diagnosis
    python train.py --mode diagnosis --epochs 500 --hidden-dim 128

    # 故障预测 (使用 ETTh1/ETTh2 等变压器时序数据)
    python train.py --mode prediction --csv-path /path/to/ETTh1.csv
    python train.py --mode prediction --hold-out 10 --epochs 200

    # 一键运行两个流程
    python train.py
    python train.py --save-model ./models/ --query "加速迟缓"
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, Linear

from src.core.config import (
    DiagnosisTrainingConfig,
    ESConfig,
    PredictionTrainingConfig,
    TrainingConfig,
    logger,
)
from src.dataset.temporal_dataset import KGTemporalDataset
from src.dataset.triple_dataset import KGTripleDataset
from src.model import TGNOilTemperaturePredict
from src.model.rgcn import FaultRGCN
from src.model.training import (
    evaluate,
    split_masks,
    train_joint_rgcn_tgn,
    train_rgcn,
)
from src.pipeline.inference import (
    DiagnosisResult,
    format_result,
    format_result_compact,
    infer_from_text,
)

# =============================================================================
# 通用工具
# =============================================================================


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


# =============================================================================
# 模块 A：故障诊断（R-GCN + 三元组知识图谱）
# =============================================================================


def run_fault_diagnosis_training(
        dataset: KGTripleDataset,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 5e-4,
        epochs: int = 300,
        weight_decay: float = 5e-4,
        device: torch.device | str = "cpu",
        save_model_path: str | None = None,
) -> FaultRGCN:
    """训练 R-GCN 故障诊断模型并返回训练好的模型。"""
    device = _resolve_device(device) if isinstance(device, str) else device

    # ---- 划分 7:1:2 训练/验证/测试集 ----
    train_mask, val_mask, test_mask = split_masks(
        dataset.num_nodes,
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    )
    logger.info(
        "故障诊断 | 数据划分: 训练=%d, 验证=%d, 测试=%d (总计 %d 节点)",
        int(train_mask.sum().item()),
        int(val_mask.sum().item()),
        int(test_mask.sum().item()),
        dataset.num_nodes,
    )

    # ---- 构建模型 ----
    model = FaultRGCN(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_bases=None,
    ).to(device)

    edge_index = dataset.edge_index.to(device)
    edge_type = dataset.edge_type.to(device)
    y = dataset.y.to(device)
    train_mask_t = train_mask.to(device)
    val_mask_t = val_mask.to(device)
    test_mask_t = test_mask.to(device)

    logger.info(
        "故障诊断 | R-GCN: nodes=%d, relations=%d, hidden=%d, layers=%d, params=%d",
        model.num_nodes, model.num_relations, hidden_dim, num_layers,
        sum(p.numel() for p in model.parameters()),
    )

    # ---- 训练 ----
    logger.info(
        "故障诊断 | 开始训练 (lr=%.0e, weight_decay=%.0e, epochs=%d, dropout=%.1f)",
        lr, weight_decay, epochs, dropout,
    )
    history = train_rgcn(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        train_mask=train_mask_t,
        val_mask=val_mask_t,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=30,
        verbose=True,
    )

    # ---- 评估 ----
    eval_results = evaluate(model, data=type("obj", (), {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "y": y,
        "test_mask": test_mask_t,
    })())
    logger.info(
        "故障诊断 | 训练完成: final_val_acc=%.4f, test_acc=%.4f, test_f1=%.4f",
        history["val_acc"][-1] if history["val_acc"] else 0.0,
        eval_results["accuracy"],
        eval_results["f1"],
    )

    # ---- 保存 ----
    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_nodes": model.num_nodes,
            "num_relations": model.num_relations,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            # 图结构一并保存，推理时无需再连 ES
            "graph_data": dataset.to_data_with_types(),
            "node_to_idx": dataset.node_to_idx,
            "fault_nodes": dataset.fault_nodes,
            "idx_to_node": dataset.idx_to_node,
            # 完整数据集对象（含图遍历映射），推理时用于提取原因/措施/工具
            "dataset": dataset,
        }, os.path.join(save_model_path, "diagnosis_model.pth"))
        logger.info("故障诊断 | 模型与图结构已保存至: %s", save_model_path)

    return model


def run_fault_diagnosis_inference(
        model: FaultRGCN,
        dataset: KGTripleDataset,
        queries: list[str],
        top_k_symptoms: int = 5,
        top_k_faults: int = 3,
        symptom_relation: str = "表现为",
) -> list[DiagnosisResult]:
    """对多个文本查询执行故障诊断推理，返回每个查询的完整结果。"""
    model.eval()
    results: list[DiagnosisResult] = []
    for query in queries:
        results.append(infer_from_text(
            model=model,
            dataset=dataset,
            query_text=query,
            top_k_symptoms=top_k_symptoms,
            top_k_faults=top_k_faults,
            symptom_relation=symptom_relation,
        ))
    return results


# =============================================================================
# 模块 B：故障预测（R-GCN + TGN + 时序异构图）
# =============================================================================

# ---- 故障诊断子模型（与时序异构图配合使用） -------------------------------

class _TemporalRGCNDiag(torch.nn.Module):
    """在时序异构图上使用的轻量 R-GCN 健康状态分类模型。

    通过 `("time_slice", "has_health_state", "health_state")` 边传播
    健康状态节点信息到 time_slice 节点，输出 per-time_slice 的分类
    logits。
    """

    def __init__(self, feature_num: int, hidden_dim: int, health_num: int):
        super().__init__()
        self.feature_num = feature_num
        self.health_num = health_num
        self.conv1 = RGCNConv((4, feature_num), hidden_dim, num_relations=1)
        self.conv2 = RGCNConv((hidden_dim, hidden_dim), hidden_dim, num_relations=1)
        self.out_linear = Linear(hidden_dim, health_num)

    def forward(self, x_dict, edge_index_dict):
        x_slice = x_dict["time_slice"]
        x_health = x_dict["health_state"]
        edge = edge_index_dict["time_slice", "has_health_state", "health_state"]
        edge_rev = edge.flip(0)
        edge_type = torch.zeros(
            edge_rev.shape[1], dtype=torch.long, device=edge_rev.device,
        )
        h1 = self.conv1((x_health, x_slice), edge_rev, edge_type=edge_type)
        h1 = F.relu(h1)
        h2 = self.conv2((h1, h1), edge_rev, edge_type=edge_type)
        logits = self.out_linear(h2)
        return logits, h2


# ---- TGN 子模型工厂 ---------------------------------------------------------

def _build_tgn_model(
        feature_num: int,
        hidden_dim: int,
        num_time_slices: int,
) -> TGNOilTemperaturePredict:
    """构建一个与时序异构图配合的 TGN 油温趋势预测模型。"""
    return TGNOilTemperaturePredict(
        in_channels={
            "transformer": 3,
            "time_slice": feature_num,
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


# ---- 时序数据切分 -----------------------------------------------------------

def _split_hold_out_indices(
        total_num: int,
        hold_out_n: int,
        test_ratio: float,
        random_state: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从时序样本尾部预留 hold_out_n 个未知样本，剩余按比例划分 train/test。"""
    available = np.arange(total_num - hold_out_n)
    hold_out_idx = np.arange(total_num - hold_out_n, total_num)
    train_idx, test_idx = train_test_split(
        available, test_size=test_ratio, random_state=random_state,
    )
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.long).to(device)
    return to_tensor(train_idx), to_tensor(test_idx), to_tensor(hold_out_idx)


# ---- 故障预测训练入口 -------------------------------------------------------

def run_fault_prediction_training(
        csv_path: str,
        transformer_id: int = 0,
        hold_out_n: int = 10,
        test_ratio: float = 0.2,
        hidden_dim: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
        random_state: int = 42,
        save_dir: str | None = None,
) -> tuple[_TemporalRGCNDiag, TGNOilTemperaturePredict, torch.Tensor, torch.Tensor, HeteroData]:
    """加载 CSV → 构建时序异构图 → 联合训练 R-GCN（健康状态）+ TGN（油温+风险）。

    Returns
    -------
    (diag_model, tgn_model, test_idx, hold_out_idx, hetero_data)
    """
    device = _resolve_device(device) if isinstance(device, str) else device

    # ---- 1. 加载数据：KGTemporalDataset 会自动检测列、推断阈值 ----
    logger.info("故障预测 | 从 CSV 加载时序数据: %s", csv_path)
    kg_dataset = KGTemporalDataset(
        csv_path=csv_path,
        transformer_id=transformer_id,
        device=device,
    )
    hetero_data = kg_dataset.data
    total_num = kg_dataset.slice_num
    feature_num = kg_dataset.feature_num
    health_num = kg_dataset.health_num

    logger.info(
        "故障预测 | 时序切片=%d, 特征数=%d, 健康状态数=%d, 设备=%s",
        total_num, feature_num, health_num, device,
    )
    logger.info(
        "故障预测 | 自动识别: feature_list=%s",
        kg_dataset.feature_list,
    )

    # ---- 2. 数据切分 ----
    train_idx, test_idx, hold_out_idx = _split_hold_out_indices(
        total_num=total_num,
        hold_out_n=hold_out_n,
        test_ratio=test_ratio,
        random_state=random_state,
        device=device,
    )
    logger.info(
        "故障预测 | 切分: 训练=%d, 测试=%d, 预留(未知样本)=%d",
        train_idx.shape[0], test_idx.shape[0], hold_out_idx.shape[0],
    )

    # ---- 3. 初始化双模型 ----
    diag_model = _TemporalRGCNDiag(
        feature_num=feature_num, hidden_dim=hidden_dim, health_num=health_num,
    ).to(device)
    tgn_model = _build_tgn_model(
        feature_num=feature_num, hidden_dim=hidden_dim, num_time_slices=total_num,
    ).to(device)

    # ---- 4. 联合训练 ----
    logger.info(
        "故障预测 | 开始联合训练 R-GCN + TGN (epochs=%d, lr=%.0e, hidden=%d)",
        epochs, lr, hidden_dim,
    )
    diag_model, tgn_model, metrics = train_joint_rgcn_tgn(
        diag_model=diag_model,
        tgn_model=tgn_model,
        hetero_data=hetero_data,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=epochs,
        lr=lr,
        log_interval=10,
        hold_out_n=hold_out_n,
        verbose=True,
    )

    logger.info(
        "故障预测 | 训练完成: diag_acc=%.4f, ot_mae=%.4f, risk_acc=%.4f",
        metrics.get("diag_acc", float("nan")),
        metrics.get("ot_mae", float("nan")),
        metrics.get("risk_acc", float("nan")),
    )

    # ---- 5. 保存模型 ----
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            diag_model.state_dict(),
            os.path.join(save_dir, "temporal_diag_rgcn.pth"),
        )
        torch.save(
            tgn_model.state_dict(),
            os.path.join(save_dir, "temporal_tgn.pth"),
        )
        # 将时序异构图随模型一起持久化：推理时直接加载本文件，无需再读取 CSV
        torch.save(hetero_data.cpu(), os.path.join(save_dir, "temporal_kg_graph.pt"))
        logger.info(
            "故障预测 | 模型与知识图谱已保存至: %s/"
            "{temporal_diag_rgcn.pth, temporal_tgn.pth, temporal_kg_graph.pt}",
            save_dir,
        )

    return diag_model, tgn_model, test_idx, hold_out_idx, hetero_data


# ---- 故障预测推理打印 -------------------------------------------------------

def print_temporal_inference(
        hetero_data,
        diag_model,
        tgn_model,
        slice_idx: int,
        feature_list: list[str],
        health_mapping: dict[int, str],
) -> None:
    """对指定时序切片执行「故障诊断 + 趋势预测」的串联推理，并完整打印。"""
    print("=" * 100)
    print(
        f"【变压器时序推理】切片ID：{slice_idx} | "
        f"时间：{hetero_data['time_slice'].date_str[slice_idx]}"
    )
    print("=" * 100)

    # --- 1. 切片基础信息（真实值） ---
    slice_feat = hetero_data["time_slice"].x_raw[slice_idx].cpu().numpy()
    true_health = hetero_data["time_slice"].y_health[slice_idx].item()
    true_future_ot = hetero_data["time_slice"].y_future_ot[slice_idx].item()
    print(f"\n[1] 切片基础运行信息：")
    print(f"  真实健康状态：{health_mapping.get(true_health, f'状态{true_health}')}")
    print(f"  未来3步真实油温：{true_future_ot:.4f}℃")
    print(f"  核心运行特征：")
    for i, feat_name in enumerate(feature_list):
        print(f"    {feat_name:6s}: {slice_feat[i]:.4f}")

    # --- 2. R-GCN 故障诊断 ---
    with torch.no_grad():
        health_logits, _ = diag_model(
            hetero_data.x_dict, hetero_data.edge_index_dict,
        )
        pred_health_logits = health_logits[slice_idx]
        pred_health = torch.argmax(pred_health_logits).item()
        pred_health_prob = F.softmax(pred_health_logits, dim=0).cpu().numpy()

    print(f"\n[2] R-GCN 知识图谱故障诊断结果：")
    print(f"  预测健康状态：{health_mapping.get(pred_health, f'状态{pred_health}')}")
    print(f"  各类别预测概率：")
    for label_id, state_name in health_mapping.items():
        print(f"    {state_name:8s}: {pred_health_prob[label_id]:.2%}")

    # --- 3. TGN 油温趋势与故障风险预测 ---
    with torch.no_grad():
        future_ot_pred, fault_risk_pred, _ = tgn_model(hetero_data)
        ot_mean_val = hetero_data["time_slice"].ot_mean.item()
        ot_std_val = hetero_data["time_slice"].ot_std.item()
        pred_future_ot = (
            future_ot_pred[slice_idx].item() * ot_std_val + ot_mean_val
        )
        pred_fault_risk = fault_risk_pred[slice_idx].item()

    risk_level = (
        "低风险" if pred_fault_risk < 0.3
        else "中风险" if pred_fault_risk < 0.7
        else "高风险"
    )
    print(f"\n[3] TGN 时序图模型趋势预测结果：")
    print(
        f"  未来3步预测油温：{pred_future_ot:.4f}℃，"
        f"真实油温：{true_future_ot:.4f}℃，"
        f"误差：{abs(pred_future_ot - true_future_ot):.4f}℃"
    )
    print(f"  未来故障发生概率：{pred_fault_risk:.2%}，风险等级：{risk_level}")

    # --- 4. 运维建议 ---
    print(f"\n[4] 最终运维建议：")
    if pred_health == 0 and pred_fault_risk < 0.3:
        print("  ✅ 设备运行正常，维持常规巡检周期")
    elif pred_health == 1 or pred_fault_risk >= 0.3:
        print("  ⚠️  设备轻微异常，缩短巡检周期，密切关注油温与负载变化")
    elif pred_health == 2 or pred_fault_risk >= 0.7:
        print("  ⚠️  设备严重过热，立即安排停电检修，检查绝缘与散热系统")
    elif pred_health == 3:
        print("  ❌  设备过载故障，立即降低负载，紧急停机检查")

    print("\n" + "=" * 100 + " 推理结束 " + "=" * 100 + "\n")


# =============================================================================
# 命令行参数
# =============================================================================

def _parse_args(
    train_cfg: TrainingConfig,
    diag_cfg: DiagnosisTrainingConfig,
    pred_cfg: PredictionTrainingConfig,
) -> argparse.Namespace:
    """从命令行解析参数；默认值全部来自 ``config/config.yaml``。"""
    parser = argparse.ArgumentParser(
        description="知识图谱统一训练：故障诊断 (R-GCN) + 故障预测 (R-GCN+TGN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 默认同时运行两个流程（所有参数来自 config/config.yaml）
  python train.py

  # 仅故障诊断（命令行可覆盖 YAML 默认值）
  python train.py --mode diagnosis --diag-epochs 500 --diag-hidden-dim 128
  python train.py --mode diagnosis --query "加速迟缓" --device cuda

  # 仅故障预测（变压器时序数据）
  python train.py --mode prediction --csv-path /path/to/ETTh1.csv
  python train.py --mode prediction --pred-hold-out 10 --pred-epochs 200 --pred-lr 1e-3
""",
    )

    # ---- 流程选择 ----
    parser.add_argument("--mode", type=str, default=train_cfg.mode,
                        choices=["both", "diagnosis", "prediction"],
                        help=f"训练模式 (默认来自 config.yaml: {train_cfg.mode})")
    parser.add_argument("--seed", type=int, default=train_cfg.seed,
                        help=f"随机种子 (默认来自 config.yaml: {train_cfg.seed})")
    parser.add_argument("--device", type=str, default=train_cfg.device,
                        help=f"训练设备，auto= CUDA 优先 (默认来自 config.yaml: {train_cfg.device})")

    # ---- 故障诊断参数（ES 三元组知识图谱） ----
    parser.add_argument("--diag-epochs", type=int, default=diag_cfg.epochs,
                        help=f"[诊断] 训练轮数 (默认来自 config.yaml: {diag_cfg.epochs})")
    parser.add_argument("--diag-hidden-dim", type=int, default=diag_cfg.hidden_dim,
                        help=f"[诊断] R-GCN 隐藏层维度 (默认来自 config.yaml: {diag_cfg.hidden_dim})")
    parser.add_argument("--diag-num-layers", type=int, default=diag_cfg.num_layers,
                        help=f"[诊断] R-GCN 层数 (默认来自 config.yaml: {diag_cfg.num_layers})")
    parser.add_argument("--diag-dropout", type=float, default=diag_cfg.dropout,
                        help=f"[诊断] Dropout 比例 (默认来自 config.yaml: {diag_cfg.dropout})")
    parser.add_argument("--diag-lr", type=float, default=diag_cfg.lr,
                        help=f"[诊断] 学习率 (默认来自 config.yaml: {diag_cfg.lr})")
    parser.add_argument("--diag-weight-decay", type=float, default=diag_cfg.weight_decay,
                        help=f"[诊断] L2 正则化 (默认来自 config.yaml: {diag_cfg.weight_decay})")
    parser.add_argument("--diag-save", type=str, default=diag_cfg.save_model_path,
                        help="[诊断] 模型保存路径 (如 ./models/rgcn_fault_diag.pt)")

    # ---- 故障预测参数（时序异构图 + CSV） ----
    parser.add_argument("--csv-path", type=str, default=pred_cfg.csv_path,
                        help="[预测] 变压器时序 CSV 数据文件路径 (默认来自 config.yaml)")
    parser.add_argument("--pred-transformer-id", type=int, default=pred_cfg.transformer_id,
                        help=f"[预测] transformer 节点 ID (默认来自 config.yaml: {pred_cfg.transformer_id})")
    parser.add_argument("--pred-hold-out", type=int, default=pred_cfg.hold_out,
                        help=f"[预测] 从数据集尾部预留的未知样本数 (默认来自 config.yaml: {pred_cfg.hold_out})")
    parser.add_argument("--pred-test-ratio", type=float, default=pred_cfg.test_ratio,
                        help=f"[预测] 测试集比例 (默认来自 config.yaml: {pred_cfg.test_ratio})")
    parser.add_argument("--pred-epochs", type=int, default=pred_cfg.epochs,
                        help=f"[预测] 训练轮数 (默认来自 config.yaml: {pred_cfg.epochs})")
    parser.add_argument("--pred-lr", type=float, default=pred_cfg.lr,
                        help=f"[预测] 学习率 (默认来自 config.yaml: {pred_cfg.lr})")
    parser.add_argument("--pred-hidden-dim", type=int, default=pred_cfg.hidden_dim,
                        help=f"[预测] 隐藏层维度 (默认来自 config.yaml: {pred_cfg.hidden_dim})")
    parser.add_argument("--pred-save-dir", type=str, default=pred_cfg.save_model_path,
                        help="[预测] 模型保存目录 (如 ./models/temporal)")

    # ---- 推理参数 ----
    parser.add_argument("--no-infer", action="store_true", default=True,
                        help="跳过推理验证阶段")
    parser.add_argument("--symptom-relation", type=str, default="表现为",
                        help="[诊断] 症状-故障关系名称 (默认: 表现为)")
    parser.add_argument("--query", type=str, nargs="+",
                        default=["发动机怠速不稳", "加速迟缓", "冒白烟"],
                        help="[诊断] 推理测试查询文本，支持多个")
    parser.add_argument("--top-k", type=int, default=3,
                        help="[诊断] 故障定位 Top-K (默认: 3)")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning("忽略未识别的参数: %s", unknown)
    return args


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    train_cfg = TrainingConfig.default()
    diag_cfg = DiagnosisTrainingConfig.default()
    pred_cfg = PredictionTrainingConfig.default()
    logger.info(
        "TrainingConfig 已从 config/config.yaml 加载 | mode=%s | seed=%d | device=%s",
        train_cfg.mode, train_cfg.seed, train_cfg.device,
    )

    args = _parse_args(train_cfg, diag_cfg, pred_cfg)
    set_seed(args.seed)
    device = _resolve_device(args.device)
    logger.info("统一训练脚本启动 | mode=%s | device=%s | seed=%d",
                args.mode, device, args.seed)

    es_config = ESConfig.default()
    logger.info("ESConfig 已从 config/config.yaml 加载 (host=%s)", es_config.host)

    # ========================================================================
    # 流程 A：故障诊断（R-GCN + 三元组知识图谱）
    # ========================================================================
    if args.mode in ("both", "diagnosis"):
        print("\n" + "=" * 64)
        print("🔎 [1/2] 故障诊断模型：知识图谱 R-GCN 训练")
        print("=" * 64)

        print("\n📦 加载 “故障诊断” 知识图谱...")
        diag_dataset = KGTripleDataset(es_config=es_config)
        print(f"  数据源:   {diag_dataset._data_source}")
        print(f"  实体数:   {diag_dataset.num_nodes}")
        print(f"  关系数:   {diag_dataset.num_original_relations}")
        print(f"  三元组数: {len(diag_dataset.triples)}")
        print(f"  整图训练节点数: {len(diag_dataset.fault_nodes)}")
        print(f"  故障类别:   {diag_dataset.get_fault_category_nodes()}")

        print(f"\n🚀 开始训练 R-GCN... (device={device})")
        diag_model = run_fault_diagnosis_training(
            dataset=diag_dataset,
            hidden_dim=args.diag_hidden_dim,
            num_layers=args.diag_num_layers,
            dropout=args.diag_dropout,
            lr=args.diag_lr,
            epochs=args.diag_epochs,
            weight_decay=args.diag_weight_decay,
            device=device,
            save_model_path=args.diag_save,
        )

        if not args.no_infer:
            print(f"\n📋 推理验证")
            print("-" * 64)
            diag_results = run_fault_diagnosis_inference(
                model=diag_model,
                dataset=diag_dataset,
                queries=args.query,
                top_k_symptoms=5,
                top_k_faults=args.top_k,
                symptom_relation=args.symptom_relation,
            )
            for q, r in zip(args.query, diag_results):
                print(f"\n🔍 输入: “{q}”")
                print(format_result_compact(r))
            print(f"\n📖 详细诊断报告 (首个查询: “{args.query[0]}”)")
            print(format_result(diag_results[0]))

    # ========================================================================
    # 流程 B：故障预测（R-GCN + TGN + 时序异构图）
    # ========================================================================
    if args.mode in ("both", "prediction"):
        if args.csv_path is None:
            raise ValueError(
                "mode=prediction/both 时必须指定 --csv-path "
                "(变压器时序 CSV 数据文件路径)"
            )
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {args.csv_path}")

        print("\n" + "=" * 64)
        print("🌡️  [2/2] 故障预测模型：时序异构图 R-GCN + TGN 联合训练")
        print("=" * 64)

        print(f"\n📦 加载 CSV: {args.csv_path}")
        print("   KGTemporalDataset 会自动检测列名、推断阈值、构建异构图")

        pred_diag_model, tgn_model, test_idx, hold_out_idx, hetero_data = (
            run_fault_prediction_training(
                csv_path=args.csv_path,
                transformer_id=args.pred_transformer_id,
                hold_out_n=args.pred_hold_out,
                test_ratio=args.pred_test_ratio,
                hidden_dim=args.pred_hidden_dim,
                epochs=args.pred_epochs,
                lr=args.pred_lr,
                device=device,
                save_dir=args.pred_save_dir,
            )
        )

        # --- 推理展示：测试集样本 + 未知样本 ---
        if not args.no_infer:
            # 重新加载一次数据集以获取 feature_list 与 health_mapping 的原始文本
            # （构建异构图时这些信息未写入 HeteroData，但读取 CSV 是轻量操作）。
            tmp = KGTemporalDataset(csv_path=args.csv_path)
            feature_list = list(tmp.feature_list)
            health_mapping = dict(tmp.health_mapping)
            del tmp

            print(f"\n📋 推理验证: feature_list={feature_list}")
            print("=" * 64)

            print(f"\n【模式 A】推理测试集样本（训练可见、未参与梯度更新）")
            print("-" * 64)
            for i in range(min(3, test_idx.shape[0])):
                sid = test_idx[i].item()
                print_temporal_inference(
                    hetero_data=hetero_data,
                    diag_model=pred_diag_model,
                    tgn_model=tgn_model,
                    slice_idx=sid,
                    feature_list=feature_list,
                    health_mapping=health_mapping,
                )

            print(f"\n【模式 B】推理未知样本（{hold_out_idx.shape[0]} 个，"
                  f"训练/测试均未使用）")
            print("-" * 64)
            for i in range(min(3, hold_out_idx.shape[0])):
                sid = hold_out_idx[i].item()
                print_temporal_inference(
                    hetero_data=hetero_data,
                    diag_model=pred_diag_model,
                    tgn_model=tgn_model,
                    slice_idx=sid,
                    feature_list=feature_list,
                    health_mapping=health_mapping,
                )

    # ========================================================================
    # 结束
    # ========================================================================
    print("\n" + "=" * 64)
    print("✅ 全部流程已完成")
    print("=" * 64)


if __name__ == "__main__":
    main()
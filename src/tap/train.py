"""时序属性预测模型训练脚本。

基于变压器时序 CSV 数据，构建 transformer / time_slice / health_state /
feature_indicator 的异构图知识图谱，联合训练 R-GCN（健康状态分类）+
TGN（油温趋势 + 故障风险预测）。对应原统一 ``src/fault/train.py --mode
prediction`` 的逻辑。

用法:
    # 时序属性预测 (使用 ETTh1/ETTh2 等变压器时序数据)
    python -m src.tap.train --csv-path /path/to/ETTh1.csv
    python -m src.tap.train --csv-path /path/to/ETTh1.csv --hold-out 10 --epochs 200

    # 也可作为脚本直接运行（已自动注入项目根到 sys.path）：
    python src/tap/train.py --csv-path /path/to/ETTh1.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, Linear

# 项目根目录加入 sys.path，使以脚本方式运行本文件时 `import src...` 等
# 包内绝对导入可解析（`python src/tap/train.py` 时 cwd 不自动入
# path，故显式注入；`python -m src.tap.train` 时 cwd 已在 path，此步无害）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.config import (
    TAPTrainingConfig,
    TrainingConfig,
    logger,
)
from src.dataset.temporal_dataset import KGTemporalDataset
from src.model import TGNModel
from src.model.training import train_joint_rgcn_tgn

# =============================================================================
# 通用工具
# =============================================================================


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(raw: str) -> torch.device:
    """根据字符串解析训练设备，``auto`` 表示 CUDA 优先。"""
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


# =============================================================================
# 时序属性预测（R-GCN + TGN + 时序异构图）
# =============================================================================

# ---- 静态链接预测子模型（与时序异构图配合使用） -------------------------------


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
) -> TGNModel:
    """构建一个与时序异构图配合的 TGN 油温属性预测模型。"""
    return TGNModel(
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


# ---- 时序属性预测训练入口 -------------------------------------------------------


def run_tap_training(
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
        chunk_size: int = 0,
        batch_size: int = 0,
) -> tuple[_TemporalRGCNDiag, TGNModel, torch.Tensor, torch.Tensor, HeteroData]:
    """加载 CSV → 构建时序异构图 → 联合训练 R-GCN（健康状态）+ TGN（油温+风险）。

    Returns
    -------
    (diag_model, tgn_model, test_idx, hold_out_idx, hetero_data)
    """
    device = resolve_device(device) if isinstance(device, str) else device

    # ---- 1. 加载数据：KGTemporalDataset 会自动检测列、推断阈值 ----
    logger.info("时序属性预测 | 从 CSV 加载时序数据: %s", csv_path)
    if chunk_size and chunk_size > 0:
        logger.info(
            "时序属性预测 | 启用分块流式读取 (chunk_size=%d)，适合超大 CSV，峰值内存与块大小成正比",
            chunk_size,
        )
    kg_dataset = KGTemporalDataset(
        csv_path=csv_path,
        transformer_id=transformer_id,
        device=device,
        chunk_size=chunk_size,
    )
    hetero_data = kg_dataset.data
    total_num = kg_dataset.slice_num
    feature_num = kg_dataset.feature_num
    health_num = kg_dataset.health_num

    logger.info(
        "时序属性预测 | 时序切片=%d, 特征数=%d, 健康状态数=%d, 设备=%s",
        total_num, feature_num, health_num, device,
    )
    logger.info(
        "时序属性预测 | 自动识别: feature_list=%s",
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
        "时序属性预测 | 切分: 训练=%d, 测试=%d, 预留(未知样本)=%d",
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
        "时序属性预测 | 开始联合训练 R-GCN + TGN (epochs=%d, lr=%.0e, hidden=%d)",
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
        batch_size=batch_size,
        verbose=True,
    )

    logger.info(
        "时序属性预测 | 训练完成: diag_acc=%.4f, ot_mae=%.4f, risk_acc=%.4f",
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
            "时序属性预测 | 模型与知识图谱已保存至: %s/"
            "{temporal_diag_rgcn.pth, temporal_tgn.pth, temporal_kg_graph.pt}",
            save_dir,
        )

    return diag_model, tgn_model, test_idx, hold_out_idx, hetero_data


# ---- 时序属性预测推理打印 -------------------------------------------------------


def print_temporal_inference(
        hetero_data,
        diag_model,
        tgn_model,
        slice_idx: int,
        feature_list: list[str],
        health_mapping: dict[int, str],
) -> None:
    """对指定时序切片执行「静态链接预测 + 属性预测」的串联推理，并完整打印。"""
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

    # --- 2. R-GCN 静态链接预测 ---
    with torch.no_grad():
        health_logits, _ = diag_model(
            hetero_data.x_dict, hetero_data.edge_index_dict,
        )
        pred_health_logits = health_logits[slice_idx]
        pred_health = torch.argmax(pred_health_logits).item()
        pred_health_prob = F.softmax(pred_health_logits, dim=0).cpu().numpy()

    print(f"\n[2] R-GCN 知识图谱静态链接预测结果：")
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
    print(f"\n[3] TGN 时序图模型属性预测结果：")
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
    tap_cfg: TAPTrainingConfig,
) -> argparse.Namespace:
    """从命令行解析参数；默认值全部来自 ``config/config.yaml``。"""
    parser = argparse.ArgumentParser(
        description="时序属性预测模型训练：时序异构图 R-GCN + TGN 联合训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 默认运行预测流程（所有参数来自 config/config.yaml）
  python -m src.tap.train --csv-path /path/to/ETTh1.csv

  # 覆盖 YAML 默认值
  python -m src.tap.train --csv-path /path/to/ETTh1.csv --hold-out 10 --epochs 200 --lr 1e-3
""",
    )

    parser.add_argument("--seed", type=int, default=train_cfg.seed,
                        help=f"随机种子 (默认来自 config.yaml: {train_cfg.seed})")
    parser.add_argument("--device", type=str, default=train_cfg.device,
                        help=f"训练设备，auto= CUDA 优先 (默认来自 config.yaml: {train_cfg.device})")

    # ---- 时序属性预测参数（时序异构图 + CSV） ----
    parser.add_argument("--csv-path", type=str, default=tap_cfg.csv_path,
                        help="[预测] 变压器时序 CSV 数据文件路径 (默认来自 config.yaml)")
    parser.add_argument("--transformer-id", type=int, default=tap_cfg.transformer_id,
                        help=f"[预测] transformer 节点 ID (默认来自 config.yaml: {tap_cfg.transformer_id})")
    parser.add_argument("--hold-out", type=int, default=tap_cfg.hold_out,
                        help=f"[预测] 从数据集尾部预留的未知样本数 (默认来自 config.yaml: {tap_cfg.hold_out})")
    parser.add_argument("--test-ratio", type=float, default=tap_cfg.test_ratio,
                        help=f"[预测] 测试集比例 (默认来自 config.yaml: {tap_cfg.test_ratio})")
    parser.add_argument("--epochs", type=int, default=tap_cfg.epochs,
                        help=f"[预测] 训练轮数 (默认来自 config.yaml: {tap_cfg.epochs})")
    parser.add_argument("--lr", type=float, default=tap_cfg.lr,
                        help=f"[预测] 学习率 (默认来自 config.yaml: {tap_cfg.lr})")
    parser.add_argument("--hidden-dim", type=int, default=tap_cfg.hidden_dim,
                        help=f"[预测] 隐藏层维度 (默认来自 config.yaml: {tap_cfg.hidden_dim})")
    parser.add_argument("--save-dir", type=str, default=tap_cfg.save_model_path,
                        help="[预测] 模型保存目录 (如 ./models/temporal)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="[预测] CSV 分块流式读取的块大小(行数)；>0 时启用，"
                             "适合超大 CSV，峰值内存与块大小成正比 (默认 0=全量读取)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="[预测] 训练小批量大小；>0 且 < 训练样本数时启用梯度累积式"
                             "小批量训练 (默认 0=全量)")

    parser.add_argument("--no-infer", action="store_true", default=True,
                        help="跳过推理验证阶段")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning("忽略未识别的参数: %s", unknown)
    return args


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    train_cfg = TrainingConfig.default()
    tap_cfg = TAPTrainingConfig.default()
    logger.info(
        "TrainingConfig 已从 config/config.yaml 加载 | seed=%d | device=%s",
        train_cfg.seed, train_cfg.device,
    )

    args = _parse_args(train_cfg, tap_cfg)
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("时序属性预测训练脚本启动 | device=%s | seed=%d", device, args.seed)

    if args.csv_path is None:
        raise ValueError(
            "必须指定 --csv-path (变压器时序 CSV 数据文件路径)"
        )
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {args.csv_path}")

    print("\n" + "=" * 64)
    print("🌡️  时序属性预测模型：时序异构图 R-GCN + TGN 联合训练")
    print("=" * 64)

    print(f"\n📦 加载 CSV: {args.csv_path}")
    print("   KGTemporalDataset 会自动检测列名、推断阈值、构建异构图")

    pred_diag_model, tgn_model, test_idx, hold_out_idx, hetero_data = (
        run_tap_training(
            csv_path=args.csv_path,
            transformer_id=args.transformer_id,
            hold_out_n=args.hold_out,
            test_ratio=args.test_ratio,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=args.save_dir,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
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

    print("\n" + "=" * 64)
    print("✅ 时序属性预测训练流程已完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

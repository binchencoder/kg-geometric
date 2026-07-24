"""静态链接预测模型训练脚本。

基于 ES 中的三元组知识图谱，使用 R-GCN 训练节点嵌入，支持自然语言查询
→ 故障推理。对应原统一 ``src/fault/train.py --mode diagnosis`` 的逻辑。

用法:
    # 静态链接预测 (默认使用 ES 中的 "车辆静态链接预测" 图谱)
    python -m src.slp.train
    python -m src.slp.train --epochs 500 --hidden-dim 128
    python -m src.slp.train --query "加速迟缓" --device cuda

    # 也可作为脚本直接运行（已自动注入项目根到 sys.path）：
    python src/slp/train.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

# 项目根目录加入 sys.path，使以脚本方式运行本文件时 `import src...` 等
# 包内绝对导入可解析（`python src/slp/train.py` 时 cwd 不自动入 path，
# 故显式注入；`python -m src.slp.train` 时 cwd 已在 path，此步无害）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.config import (
    SLPTrainingConfig,
    ESConfig,
    TrainingConfig,
    logger,
)
from src.dataset.triple_dataset import KGTripleDataset
from src.model.rgcn import FaultRGCN
from src.model.training import (
    evaluate,
    split_masks,
    train_rgcn,
)
from src.pipeline.inference import (
    SLPResult,
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


def resolve_device(raw: str) -> torch.device:
    """根据字符串解析训练设备，``auto`` 表示 CUDA 优先。"""
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


# =============================================================================
# 静态链接预测（R-GCN + 三元组知识图谱）
# =============================================================================


def run_slp_training(
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
    """训练 R-GCN 静态链接预测模型并返回训练好的模型。"""
    device = resolve_device(device) if isinstance(device, str) else device

    # ---- 划分 7:1:2 训练/验证/测试集 ----
    train_mask, val_mask, test_mask = split_masks(
        dataset.num_nodes,
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    )
    logger.info(
        "静态链接预测 | 数据划分: 训练=%d, 验证=%d, 测试=%d (总计 %d 节点)",
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
        "静态链接预测 | R-GCN: nodes=%d, relations=%d, hidden=%d, layers=%d, params=%d",
        model.num_nodes, model.num_relations, hidden_dim, num_layers,
        sum(p.numel() for p in model.parameters()),
    )

    # ---- 训练 ----
    logger.info(
        "静态链接预测 | 开始训练 (lr=%.0e, weight_decay=%.0e, epochs=%d, dropout=%.1f)",
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
        "静态链接预测 | 训练完成: final_val_acc=%.4f, test_acc=%.4f, test_f1=%.4f",
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
        }, os.path.join(save_model_path, "slp_model.pth"))
        logger.info("静态链接预测 | 模型与图结构已保存至: %s", save_model_path)

    return model


def run_fault_diagnosis_inference(
        model: FaultRGCN,
        dataset: KGTripleDataset,
        queries: list[str],
        top_k: int = 5,
        relation_mapping: "Optional[Dict[str, str]]" = None,
        device: str = "cpu",
) -> list[SLPResult]:
    """对多个文本查询执行静态链接预测推理，返回每个查询的完整结果。"""
    model.eval()
    results: list[SLPResult] = []
    for query in queries:
        results.append(infer_from_text(
            model=model,
            dataset=dataset,
            query_text=query,
            top_k=top_k,
            relation_mapping=relation_mapping,
            device=device,
        ))
    return results


# =============================================================================
# 命令行参数
# =============================================================================


def _parse_args(
    train_cfg: TrainingConfig,
    diag_cfg: SLPTrainingConfig,
) -> argparse.Namespace:
    """从命令行解析参数；默认值全部来自 ``config/config.yaml``。"""
    parser = argparse.ArgumentParser(
        description="静态链接预测模型训练：知识图谱 R-GCN (三元组 KG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 默认运行诊断流程（所有参数来自 config/config.yaml）
  python -m src.slp.train

  # 覆盖 YAML 默认值
  python -m src.slp.train --epochs 500 --hidden-dim 128
  python -m src.slp.train --query "加速迟缓" --device cuda
""",
    )

    parser.add_argument("--seed", type=int, default=train_cfg.seed,
                        help=f"随机种子 (默认来自 config.yaml: {train_cfg.seed})")
    parser.add_argument("--device", type=str, default=train_cfg.device,
                        help=f"训练设备，auto= CUDA 优先 (默认来自 config.yaml: {train_cfg.device})")

    # ---- 静态链接预测参数（ES 三元组知识图谱） ----
    parser.add_argument("--epochs", type=int, default=diag_cfg.epochs,
                        help=f"[诊断] 训练轮数 (默认来自 config.yaml: {diag_cfg.epochs})")
    parser.add_argument("--hidden-dim", type=int, default=diag_cfg.hidden_dim,
                        help=f"[诊断] R-GCN 隐藏层维度 (默认来自 config.yaml: {diag_cfg.hidden_dim})")
    parser.add_argument("--num-layers", type=int, default=diag_cfg.num_layers,
                        help=f"[诊断] R-GCN 层数 (默认来自 config.yaml: {diag_cfg.num_layers})")
    parser.add_argument("--dropout", type=float, default=diag_cfg.dropout,
                        help=f"[诊断] Dropout 比例 (默认来自 config.yaml: {diag_cfg.dropout})")
    parser.add_argument("--lr", type=float, default=diag_cfg.lr,
                        help=f"[诊断] 学习率 (默认来自 config.yaml: {diag_cfg.lr})")
    parser.add_argument("--weight-decay", type=float, default=diag_cfg.weight_decay,
                        help=f"[诊断] L2 正则化 (默认来自 config.yaml: {diag_cfg.weight_decay})")
    parser.add_argument("--save", type=str, default=diag_cfg.save_model_path,
                        help="[诊断] 模型保存路径 (如 ./models/rgcn_fault_diag.pt)")

    # ---- 推理参数 ----
    parser.add_argument("--no-infer", action="store_true", default=True,
                        help="跳过推理验证阶段")
    parser.add_argument("--query", type=str, nargs="+",
                        default=["发动机怠速不稳", "加速迟缓", "冒白烟"],
                        help="[诊断] 推理测试查询文本，支持多个")
    parser.add_argument("--top-k", type=int, default=3,
                        help="[诊断] 静态链接预测 Top-K (默认: 3)")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning("忽略未识别的参数: %s", unknown)
    return args


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    train_cfg = TrainingConfig.default()
    diag_cfg = SLPTrainingConfig.default()
    logger.info(
        "TrainingConfig 已从 config/config.yaml 加载 | seed=%d | device=%s",
        train_cfg.seed, train_cfg.device,
    )

    args = _parse_args(train_cfg, diag_cfg)
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("静态链接预测训练脚本启动 | device=%s | seed=%d", device, args.seed)

    es_config = ESConfig.default()
    logger.info("ESConfig 已从 config/config.yaml 加载 (host=%s)", es_config.host)

    print("\n" + "=" * 64)
    print("🔎 静态链接预测模型：知识图谱 R-GCN 训练")
    print("=" * 64)

    print("\n📦 加载 “静态链接预测” 知识图谱...")
    diag_dataset = KGTripleDataset(es_config=es_config)
    print(f"  数据源:   {diag_dataset._data_source}")
    print(f"  实体数:   {diag_dataset.num_nodes}")
    print(f"  关系数:   {diag_dataset.num_original_relations}")
    print(f"  三元组数: {len(diag_dataset.triples)}")
    print(f"  整图训练节点数: {len(diag_dataset.fault_nodes)}")
    print(f"  故障类别:   {diag_dataset.get_fault_category_nodes()}")

    print(f"\n🚀 开始训练 R-GCN... (device={device})")
    diag_model = run_slp_training(
        dataset=diag_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=device,
        save_model_path=args.save,
    )

    if not args.no_infer:
        print(f"\n📋 推理验证")
        print("-" * 64)
        diag_results = run_fault_diagnosis_inference(
            model=diag_model,
            dataset=diag_dataset,
            queries=args.query,
            top_k=5,
            device=device,
        )
        for q, r in zip(args.query, diag_results):
            print(f"\n🔍 输入: “{q}”")
            print(format_result_compact(r))
        print(f"\n📖 详细诊断报告 (首个查询: “{args.query[0]}”)")
        print(format_result(diag_results[0]))

    print("\n" + "=" * 64)
    print("✅ 静态链接预测训练流程已完成")
    print("=" * 64)


if __name__ == "__main__":
    main()

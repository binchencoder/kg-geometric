"""知识图谱故障诊断 —— 基于 R-GCN 的症状到维修方案完整推理。

模型使用 R-GCN 学习多关系感知的节点嵌入，支持四阶段推理：
1. 语义匹配：症状文本 → 图中症状节点
2. 故障定位：沿"表现为"反向定位故障类别
3. 答案生成：正向提取原因/措施/工具
4. 结果组装：结构化诊断报告

运行方式：
    python demo/kg_fault_diagnosis.py

    # 指定自定义症状
    python demo/kg_fault_diagnosis.py --query "加速迟缓"
    python demo/kg_fault_diagnosis.py --query "冒白烟" --top-k 5
    python demo/kg_fault_diagnosis.py --query "抖动厉害"
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from src.model.rgcn import FaultRGCN
from src.model.gcn import FaultGCN
from src.model.training import split_masks, train_rgcn, train, evaluate
from src.dataset.kg_fault_demo import KGFaultDataset
from src.pipeline.inference import (
    infer_from_text,
    format_result,
    format_result_compact,
    DiagnosisResult,
)
from src.pipeline.diagnosis import topk_fault_diagnosis, print_topk_diagnosis
from src.core.config import logger

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 保持 __all__ 向后兼容 predict.py
__all__ = [
    "FaultGCN",
    "FaultRGCN",
    "KGFaultDataset",
    "split_masks",
    "train",
    "train_rgcn",
    "evaluate",
    "infer_from_text",
    "format_result",
    "format_result_compact",
    "topk_fault_diagnosis",
    "print_topk_diagnosis",
]


def run_training(
        dataset: KGFaultDataset,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 5e-4,
        epochs: int = 300,
        weight_decay: float = 5e-4,
        device: str = "cpu",
) -> FaultRGCN:
    """训练 R-GCN 模型并返回训练好的模型。

    Parameters
    ----------
    dataset : KGFaultDataset
        知识图谱数据集（含 edge_index, edge_type, y）。
    hidden_dim : int
        隐藏层/嵌入维度。
    num_layers : int
        R-GCN 层数。
    dropout : float
        Dropout 比例。
    lr : float
        学习率。
    epochs : int
        最大训练轮数。
    weight_decay : float
        L2 正则化。
    device : str
        训练设备。

    Returns
    -------
    FaultRGCN
        训练好的模型。
    """
    # 划分 7:1:2 训练/验证/测试集
    train_mask, val_mask, test_mask = split_masks(
        dataset.num_nodes,
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    )

    logger.info(
        "数据集划分: 训练=%d, 验证=%d, 测试=%d (总计 %d 节点)",
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
        dataset.num_nodes,
    )

    # 构建 R-GCN 模型
    model = FaultRGCN(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_bases=None,  # 不使用 basis decomposition（关系类型少时不需要）
    ).to(device)

    edge_index = dataset.edge_index.to(device)
    edge_type = dataset.edge_type.to(device)
    y = dataset.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    logger.info(
        "R-GCN 模型: nodes=%d, relations=%d, hidden=%d, layers=%d, "
        "params=%d",
        model.num_nodes, model.num_relations, hidden_dim, num_layers,
        sum(p.numel() for p in model.parameters()),
    )

    # 训练
    logger.info(
        "开始训练 (lr=%.0e, weight_decay=%.0e, epochs=%d, dropout=%.1f)",
        lr, weight_decay, epochs, dropout,
    )
    history = train_rgcn(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=30,
        verbose=True,
    )

    # 最终评估
    dataset.test_mask = test_mask.cpu()
    dataset.y_full = y.cpu()
    eval_results = evaluate(model, data=type('obj', (), {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'y': y,
        'test_mask': test_mask,
    })())

    logger.info(
        "训练完成: final_val_acc=%.4f, test_acc=%.4f, test_f1=%.4f",
        history['val_acc'][-1] if history['val_acc'] else 0.0,
        eval_results['accuracy'],
        eval_results['f1'],
    )

    return model


def run_inference(
        model: FaultRGCN,
        dataset: KGFaultDataset,
        query: str,
        top_k_symptoms: int = 5,
        top_k_faults: int = 3,
) -> DiagnosisResult:
    """执行四阶段故障诊断推理。

    Parameters
    ----------
    model : FaultRGCN
        训练好的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    query : str
        用户输入的症状描述文本。
    top_k_symptoms : int
        语义匹配的候选症状数。
    top_k_faults : int
        故障定位的候选故障数。

    Returns
    -------
    DiagnosisResult
        完整诊断结果。
    """
    model.eval()
    return infer_from_text(
        model=model,
        dataset=dataset,
        query_text=query,
        top_k_symptoms=top_k_symptoms,
        top_k_faults=top_k_faults,
    )


def main() -> None:
    """主入口：训练 → 推理演示。"""
    parser = argparse.ArgumentParser(
        description="R-GCN 知识图谱故障诊断 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo/kg_fault_diagnosis.py
  python demo/kg_fault_diagnosis.py --query "加速迟缓"
  python demo/kg_fault_diagnosis.py --query "抖动" --top-k 5
  python demo/kg_fault_diagnosis.py --query "水温表" --epochs 500
        """,
    )
    parser.add_argument("--query", default="手刹拉不动",
                        help="输入症状文本（默认使用示例症状）")
    parser.add_argument("--top-k", type=int, default=3,
                        help="故障定位 Top-K (默认: 3)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮数 (默认: 300)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="隐藏层维度 (默认: 64)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率 (默认: 5e-4)")
    parser.add_argument("--device", default="cpu",
                        help="设备 (默认: cpu)")
    args = parser.parse_args()

    # ---- 阶段 1: 加载数据集 ----
    print("\n📦 加载车辆故障知识图谱...")
    dataset = KGFaultDataset()
    print(f"  数据源: {dataset._data_source}")
    print(f"  实体: {dataset.num_nodes}")
    print(f"  关系类型: {dataset.num_original_relations}")
    print(f"  三元组: {len(dataset.triples)}")
    print(f"  整图训练节点: {len(dataset.fault_nodes)}")
    print(f"  故障类别: {dataset.get_fault_category_nodes()}")
    print(f"  关系列表: {dataset.relation_list}")

    # ---- 阶段 2: 训练 ----
    print(f"\n🚀 训练 R-GCN 模型 (device={args.device})...")
    model = run_training(
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.3,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )

    # ---- 阶段 3: 推理演示 ----
    print(f"\n📋 推理演示")
    print("=" * 64)

    # 预定义的测试用例
    test_queries = args.query or None
    if test_queries:
        test_cases = [test_queries]
    else:
        test_cases = [
            "加速迟缓",
            "冒白烟",
            "水温表红色",
            "抖动厉害",
            "启动异响",
        ]

    for query in test_cases:
        print(f"\n🔍 输入: \"{query}\"")
        result = run_inference(
            model, dataset, query,
            top_k_symptoms=5,
            top_k_faults=args.top_k,
        )
        print(format_result_compact(result))
        print()

    # ---- 完整诊断报告（第一个查询） ----
    detailed = run_inference(
        model, dataset,
        test_cases[0],
        top_k_symptoms=5,
        top_k_faults=args.top_k,
    )
    print(format_result(detailed))


if __name__ == "__main__":
    main()

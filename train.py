"""知识图谱故障诊断模型训练脚本。

使用 R-GCN 在知识图谱上训练节点嵌入模型，用于故障诊断推理。

用法:
    python train.py
    python train.py --epochs 500 --hidden-dim 128
    python train.py --query "加速迟缓" --device cuda
    python train.py --save-model ./models/rgcn.pt --no-infer
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from src.model.rgcn import FaultRGCN
from src.model.training import split_masks, train_rgcn, evaluate
from src.dataset.triple_dataset import KGTripleDataset
from src.pipeline.inference import (
    infer_from_text,
    format_result,
    format_result_compact,
    DiagnosisResult,
)
from src.core.config import logger, ESConfig


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(
        dataset: KGTripleDataset,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 5e-4,
        epochs: int = 300,
        weight_decay: float = 5e-4,
        device: str = "cpu",
        save_model_path: str | None = None,
) -> FaultRGCN:
    """训练 R-GCN 模型并返回训练好的模型。

    Parameters
    ----------
    dataset : KGTripleDataset
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
    save_model_path : str or None
        模型保存路径，为 None 则不保存。

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

    # 保存模型
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_nodes': model.num_nodes,
            'num_relations': model.num_relations,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
        }, save_model_path)
        logger.info("模型已保存至: %s", save_model_path)

    return model


def run_inference(
        model: FaultRGCN,
        dataset: KGTripleDataset,
        query: str,
        top_k_symptoms: int = 5,
        top_k_faults: int = 3,
        symptom_relation: str = "表现为",
) -> DiagnosisResult:
    """执行四阶段故障诊断推理。

    Parameters
    ----------
    model : FaultRGCN
        训练好的 R-GCN 模型。
    dataset : KGTripleDataset
        知识图谱数据集。
    query : str
        用户输入的症状描述文本。
    top_k_symptoms : int
        语义匹配的候选症状数。
    top_k_faults : int
        故障定位的候选故障数。
    symptom_relation : str
        症状-故障关系名称。

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
        symptom_relation=symptom_relation,
    )


def _parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="R-GCN 知识图谱故障诊断模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py
  python train.py --epochs 500 --hidden-dim 128
  python train.py --query "加速迟缓" --device cuda
  python train.py --save-model ./models/rgcn.pt --no-infer
        """,
    )

    # ---- 模型结构参数 ----
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="隐藏层维度 (默认: 64)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="R-GCN 层数 (默认: 2)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout 比例 (默认: 0.3)")

    # ---- 训练参数 ----
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮数 (默认: 300)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="学习率 (默认: 5e-4)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="L2 正则化 (默认: 5e-4)")
    parser.add_argument("--device", default="cpu",
                        help="训练设备 (默认: cpu)")

    # ---- 模型持久化 ----
    parser.add_argument("--save_model_path", type=str, default=None,
                        help="模型保存路径 (如 ./models/rgcn.pt)")

    # ---- ES 连接参数 ----
    parser.add_argument("--es-host", type=str, default="10.1.13.30",
                        help="ES 主机地址")
    parser.add_argument("--es-port", type=int, default=30920,
                        help="ES 端口")
    parser.add_argument("--es-username", type=str, default="elastic",
                        help="ES 用户名")
    parser.add_argument("--es-password", type=str, default="E1OfAx4Nf55513tU4i40eQbA",
                        help="ES 密码")
    parser.add_argument("--es-scheme", type=str, default="http",
                        help="ES 协议 (http/https)")

    # ---- 推理参数 ----
    parser.add_argument("--no-infer", action="store_true", default=False,
                        help="跳过推理验证阶段")
    parser.add_argument("--symptom-relation", type=str, default="表现为",
                        help="症状-故障关系名称 (默认: 表现为)")
    parser.add_argument("--query", type=str, nargs="+",
                        default=["发动机怠速不稳", "加速迟缓", "冒白烟"],
                        help="推理测试查询文本，支持多个 (默认: 发动机怠速不稳 加速迟缓 冒白烟)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="故障定位 Top-K (默认: 3)")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning("忽略未识别的参数: %s", unknown)
    return args


def main() -> None:
    """主入口：加载数据 → 训练 → (可选) 推理验证。"""
    args = _parse_args()
    set_seed(args.seed)

    es_config = ESConfig(
        host=args.es_host,
        port=args.es_port,
        username=args.es_username,
        password=args.es_password,
        scheme=args.es_scheme,
    )

    # ---- 阶段 1: 加载数据集 ----
    print("\n📦 加载 “车辆故障诊断” 知识图谱...")
    dataset = KGTripleDataset(es_config=es_config)
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=args.device,
        save_model_path=args.save_model_path,
    )

    # ---- 阶段 3: 推理验证 ----
    if args.no_infer:
        print("\n⏭️  已跳过推理验证阶段。")
        return

    print(f"\n📋 推理验证")
    print("=" * 64)

    for query in args.query:
        print(f"\n🔍 输入: \"{query}\"")
        result = run_inference(
            model, dataset, query,
            top_k_symptoms=5,
            top_k_faults=args.top_k,
            symptom_relation=args.symptom_relation,
        )
        print(format_result_compact(result))
        print()

    # ---- 完整诊断报告（第一个查询） ----
    detailed = run_inference(
        model, dataset,
        args.query[0],
        top_k_symptoms=5,
        top_k_faults=args.top_k,
        symptom_relation=args.symptom_relation,
    )
    print(format_result(detailed))


if __name__ == "__main__":
    main()

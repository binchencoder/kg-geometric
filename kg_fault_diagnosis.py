"""知识图谱故障诊断 —— 基于 GCN 的症状到故障 Top-K 推理。

模型使用 GCN 学习节点嵌入，支持两个阶段：
1. 故障节点 vs 正常实体的节点分类。
2. 给定一个或多个症状，按相似度对候选故障节点进行排序。

本模块是 src/ 包的入口包装，所有实现已迁移至 src/model/ 和 src/graph/ 子包。

运行方式：
    python kg_fault_diagnosis.py
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.model.gcn import FaultGCN
from src.model.training import split_masks, train, evaluate
from src.pipeline.diagnosis import topk_fault_diagnosis, print_topk_diagnosis
from src.graph.demo import KGFaultDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 保持 __all__ 向后兼容
__all__ = [
    "FaultGCN",
    "KGFaultDataset",
    "split_masks",
    "train",
    "evaluate",
    "topk_fault_diagnosis",
    "print_topk_diagnosis",
]


def main() -> None:
    """演示：内置小规模知识图谱上的 GCN 训练 + Top-K 故障诊断。"""
    from torch_geometric.data import Data

    dataset = KGFaultDataset()
    data: Data = dataset.to_data()
    data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)

    model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
    train(model, data)
    evaluate(model, data)

    example_symptoms = ["振动过高", "温度过高"]
    results = topk_fault_diagnosis(model, data, dataset.node_to_idx,
                                   dataset.fault_nodes, example_symptoms, top_k=3)
    print_topk_diagnosis(results, example_symptoms)

    example_symptoms_2 = ["电流过高"]
    results_2 = topk_fault_diagnosis(model, data, dataset.node_to_idx,
                                     dataset.fault_nodes, example_symptoms_2, top_k=3)
    print_topk_diagnosis(results_2, example_symptoms_2)


if __name__ == "__main__":
    main()

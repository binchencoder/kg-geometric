"""故障诊断推理 —— Top-K 故障根因排序。

基于症状节点嵌入的余弦相似度，对候选故障节点进行排序。

同时支持 FaultGCN（GCN 模型，`model.encode(data)`）和 FaultRGCN（R-GCN 模型，
`model.encode(edge_index, edge_type)`）两种接口。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.model.gcn import FaultGCN
from src.model.rgcn import FaultRGCN


def _extract_embeddings(
    model,
    data,
) -> torch.Tensor:
    """根据模型类型适配 encode 接口，返回节点嵌入张量。

    - FaultGCN: ``model.encode(data)`` —— 接受 PyG Data 对象
    - FaultRGCN: ``model.encode(edge_index, edge_type)`` —— 接受边索引与边类型
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, FaultRGCN):
            edge_index = data.edge_index
            edge_type = getattr(data, "edge_type", None)
            if edge_type is None:
                edge_type = torch.zeros(
                    edge_index.size(1), dtype=torch.long, device=edge_index.device
                )
            return model.encode(edge_index, edge_type)
        else:
            # 兼容 FaultGCN 及其他提供 encode(data) 接口的模型
            return model.encode(data)


def topk_fault_diagnosis(
    model,
    data,
    node_to_idx: Dict[str, int],
    fault_nodes: List[str],
    symptoms: Sequence[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """给定症状节点，基于嵌入余弦相似度推理最可能的故障根因。

    自动适配 ``FaultGCN`` / ``FaultRGCN`` 两种模型。

    Parameters
    ----------
    model : FaultGCN or FaultRGCN
        已训练好的图神经网络模型。
    data
        图数据（PyG Data，需包含 edge_index；对 FaultRGCN 还需含 edge_type）。
    node_to_idx : Dict[str, int]
        节点名称到索引的映射。
    fault_nodes : List[str]
        候选故障节点名称列表。
    symptoms : Sequence[str]
        症状节点名称列表。
    top_k : int
        返回的 Top-K 结果数。

    Returns
    -------
    List[Tuple[str, float]]
        按相似度降序排列的 (故障节点, 相似度) 列表。
    """
    node_emb = _extract_embeddings(model, data)

    symptom_indices = [node_to_idx[s] for s in symptoms if s in node_to_idx]
    if not symptom_indices:
        raise ValueError(f"None of the symptoms exist in the graph: {symptoms}")

    query_emb = node_emb[symptom_indices].mean(dim=0, keepdim=True)
    candidate_indices = [node_to_idx[f] for f in fault_nodes if f in node_to_idx]
    candidate_emb = node_emb[candidate_indices]

    query_norm = F.normalize(query_emb, p=2, dim=-1)
    cand_norm = F.normalize(candidate_emb, p=2, dim=-1)
    scores = (cand_norm * query_norm).sum(dim=-1)

    valid_fault_names = [f for f in fault_nodes if f in node_to_idx]
    ranked = sorted(
        zip(valid_fault_names, scores.cpu().tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


def print_topk_diagnosis(results: List[Tuple[str, float]], symptoms: Sequence[str]) -> None:
    """格式化输出 Top-K 故障诊断结果。

    Parameters
    ----------
    results : List[Tuple[str, float]]
        topk_fault_diagnosis 的返回值。
    symptoms : Sequence[str]
        输入的症状节点名称。
    """
    print("\nInput symptoms:", ", ".join(symptoms))
    print("Top-K fault diagnosis results:")
    for rank, (fault, score) in enumerate(results, start=1):
        print(f"  {rank}. {fault:<15} score={score:.4f}")
"""静态链接预测推理管线 —— 从查询文本到结构化关系结果。

实现两阶段通用推理流程：

1. 语义匹配 (match_nodes):
   用户输入查询文本 → 使用 R-GCN 编码器计算与图中所有节点的语义相似度
   → 选取 Top-K 最相似的节点作为头实体候选。

2. 关系检索 (retrieve_relations):
   以最佳匹配头实体为中心，按 relation_mapping 中给定的每个关系名正向
   检索其相连的尾实体，传入几个关系就输出几个关系的结果。

使用示例::

    dataset = KGTripleDataset()
    model = FaultRGCN(...)
    # 训练 model ...

    result = infer_from_text(
        model, dataset, "加速迟缓",
        relation_mapping={"causes": "由...引起", "actions": "维修措施"},
        device="cpu",
    )
    print(format_result(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.core.config import logger


# ================================================================
# 数据结构
# ================================================================


@dataclass
class SLPResult:
    """一次完整的静态链接预测结果。"""

    # 用户原始输入
    query_text: str = ""

    # 语义匹配到的头实体候选 [(node_text, similarity_score), ...]
    matched_nodes: List[Tuple[str, float]] = field(default_factory=list)

    # 最佳匹配头实体
    best_match: str = ""

    # 按 relation_mapping 的 key 分组的关系结果 {key: [tail_nodes, ...]}
    relations: Dict[str, List[str]] = field(default_factory=dict)


# ================================================================
# 阶段 1: 语义匹配
# ================================================================


def match_nodes(
        model,
        dataset,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    """计算输入文本与图中所有节点的语义相似度。

    基于字符级 Jaccard 重叠构造查询向量，再利用 R-GCN 节点嵌入计算与全图
    节点的余弦相似度，返回 Top-K 最相似节点作为头实体候选。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGTripleDataset
        知识图谱数据集。
    query_text : str
        查询文本。
    top_k : int
        返回的最相似节点数量。
    similarity_threshold : float
        最低字符级相似度阈值（0-1），低于此值的匹配将被过滤。

    Returns
    -------
    List[Tuple[str, float]]
        [(node_text, similarity_score), ...] 按相似度降序排列。
    """
    candidate_nodes = [
        dataset.idx_to_node[i] for i in range(dataset.num_nodes)
    ]
    if not candidate_nodes:
        logger.warning("图中没有任何节点，无法进行语义匹配")
        return []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # 获取所有候选节点的嵌入
        candidate_indices = [
            dataset.node_to_idx[s] for s in candidate_nodes
            if s in dataset.node_to_idx
        ]
        if not candidate_indices:
            return []

        candidate_embs = model.node_emb(
            torch.tensor(candidate_indices, device=device)
        )  # [num_candidates, hidden_dim]

        # 查询文本编码：用文本中的字符在节点嵌入中的平均表示
        query_chars = set(query_text)
        matched_indices = []
        matched_weights = []

        for s, node_name in enumerate(candidate_nodes):
            if node_name not in dataset.node_to_idx:
                continue
            # 计算 Jaccard 字符重叠度作为初始权重
            node_chars = set(node_name)
            overlap = len(query_chars & node_chars)
            if overlap > 0:
                jaccard = overlap / max(len(query_chars | node_chars), 1)
                if jaccard >= similarity_threshold:
                    matched_indices.append(s)
                    matched_weights.append(jaccard)

        if not matched_indices:
            # 没有任何字符重叠，返回 top-k 作为降级方案
            logger.debug("无字符级匹配，返回前 %d 个候选节点", top_k)
            matched_indices = list(range(min(top_k, len(candidate_nodes))))
            matched_weights = [0.0] * len(matched_indices)

        # 加权均值构成 query embedding
        weights_tensor = torch.tensor(
            matched_weights, device=device, dtype=torch.float
        )
        matched_embs = candidate_embs[matched_indices]  # [M, hidden_dim]
        query_emb = (
                matched_embs * weights_tensor.unsqueeze(-1)
        ).sum(dim=0, keepdim=True)
        query_emb = F.normalize(query_emb / (weights_tensor.sum() + 1e-8), p=2)

        # 计算所有候选节点的余弦相似度
        candidate_embs_norm = F.normalize(candidate_embs, p=2)
        scores = (candidate_embs_norm * query_emb).sum(dim=-1)  # [num_candidates]

        # 排序取 top-k
        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(scores))
        )
        results = [
            (candidate_nodes[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
        logger.info(
            "语义匹配: '%s' → %d 个候选节点 (%s → %.3f)",
            query_text,
            len(results),
            results[0][0] if results else "none",
            results[0][1] if results else 0.0,
        )
        return results


def _text_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的字符级 Jaccard 相似度（纯文本后备方案）。"""
    set_a = set(text_a)
    set_b = set(text_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ================================================================
# 阶段 2: 关系检索
# ================================================================


def retrieve_relations(
        dataset,
        node: str,
        relation_mapping: Dict[str, str],
) -> Dict[str, List[str]]:
    """以给定节点为中心，按关系映射检索其正向相连的尾实体。

    Parameters
    ----------
    dataset : KGTripleDataset
        知识图谱数据集。
    node : str
        中心节点名称（通常为语义匹配得到的最佳头实体）。
    relation_mapping : Dict[str, str]
        输出字段名 → 关系名的映射。传入几个关系就返回几个关系的结果。

    Returns
    -------
    Dict[str, List[str]]
        {关系名: [tail_nodes, ...]}，key 为 mapping 中的实际关系名。
    """
    return dataset.get_node_relations(node, relation_mapping=relation_mapping)


# ================================================================
# 阶段 4: 结果组装
# ================================================================


def format_result(result: SLPResult) -> str:
    """将 SLPResult 格式化为自然语言静态链接预测报告。

    Parameters
    ----------
    result : SLPResult
        完整推理结果。

    Returns
    -------
    str
        结构化的自然语言推理报告。
    """
    lines = []

    # 标题
    lines.append("=" * 64)
    lines.append("  静态链接预测报告")
    lines.append("=" * 64)

    # 用户输入
    lines.append(f"\n📋 查询: \"{result.query_text}\"")

    # 匹配的头实体
    if result.matched_nodes:
        lines.append(f"\n🔍 匹配到 {len(result.matched_nodes)} 个相似实体:")
        for i, (node, score) in enumerate(result.matched_nodes[:5], 1):
            bar = "█" * min(20, max(1, int(score * 20)))
            lines.append(f"  {i}. {node}  [{bar}] {score:.3f}")

    # 最佳匹配实体
    if result.best_match:
        lines.append(f"\n🎯 最佳匹配实体: 【{result.best_match}】")

    # 关系结果
    if result.relations:
        lines.append("\n📎 关系结果:")
        for relation, nodes in result.relations.items():
            if nodes:
                lines.append(f"    • {relation}: {', '.join(nodes[:5])}")

    lines.append(f"\n{'=' * 64}")
    return "\n".join(lines)


def format_result_compact(result: SLPResult) -> str:
    """紧凑版格式化输出。

    Parameters
    ----------
    result : SLPResult
        完整推理结果。

    Returns
    -------
    str
        紧凑的自然语言摘要。
    """
    if not result.best_match:
        return "未能识别到匹配的实体。"

    parts = [
        f"查询「{result.query_text}」匹配到【{result.best_match}】。",
    ]

    if result.relations:
        for relation, nodes in result.relations.items():
            if nodes:
                parts.append(f"{relation}：{'、'.join(nodes)}。")

    return "".join(parts)


# ================================================================
# 统一推理入口
# ================================================================


def infer_from_text(
        model,
        dataset,
        query_text: str,
        device: str,
        relation_mapping: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
) -> SLPResult:
    """通用静态链接预测：从查询文本匹配头实体并检索各关系尾实体。

    流程：
      1. 语义匹配 —— 将查询文本与图中所有节点做字符级语义匹配，
         得到 Top-K 头实体候选；
      2. 关系检索 —— 以最佳匹配头实体为中心，按 relation_mapping 中给定的
         每个关系检索其正向相连的尾实体，传入几个关系就输出几个关系的
         结果（relation_mapping 为 None 时使用数据集默认关系映射）。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGTripleDataset
        知识图谱数据集。
    query_text : str
        查询文本（如实体名称片段）。
    relation_mapping : Optional[Dict[str, str]]
        输出字段名 → 关系名的映射，例如
        {"causes": "由...引起", "actions": "维修措施"}。
    top_k : int
        语义匹配返回的最相似头实体数量。
    similarity_threshold : float
        字符级相似度最低阈值。
    device : Optional[str]
        推理设备，None 则跟随模型当前设备。

    Returns
    -------
    SLPResult
        包含匹配头实体与按关系分组的结果。
    """
    model = model.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    dataset.edge_type = dataset.edge_type.to(device)

    if relation_mapping is None:
        relation_mapping = dataset.default_relation_mapping

    result = SLPResult(query_text=query_text)

    # ---- 阶段 1: 语义匹配（全图）----
    matched_nodes = match_nodes(
        model, dataset, query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    result.matched_nodes = matched_nodes

    if not matched_nodes:
        logger.warning("语义匹配失败，无法进行后续关系检索")
        return result

    # ---- 阶段 2: 关系检索（按 relation_mapping）----
    best_match = matched_nodes[0][0]
    result.best_match = best_match

    relations = retrieve_relations(
        dataset,
        best_match,
        relation_mapping=relation_mapping
    )
    result.relations = relations

    logger.info(
        "推理管线完成: '%s' → 最佳匹配 %s，检索到 %d 个关系",
        query_text, best_match, len(relations),
    )
    return result

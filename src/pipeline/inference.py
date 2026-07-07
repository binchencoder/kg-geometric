"""故障诊断推理管线 —— 从症状文本到结构化维修方案。

实现四阶段推理流程：

1. 语义匹配 (match_symptoms):
   用户输入自然语言症状描述 → 使用 R-GCN 编码器计算与图中所有
   "表现为" tail 节点的语义相似度 → 选取 Top-K 最相似的症状节点作为起点。

2. 故障定位 (locate_fault):
   沿"表现为"边的反向遍历 → 找到与匹配症状节点相连的故障类别节点
   → 利用 R-GCN 节点嵌入对候选故障节点进行余弦相似度打分排序。

3. 答案生成 (generate_answer):
   以确定的最佳故障节点为中心，沿知识图谱结构正向提取：
   - "由...引起" → 故障原因列表
   - "维修措施"   → 维修方案列表
   - "需要工具"   → 所需工具列表

4. 结果组装 (format_result):
   将结构化信息组装为自然语言诊断报告。

使用示例::

    dataset = KGFaultDataset()
    model = FaultRGCN(...)
    # 训练 model ...

    result = infer_from_text(
        model, dataset, "加速迟缓",
        top_k_symptoms=3, top_k_faults=3,
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
class DiagnosisResult:
    """一次完整的故障诊断结果。"""

    # 用户原始输入
    query_text: str = ""

    # 匹配到的症状节点
    matched_symptoms: List[Tuple[str, float]] = field(default_factory=list)
    # [(symptom_text, similarity_score), ...]

    # 故障定位结果（按置信度降序）
    fault_candidates: List[Tuple[str, float]] = field(default_factory=list)
    # [(fault_name, confidence_score), ...]

    # 最佳故障的详细诊断信息
    best_fault: str = ""
    causes: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    system: str = ""
    category: str = ""

    # 当图谱关系名与默认语义角色不匹配时，按真实关系名分组的关系数据
    relations: Dict[str, List[str]] = field(default_factory=dict)

    # 备选故障的诊断信息
    alternative_faults: List[Dict] = field(default_factory=list)


# ================================================================
# 阶段 1: 语义匹配
# ================================================================


def match_symptoms(
        model,
        dataset,
        query_text: str,
        symptom_relation: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    """计算输入文本与图中节点的语义相似度。

    优先匹配症状节点（symptom_relation 的 tail）。当图谱不存在该关系时，
    退化为全图节点匹配，实现 type-agnostic 推理。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    query_text : str
        用户输入的自然语言症状描述。
    top_k : int
        返回的最相似节点数量。
    similarity_threshold : float
        最低相似度阈值（0-1），低于此值的匹配将被过滤。
    symptom_relation : str
        用于识别症状节点的关系名称（默认: "表现为"）。

    Returns
    -------
    List[Tuple[str, float]]
        [(node_text, similarity_score), ...] 按相似度降序排列。
    """
    # type-agnostic fallback：没有指定关系时匹配所有节点
    if dataset.has_relation(symptom_relation):
        candidate_nodes = dataset.get_symptom_nodes(symptom_relation)
    else:
        candidate_nodes = [
            dataset.idx_to_node[i] for i in range(dataset.num_nodes)
        ]

    if not candidate_nodes:
        logger.warning("图中没有候选节点，无法进行语义匹配")
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
        # 策略：对包含查询文本字符的所有节点的嵌入求均值作为查询向量
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
# 阶段 2: 故障定位
# ================================================================


def locate_fault(
        model,
        dataset,
        matched_symptoms: List[Tuple[str, float]],
        symptom_relation: str,
        top_k: int = 3,
) -> List[Tuple[str, float]]:
    """根据匹配到的节点，定位最可能的中心节点。

    工作流程：
    1. 若存在 symptom_relation 关系，沿其反向遍历找到相连的故障类别节点；
       否则退化为 type-agnostic：收集所有邻居节点作为候选。
    2. 用 R-GCN 编码的节点嵌入计算余弦相似度排序。
    3. 按综合得分降序返回 Top-K 候选。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    matched_symptoms : List[Tuple[str, float]]
        match_symptoms 的输出，[(node, score), ...]。
    top_k : int
        返回的候选节点数量。
    symptom_relation : str
        用于识别症状节点与故障节点之间关系的名称（默认: "表现为"）。

    Returns
    -------
    List[Tuple[str, float]]
        [(node_name, confidence_score), ...] 按置信度降序。
    """
    if not matched_symptoms:
        return []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Step 1: 通过图拓扑找到候选中心节点
        candidate_faults: Dict[str, float] = {}  # node_name → topology_score

        for symptom, sim_score in matched_symptoms:
            if dataset.has_relation(symptom_relation):
                # 沿 symptom_relation 反向查找
                fault_heads = dataset.get_backward(symptom, symptom_relation)
            else:
                # type-agnostic：收集该节点的所有邻居
                fault_heads = dataset.get_all_neighbors(symptom)

            for fault_name in fault_heads:
                if fault_name not in dataset.node_to_idx:
                    continue
                # 累积拓扑权重（匹配得分越高则候选可能性越大）
                candidate_faults[fault_name] = max(
                    candidate_faults.get(fault_name, 0.0),
                    sim_score,
                )

        if not candidate_faults:
            logger.warning("未找到与候选节点相连的中心节点，尝试全局搜索...")
            # 降级：使用所有故障类别节点；若仍为空则使用全图节点
            all_faults = dataset.get_fault_category_nodes()
            if not all_faults:
                all_faults = list(dataset.fault_nodes)
            candidate_faults = {f: 0.0 for f in all_faults}

        # Step 2: 用 R-GCN 嵌入计算语义相似度
        # 对匹配的节点嵌入求加权平均作为 query
        symptom_indices = [
            dataset.node_to_idx[s]
            for s, _ in matched_symptoms
            if s in dataset.node_to_idx
        ]
        if symptom_indices:
            emb = model.encode(dataset.edge_index, dataset.edge_type)
            symptom_embs = emb[symptom_indices]
            # 加权平均
            weights = torch.tensor(
                [score for _, score in matched_symptoms
                 if matched_symptoms[0][0] in dataset.node_to_idx][:len(symptom_indices)],
                device=device,
            )
            if len(weights) < len(symptom_indices):
                weights = torch.ones(len(symptom_indices), device=device)
            query_emb = (
                    symptom_embs * weights.unsqueeze(-1)
            ).sum(dim=0, keepdim=True)
            query_emb = F.normalize(query_emb / (weights.sum() + 1e-8), p=2)

            # 计算每个候选节点与 query 的余弦相似度
            scored: List[Tuple[str, float]] = []
            for fault_name, topo_score in candidate_faults.items():
                if fault_name not in dataset.node_to_idx:
                    continue
                fault_idx = dataset.node_to_idx[fault_name]
                fault_emb = F.normalize(emb[fault_idx].unsqueeze(0), p=2)
                cos_sim = (fault_emb * query_emb).sum().item()
                # 综合拓扑得分和嵌入相似度
                combined = 0.4 * topo_score + 0.6 * max(cos_sim, 0.0)
                scored.append((fault_name, combined))
        else:
            # 纯拓扑降级
            scored = list(candidate_faults.items())

        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_k]

        logger.info(
            "中心定位: %d 个候选 → Top-%d: %s (%.3f)",
            len(scored),
            min(top_k, len(result)),
            result[0][0] if result else "none",
            result[0][1] if result else 0.0,
        )
        return result


# ================================================================
# 阶段 3: 答案生成
# ================================================================


def generate_answer(
        dataset,
        fault_name: str,
        relation_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """以中心节点为中心，沿知识图谱提取诊断结构。

    Parameters
    ----------
    dataset : KGFaultDataset
        知识图谱数据集。
    fault_name : str
        中心节点名称。
    relation_mapping : Optional[Dict[str, str]]
        语义角色 → 实际关系名的映射，用于 type-agnostic 输出。

    Returns
    -------
    Dict[str, List[str]]
        {"causes": [...], "actions": [...], "tools": [...],
         "system": [...], "symptoms": [...]} 或按真实关系名分组。
    """
    return dataset.get_fault_info(fault_name, relation_mapping=relation_mapping)


# ================================================================
# 阶段 4: 结果组装
# ================================================================


def format_result(result: DiagnosisResult) -> str:
    """将 DiagnosisResult 格式化为自然语言诊断报告。

    Parameters
    ----------
    result : DiagnosisResult
        完整诊断结果。

    Returns
    -------
    str
        结构化的自然语言诊断报告。
    """
    lines = []

    # 标题
    lines.append("=" * 64)
    lines.append("  故障诊断报告")
    lines.append("=" * 64)

    # 用户输入
    lines.append(f"\n📋 输入症状: \"{result.query_text}\"")

    # 匹配的症状
    if result.matched_symptoms:
        lines.append(f"\n🔍 匹配到 {len(result.matched_symptoms)} 个相似症状:")
        for i, (symptom, score) in enumerate(result.matched_symptoms[:5], 1):
            bar = "█" * min(20, max(1, int(score * 20)))
            lines.append(f"  {i}. {symptom}  [{bar}] {score:.3f}")

    # 故障定位结果
    if result.fault_candidates:
        lines.append(f"\n🎯 故障定位 (Top-{len(result.fault_candidates)}):")
        for rank, (fault, score) in enumerate(result.fault_candidates, 1):
            marker = "►" if rank == 1 else "  "
            lines.append(f"  {marker} #{rank}: {fault}  (置信度: {score:.3f})")

    # 最佳故障详情
    if result.best_fault:
        lines.append(f"\n📊 最佳匹配故障详情: 【{result.best_fault}】")

        if result.system:
            lines.append(f"  所属系统: {result.system}")
        if result.category:
            lines.append(f"  故障类别: {result.category}")

        if result.causes:
            lines.append(f"\n  ⚠️  可能原因 ({len(result.causes)} 项):")
            for cause in result.causes:
                lines.append(f"    • {cause}")

        if result.actions:
            lines.append(f"\n  🔧 维修措施 ({len(result.actions)} 项):")
            for action in result.actions:
                lines.append(f"    • {action}")

        if result.tools:
            lines.append(f"\n  🛠️  所需工具 ({len(result.tools)} 项):")
            for tool in result.tools:
                lines.append(f"    • {tool}")

        if result.relations:
            lines.append(f"\n  📎 关系详情 (Type-agnostic):")
            for relation, nodes in result.relations.items():
                lines.append(f"    • {relation}: {', '.join(nodes[:5])}")

    # 备选故障
    if result.alternative_faults:
        lines.append(f"\n📝 备选故障诊断:")
        for alt in result.alternative_faults:
            lines.append(f"  ── 【{alt['fault']}】 (置信度: {alt['confidence']:.3f}) ──")
            if alt.get("causes"):
                lines.append(f"    原因: {'、'.join(alt['causes'][:3])}")
            if alt.get("actions"):
                lines.append(f"    措施: {'、'.join(alt['actions'][:3])}")

    lines.append(f"\n{'=' * 64}")
    return "\n".join(lines)


def format_result_compact(result: DiagnosisResult) -> str:
    """紧凑版格式化输出。

    Parameters
    ----------
    result : DiagnosisResult
        完整诊断结果。

    Returns
    -------
    str
        紧凑的自然语言诊断摘要。
    """
    if not result.best_fault:
        return "未能识别到匹配的故障。"

    parts = [
        f"检测到故障现象「{result.query_text}」匹配到【{result.best_fault}】。",
    ]

    if result.causes:
        parts.append(
            f"可能原因：{'、'.join(result.causes)}。"
        )

    if result.actions or result.tools:
        suggestion = []
        if result.tools:
            suggestion.append(f"使用{'和'.join(result.tools)}检查")
        if result.actions:
            suggestion.append(f"视情况{'或'.join(result.actions)}")
        parts.append(f"建议维修：{'，'.join(suggestion)}。")

    return "".join(parts)


# ================================================================
# 统一推理入口
# ================================================================


def infer_from_text(
        model,
        dataset,
        query_text: str,
        symptom_relation: str,
        top_k_symptoms: int = 5,
        top_k_faults: int = 3,
        similarity_threshold: float = 0.0,
        device: Optional[str] = None,
        relation_mapping: Optional[Dict[str, str]] = None,
) -> DiagnosisResult:
    """完整的四阶段故障诊断推理管线。

    从用户输入的自然语言症状文本，经过语义匹配 → 故障定位
    → 答案生成 → 结果组装，输出完整的结构化诊断报告。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    query_text : str
        用户输入的自然语言症状描述。
    top_k_symptoms : int
        语义匹配返回的最相似症状数。
    top_k_faults : int
        故障定位返回的候选故障数。
    similarity_threshold : float
        症状匹配的最低相似度阈值。
    device : Optional[str]
        推理设备，None 则跟随模型当前设备。
    relation_mapping : Optional[Dict[str, str]]
        语义角色 → 实际关系名的映射。用于 type-agnostic 输出。
        例如 {"causes": "原因", "actions": "处理措施"}。
    symptom_relation : str
        用于识别症状节点与故障节点之间关系的名称（默认: "表现为"）。

    Returns
    -------
    DiagnosisResult
        包含完整诊断信息的结构化结果。
    """
    if device is not None:
        model = model.to(device)
        dataset.edge_index = dataset.edge_index.to(device)
        dataset.edge_type = dataset.edge_type.to(device)

    result = DiagnosisResult(query_text=query_text)

    # ---- Phase 1: 语义匹配 ----
    matched_symptoms = match_symptoms(
        model, dataset, query_text,
        top_k=top_k_symptoms,
        similarity_threshold=similarity_threshold,
        symptom_relation=symptom_relation,
    )
    result.matched_symptoms = matched_symptoms

    if not matched_symptoms:
        logger.warning("症状匹配失败，无法进行后续推理")
        return result

    # ---- Phase 2: 故障定位 ----
    fault_candidates = locate_fault(
        model, dataset,
        matched_symptoms=matched_symptoms,
        symptom_relation=symptom_relation,
        top_k=top_k_faults,
    )
    result.fault_candidates = fault_candidates

    if not fault_candidates:
        return result

    # ---- Phase 3: 答案生成（最佳故障） ----
    best_fault, best_score = fault_candidates[0]
    result.best_fault = best_fault

    info = generate_answer(dataset, best_fault, relation_mapping=relation_mapping)
    standard_keys = {"causes", "actions", "tools", "system", "category", "symptoms"}
    if set(info.keys()).issubset(standard_keys):
        result.causes = info.get("causes", [])
        result.actions = info.get("actions", [])
        result.tools = info.get("tools", [])
        result.system = (
            info.get("system", [""])[0] if info.get("system") else ""
        )
        result.category = (
            info.get("category", [""])[0] if info.get("category") else ""
        )
    else:
        # type-agnostic：保留原始关系分组，并尝试将常见同义词回填到标准字段
        result.relations = info
        result.causes = info.get("causes", info.get("原因", []))
        result.actions = info.get("actions", info.get("维修措施", info.get("处理措施", [])))
        result.tools = info.get("tools", info.get("需要工具", info.get("工具", [])))
        result.system = (
            info.get("system", info.get("系统", [""]))[0]
            if info.get("system") or info.get("系统")
            else ""
        )
        result.category = (
            info.get("category", info.get("类别", [""]))[0]
            if info.get("category") or info.get("类别")
            else ""
        )

    # ---- Phase 4: 备选故障详情 ----
    for fault_name, score in fault_candidates[1:]:
        alt_info = generate_answer(dataset, fault_name, relation_mapping=relation_mapping)
        alt_result = {
            "fault": fault_name,
            "confidence": score,
            "causes": alt_info.get("causes", []),
            "actions": alt_info.get("actions", []),
            "tools": alt_info.get("tools", []),
        }
        if not (alt_result["causes"]
                or alt_result["actions"]
                or alt_result["tools"]):
            alt_result["relations"] = alt_info
        result.alternative_faults.append(alt_result)

    logger.info(
        "推理管线完成: '%s' → %s (%.3f), %d causes, %d actions, %d tools",
        query_text, result.best_fault, best_score,
        len(result.causes), len(result.actions), len(result.tools),
    )
    return result

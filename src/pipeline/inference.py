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

    # 备选故障的诊断信息
    alternative_faults: List[Dict] = field(default_factory=list)


# ================================================================
# 阶段 1: 语义匹配
# ================================================================


def match_symptoms(
        model,
        dataset,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    """计算输入文本与图中所有症状节点的语义相似度。

    使用 R-GCN 的 node_emb 层（未经图卷积的嵌入）作为文本表示，
    通过简单的子串匹配 + 嵌入相似度进行语义检索。

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    query_text : str
        用户输入的自然语言症状描述。
    top_k : int
        返回的最相似症状数量。
    similarity_threshold : float
        最低相似度阈值（0-1），低于此值的匹配将被过滤。

    Returns
    -------
    List[Tuple[str, float]]
        [(symptom_text, similarity_score), ...] 按相似度降序排列。
    """
    symptom_nodes = dataset.get_symptom_nodes()
    if not symptom_nodes:
        logger.warning("图中没有症状节点，无法进行语义匹配")
        return []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # 获取所有症状节点的嵌入
        symptom_indices = [
            dataset.node_to_idx[s] for s in symptom_nodes
            if s in dataset.node_to_idx
        ]
        if not symptom_indices:
            return []

        symptom_embs = model.node_emb(
            torch.tensor(symptom_indices, device=device)
        )  # [num_symptoms, hidden_dim]

        # 查询文本编码：用文本中的字符在节点嵌入中的平均表示
        # 策略：对包含查询文本字符的所有节点的嵌入求均值作为查询向量
        query_chars = set(query_text)
        matched_indices = []
        matched_weights = []

        for s, node_name in enumerate(symptom_nodes):
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
            logger.debug("无字符级匹配，返回前 %d 个症状", top_k)
            matched_indices = list(range(min(top_k, len(symptom_nodes))))
            matched_weights = [0.0] * len(matched_indices)

        # 加权均值构成 query embedding
        weights_tensor = torch.tensor(
            matched_weights, device=device, dtype=torch.float
        )
        matched_embs = symptom_embs[matched_indices]  # [M, hidden_dim]
        query_emb = (
            matched_embs * weights_tensor.unsqueeze(-1)
        ).sum(dim=0, keepdim=True)
        query_emb = F.normalize(query_emb / (weights_tensor.sum() + 1e-8), p=2)

        # 计算所有症状节点的余弦相似度
        symptom_embs_norm = F.normalize(symptom_embs, p=2)
        scores = (symptom_embs_norm * query_emb).sum(dim=-1)  # [num_symptoms]

        # 排序取 top-k
        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(scores))
        )
        results = [
            (symptom_nodes[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
        logger.info(
            "语义匹配: '%s' → %d 个症状 (%s → %.3f)",
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
        top_k: int = 3,
) -> List[Tuple[str, float]]:
    """根据匹配到的症状节点，定位最可能的故障类别。

    工作流程：
    1. 沿"表现为"边反向遍历，找到与症状相连的故障类别节点
    2. 用 R-GCN 编码的节点嵌入计算症状聚合向量与故障节点的余弦相似度
    3. 按相似度降序排列返回 Top-K 候选故障

    Parameters
    ----------
    model : FaultRGCN
        已训练的 R-GCN 模型。
    dataset : KGFaultDataset
        知识图谱数据集。
    matched_symptoms : List[Tuple[str, float]]
        match_symptoms 的输出，[(symptom, score), ...]。
    top_k : int
        返回的候选故障数量。

    Returns
    -------
    List[Tuple[str, float]]
        [(fault_name, confidence_score), ...] 按置信度降序。
    """
    if not matched_symptoms:
        return []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Step 1: 通过图拓扑找到候选故障节点
        candidate_faults: Dict[str, float] = {}  # fault_name → topology_score

        for symptom, sim_score in matched_symptoms:
            # 沿"表现为"反向查找：谁"表现为"这个症状？
            fault_heads = dataset.get_backward(symptom, "表现为")
            for fault_name in fault_heads:
                if fault_name not in dataset.node_to_idx:
                    continue
                # 累积拓扑权重（症状相似度越高则故障可能性越大）
                candidate_faults[fault_name] = max(
                    candidate_faults.get(fault_name, 0.0),
                    sim_score,
                )

        if not candidate_faults:
            logger.warning("未找到与症状相连的故障节点，尝试全局搜索...")
            # 降级：使用所有故障类别节点
            all_faults = dataset.get_fault_category_nodes()
            if not all_faults:
                all_faults = [
                    f for f in dataset.fault_nodes
                ]
            candidate_faults = {f: 0.0 for f in all_faults}

        # Step 2: 用 R-GCN 嵌入计算语义相似度
        # 对匹配的症状节点嵌入求加权平均作为 query
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

            # 计算每个候选故障与 query 的余弦相似度
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
            "故障定位: %d 个候选 → Top-%d: %s (%.3f)",
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
) -> Dict[str, List[str]]:
    """以故障节点为中心，沿知识图谱提取诊断结构。

    Parameters
    ----------
    dataset : KGFaultDataset
        知识图谱数据集。
    fault_name : str
        故障类别节点名称。

    Returns
    -------
    Dict[str, List[str]]
        {"causes": [...], "actions": [...], "tools": [...],
         "system": [...], "symptoms": [...]}
    """
    return dataset.get_fault_info(fault_name)


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
        top_k_symptoms: int = 5,
        top_k_faults: int = 3,
        similarity_threshold: float = 0.0,
        device: Optional[str] = None,
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
    )
    result.matched_symptoms = matched_symptoms

    if not matched_symptoms:
        logger.warning("症状匹配失败，无法进行后续推理")
        return result

    # ---- Phase 2: 故障定位 ----
    fault_candidates = locate_fault(
        model, dataset, matched_symptoms,
        top_k=top_k_faults,
    )
    result.fault_candidates = fault_candidates

    if not fault_candidates:
        return result

    # ---- Phase 3: 答案生成（最佳故障） ----
    best_fault, best_score = fault_candidates[0]
    result.best_fault = best_fault

    info = generate_answer(dataset, best_fault)
    result.causes = info.get("causes", [])
    result.actions = info.get("actions", [])
    result.tools = info.get("tools", [])
    result.system = (
        info.get("system", [""])[0] if info.get("system") else ""
    )
    result.category = (
        info.get("category", [""])[0] if info.get("category") else ""
    )

    # ---- Phase 4: 备选故障详情 ----
    for fault_name, score in fault_candidates[1:]:
        alt_info = generate_answer(dataset, fault_name)
        result.alternative_faults.append({
            "fault": fault_name,
            "confidence": score,
            "causes": alt_info.get("causes", []),
            "actions": alt_info.get("actions", []),
            "tools": alt_info.get("tools", []),
        })

    logger.info(
        "推理管线完成: '%s' → %s (%.3f), %d causes, %d actions, %d tools",
        query_text, result.best_fault, best_score,
        len(result.causes), len(result.actions), len(result.tools),
    )
    return result

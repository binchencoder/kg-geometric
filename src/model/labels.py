"""故障标签构建器 —— 从知识图谱中自动识别故障节点并生成训练标签。

识别策略：当 relation ∈ fault_relations 时，收集 tail 实体作为故障节点。
与 kg_fault_demo.py 逻辑一致（如 "由...引起" 关系 → tail 即为故障原因）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch

from src.core.config import logger

if TYPE_CHECKING:
    from src.es.streamer import ESTripletStreamer
    from src.es.vocabulary import KGVocabulary


class FaultLabelBuilder:
    """从已解析的知识图谱中自动识别故障节点并构建训练标签。

    识别策略：当 relation ∈ fault_relations 时，收集 tail 实体作为故障节点。
    例如：
    - "由...引起" 关系 → tail 为故障原因
    - "类型为" 关系   → tail 为故障类别
    """

    DEFAULT_FAULT_RELATIONS = [
        "由...引起", "原因在于", "类型为", "type", "类别为", "分类为",
        "故障类型", "fault_type", "发生故障",
    ]

    def __init__(
            self,
            vocab: "KGVocabulary",
            fault_relations: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        vocab : KGVocabulary
            全局词汇表，已包含所有实体和关系类型。
        fault_relations : Optional[List[str]]
            指向故障原因的关系名称列表，如 ["由...引起", "原因在于"]。
            默认包含 "由...引起" / "类型为" 等常见关系。
        """
        self.vocab = vocab
        self.fault_relations = fault_relations or self.DEFAULT_FAULT_RELATIONS

        self.fault_nodes: List[str] = []
        self.y: Optional[torch.Tensor] = None
        self.fault_mask: Optional[torch.Tensor] = None

    def build_from_streamer(
            self,
            streamer: "ESTripletStreamer",
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            extra_filters: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """全量扫描三元组，根据关系模式识别故障节点。

        与 kg_fault_demo.py 一致：当 relation ∈ fault_relations 时，
        将 tail 实体收集为故障节点（故障原因）。

        Returns
        -------
        Tuple[torch.Tensor, List[str]]
            (y: 标签张量 [num_entities], fault_nodes: 故障节点名称列表)
        """
        _fault_relation_set = set(self.fault_relations)
        fault_set: Set[str] = set()

        total_scanned = 0
        logger.info("开始扫描三元组以识别故障节点 ...")
        for batch in streamer.stream_triplets(
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
                extra_filters=extra_filters,
                resume=False,
        ):
            for t in batch:
                head, rel, tail = t["head"], t["relation"], t["tail"]
                if rel in _fault_relation_set:
                    if tail in self.vocab.entity2idx:
                        fault_set.add(tail)
                total_scanned += 1

        self.fault_nodes = sorted(fault_set)

        num_entities = self.vocab.num_entities
        y_list = []
        for i in range(num_entities):
            entity_name = self.vocab.idx2entity[i]
            y_list.append(1 if entity_name in fault_set else 0)

        self.y = torch.tensor(y_list, dtype=torch.long)
        self.fault_mask = self.y.bool()

        fault_count = int(self.y.sum().item())
        logger.info(
            "故障节点识别完成: 扫描 %d 条三元组, 识别 %d 个故障节点 / 共 %d 个实体 (%.2f%%)",
            total_scanned, fault_count, num_entities,
            100.0 * fault_count / max(num_entities, 1),
        )
        if fault_count == 0:
            logger.warning(
                "未识别到任何故障节点！请检查 fault_relations=%s "
                "是否与知识图谱中的实际关系匹配",
                self.fault_relations,
            )

        return self.y, self.fault_nodes

    @property
    def num_faults(self) -> int:
        return len(self.fault_nodes)

    @property
    def num_entities(self) -> int:
        return self.vocab.num_entities

    @property
    def num_normal(self) -> int:
        return self.num_entities - self.num_faults

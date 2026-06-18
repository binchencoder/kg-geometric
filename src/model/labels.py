"""故障标签构建器 —— 从知识图谱中自动识别故障节点并生成训练标签。

识别策略基于关系模式匹配：查找 (实体 --关系类型--> 尾实体) 中
relation ∈ fault_relations 且 tail ∈ fault_tails 的实体，标记为故障节点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch

from ..core.config import logger

if TYPE_CHECKING:
    from ..es.streamer import ESTripletStreamer
    from ..es.vocabulary import KGVocabulary


class FaultLabelBuilder:
    """从已解析的知识图谱中自动识别故障节点并构建训练标签。

    识别策略：
    1. 基于关系模式匹配：查找 (实体 --关系类型--> 尾实体) 中
       relation ∈ fault_relations 且 tail ∈ fault_tails 的实体，
       将其标记为故障节点。
    2. 默认匹配："类型为" → "故障" 模式，覆盖常见工业故障分类。
    """

    DEFAULT_FAULT_RELATIONS = [
        "类型为", "type", "rdf:type", "类别为", "分类为",
        "is_fault", "故障类型", "fault_type",
    ]
    DEFAULT_FAULT_TAILS = ["故障", "fault", "Failure", "异常", "失效"]

    def __init__(
            self,
            vocab: "KGVocabulary",
            fault_relations: Optional[List[str]] = None,
            fault_tails: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        vocab : KGVocabulary
            全局词汇表，已包含所有实体和关系类型。
        fault_relations : Optional[List[str]]
            指示故障分类的关系名称列表。
        fault_tails : Optional[List[str]]
            故障类别的尾实体名称列表。
        """
        self.vocab = vocab
        self.fault_relations = fault_relations or self.DEFAULT_FAULT_RELATIONS
        self.fault_tails = fault_tails or self.DEFAULT_FAULT_TAILS

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

        遍历所有三元组，当 relation ∈ fault_relations 且 tail ∈ fault_tails 时，
        将 head 实体标记为故障节点。

        Returns
        -------
        Tuple[torch.Tensor, List[str]]
            (y: 标签张量 [num_entities], fault_nodes: 故障节点名称列表)
        """
        _fault_relation_set = set(self.fault_relations)
        _fault_tail_set = set(self.fault_tails)
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
                if rel in _fault_relation_set and tail in _fault_tail_set:
                    if head in self.vocab.entity2idx:
                        fault_set.add(head)
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
                "未识别到任何故障节点！请检查 fault_relations=%s 和 fault_tails=%s "
                "是否与知识图谱中的实际关系/尾实体匹配",
                self.fault_relations, self.fault_tails,
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

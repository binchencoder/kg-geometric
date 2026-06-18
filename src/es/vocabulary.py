"""全局词汇表管理器 —— 管理实体和关系的全局 ID 映射。

支持增量构建、持久化、名称质量启发式检测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..core.config import logger


class KGVocabulary:
    """管理实体和关系的全局 ID 映射。

    支持增量构建和持久化，避免每次训练都重新扫描。
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.entity2idx: Dict[str, int] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.idx2relation: Dict[int, str] = {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        # 动态注入属性，供 _warn_if_names_look_like_ids 使用
        self._entity_index_hint: str = "knowledge_entity_index"
        self._relation_type_index_hint: str = "knowledge_entity_type_relation_index"

    @property
    def num_entities(self) -> int:
        return len(self.entity2idx)

    @property
    def num_relations(self) -> int:
        return len(self.relation2idx)

    def add_entity(self, entity: str) -> int:
        """添加实体，返回其索引。"""
        if entity not in self.entity2idx:
            idx = len(self.entity2idx)
            self.entity2idx[entity] = idx
            self.idx2entity[idx] = entity
        return self.entity2idx[entity]

    def add_relation(self, relation: str) -> int:
        """添加关系类型，返回其索引。"""
        if relation not in self.relation2idx:
            idx = len(self.relation2idx)
            self.relation2idx[relation] = idx
            self.idx2relation[idx] = relation
        return self.relation2idx[relation]

    def build_from_streamer(
            self,
            streamer,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            extra_filters: Optional[dict] = None,
    ) -> int:
        """从 ESTripletStreamer 流式构建词汇表。

        Parameters
        ----------
        extra_filters : Optional[dict]
            额外 ES 查询过滤条件，如 {"match": {"graphId": "xxx"}}。

        Returns
        -------
        int
            总三元组数。
        """
        total = 0
        for batch in streamer.stream_triplets(
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
                extra_filters=extra_filters,
        ):
            for t in batch:
                self.add_entity(t["head"])
                self.add_entity(t["tail"])
                self.add_relation(t["relation"])
                total += 1

            if total % 100000 == 0:
                logger.info(
                    "词汇表构建中... 实体: %d, 关系: %d, 三元组: %d",
                    self.num_entities, self.num_relations, total,
                )

        logger.info(
            "词汇表构建完成: 实体=%d, 关系=%d, 三元组=%d",
            self.num_entities, self.num_relations, total,
        )
        self._warn_if_names_look_like_ids()
        return total

    # ---------- 名称校验 ----------
    @staticmethod
    def _looks_like_id(s: str) -> bool:
        """启发式检测：字符串是否看起来像原始 ID 而非人类可读名称。

        规则：
        - 纯数字串（如 "1234567890"）
        - 不含任何中文字符且长度 > 20 的字母数字串
        """
        if not s:
            return False
        if any('\u4e00' <= c <= '\u9fff' for c in s):
            return False
        if s.isdigit() and len(s) >= 10:
            return True
        if len(s) >= 20 and all(c.isalnum() or c in '-_' for c in s):
            return True
        return False

    def _warn_if_names_look_like_ids(self) -> None:
        """抽样检测词汇表中的键是否看起来像 ID，若是则发出明确警告。"""
        entity_keys = list(self.entity2idx.keys())
        relation_keys = list(self.relation2idx.keys())

        entity_sample = entity_keys[:20]
        relation_sample = relation_keys[:20]

        entity_id_count = sum(1 for e in entity_sample if self._looks_like_id(e))
        relation_id_count = sum(1 for r in relation_sample if self._looks_like_id(r))

        if entity_id_count > len(entity_sample) // 2 and entity_sample:
            logger.warning(
                "⚠️  词汇表中 %d/%d 个实体名称看起来像原始 ID（如 %s ...），"
                "请检查：\n"
                "  1. 是否启用了 --no-resolve？\n"
                "  2. --entity-name-field 是否配置正确？（当前可能不是 'name'）\n"
                "  3. 实体索引 %s 的 ID/名称字段是否匹配？\n"
                "  可使用 --entity-name-field <字段名> 指定正确的名称字段。",
                entity_id_count, len(entity_sample),
                entity_sample[0][:40] if entity_sample else "",
                getattr(self, '_entity_index_hint', 'knowledge_entity_index'),
            )

        if relation_id_count > len(relation_sample) // 2 and relation_sample:
            logger.warning(
                "⚠️  词汇表中 %d/%d 个关系类型名称看起来像原始 ID（如 %s ...），"
                "请检查：\n"
                "  1. 是否启用了 --no-resolve？\n"
                "  2. --relation-type-name-field 是否配置正确？（当前可能不是 'name'）\n"
                "  3. 关系类型索引 %s 的 ID/名称字段是否匹配？\n"
                "  可使用 --relation-type-name-field <字段名> 指定正确的名称字段。",
                relation_id_count, len(relation_sample),
                relation_sample[0][:40] if relation_sample else "",
                getattr(self, '_relation_type_index_hint', 'knowledge_entity_type_relation_index'),
            )

    def save(self, filepath: str) -> None:
        """持久化词汇表到磁盘。保存前校验名称质量。"""
        self._warn_if_names_look_like_ids()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entity2idx": self.entity2idx,
            "relation2idx": self.relation2idx,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("词汇表已保存至 %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "KGVocabulary":
        """从磁盘加载词汇表。"""
        with open(filepath) as f:
            data = json.load(f)
        vocab = cls()
        vocab.entity2idx = data["entity2idx"]
        vocab.relation2idx = data["relation2idx"]
        vocab.idx2entity = {v: k for k, v in vocab.entity2idx.items()}
        vocab.idx2relation = {v: k for k, v in vocab.relation2idx.items()}
        logger.info(
            "词汇表已加载: 实体=%d, 关系=%d",
            vocab.num_entities, vocab.num_relations,
        )
        return vocab

"""ID→名称解析器 —— 将 ES 中的原始 ID 映射为人类可读名称。

从 ES 实体/关系类型索引构建 ID→名称双向映射，用于三元组 ID 解析。
"""

from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from ..core.config import logger


class IDNameResolver:
    """从 ES 实体/关系类型索引构建 ID→名称映射，用于三元组 ID 解析。

    对齐原 ESKnowledgeGraphReader 的三步查询逻辑（第 1-2 步）：
    1. 扫描 entity_index → 构建 entityId → entityName 映射
    2. 扫描 relation_type_index → 构建 relationTypeId → relationName 映射

    这两个索引通常较小，使用 scroll/scan 即可，无需 search_after。
    """

    def __init__(self, es: Elasticsearch):
        self.es = es
        self.entity_map: Dict[str, str] = {}  # entityId → name
        self.relation_type_map: Dict[str, str] = {}  # relationTypeId → name
        self.entity_name_to_id: Dict[str, str] = {}  # name → entityId (反向映射)
        self.relation_name_to_id: Dict[str, str] = {}  # name → relationTypeId (反向映射)

    def _index_exists(self, index: str) -> bool:
        try:
            return self.es.indices.exists(index=index)
        except Exception:
            return False

    def _scan_index(
            self,
            index: str,
            query: dict,
            batch_size: int = 5000,
    ) -> Iterator[dict]:
        """流式扫描索引（scroll 方式，适合小索引）。"""
        if not self._index_exists(index):
            raise RuntimeError(
                f"索引 '{index}' 不存在，请确认索引名称。"
            )

        yield from scan(
            client=self.es,
            index=index,
            query={"query": query},
            size=batch_size,
            scroll="10m",
            preserve_order=False,
        )

    def _sample_index_docs(self, index: str, size: int = 3) -> list:
        """采样索引中的前几篇文档，用于诊断字段名/数据格式。"""
        try:
            resp = self.es.search(
                index=index,
                body={"query": {"match_all": {}}, "size": size},
                request_timeout=10,
            )
            return [hit["_source"] for hit in resp["hits"]["hits"]]
        except Exception as e:
            logger.warning("采样索引 '%s' 失败: %s", index, e)
            return []

    @staticmethod
    def _safe_get(source: dict, field: str, default: str = "") -> str:
        """安全取值，支持嵌套字段。"""
        if field in source:
            val = source[field]
            return str(val).strip() if val is not None else default
        parts = field.split(".")
        current = source
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return str(current).strip() if current is not None else default

    def _populate_map_from_scan(
            self,
            index: str,
            query: dict,
            id_field: str,
            name_field: str,
            target_map: Dict[str, str],
            reverse_map: Optional[Dict[str, str]] = None,
            batch_size: int = 5000,
    ) -> int:
        """从扫描结果中填充 ID→名称映射。"""
        count = 0
        for doc in self._scan_index(
                index=index,
                query=query,
                batch_size=batch_size,
        ):
            try:
                source = doc.get("_source", doc)
                eid = self._safe_get(source, id_field)
                name = self._safe_get(source, name_field)
                if eid and name:
                    target_map[eid] = name
                    if reverse_map is not None:
                        reverse_map[name] = eid
                    count += 1
            except Exception as e:
                logger.warning("解析文档 %s 失败: %s", doc.get("_id", "?"), e)
        return count

    def _diagnose_empty_map(
            self,
            index: str,
            id_field: str,
            name_field: str,
            filter_key: str,
            filter_value: str,
    ) -> None:
        """当映射为空时，采样索引文档以帮助诊断问题。"""
        samples = self._sample_index_docs(index, size=3)
        if samples:
            logger.warning(
                "⚠️  映射为空！采样索引 '%s' 的文档:\n"
                "  过滤条件: %s=%s\n"
                "  期望字段: id=%s, name=%s\n"
                "  实际文档字段: %s",
                index, filter_key, filter_value,
                id_field, name_field,
                list(samples[0].keys()) if samples else "N/A",
            )
            for i, doc in enumerate(samples):
                logger.warning(
                    "  样本%d: %s",
                    i + 1,
                    {k: str(v)[:80] for k, v in doc.items()},
                )
        else:
            logger.warning(
                "⚠️  映射为空且无法采样！索引 '%s' 可能为空或不可访问。",
                index,
            )

    def build_entity_map(
            self,
            entity_index: str = "knowledge_entity_index",
            graph_id: Optional[str] = "980044155496734720",
            id_field: str = "entityId",
            name_field: str = "name",
            batch_size: int = 5000,
            extra_query: Optional[dict] = None,
            debug: bool = False,
    ) -> int:
        """扫描实体索引，构建 entityId → name 映射。

        Returns
        -------
        int
            加载的实体数量。
        """
        self.entity_map.clear()
        self.entity_name_to_id.clear()

        if extra_query:
            query = extra_query
        elif graph_id:
            query = {"match": {"graphId": graph_id}}
        else:
            query = {"match_all": {}}

        count = self._populate_map_from_scan(
            index=entity_index,
            query=query,
            id_field=id_field,
            name_field=name_field,
            target_map=self.entity_map,
            reverse_map=self.entity_name_to_id,
            batch_size=batch_size,
        )

        logger.info("从索引 %s 读取了 %d 个实体 (ID↔名称, id_field=%s, name_field=%s)",
                    entity_index, count, id_field, name_field)

        # ---------- 空结果诊断与自动 fallback ----------
        if count == 0:
            self._diagnose_empty_map(
                entity_index, id_field, name_field,
                "graphId", str(graph_id or "N/A"),
            )
            has_filter = bool(extra_query) or bool(graph_id)
            if has_filter:
                logger.warning(
                    "⚠️  实体映射在 graph_id 过滤下为空，尝试全量扫描 (match_all) ..."
                )
                count = self._populate_map_from_scan(
                    index=entity_index,
                    query={"match_all": {}},
                    id_field=id_field,
                    name_field=name_field,
                    target_map=self.entity_map,
                    reverse_map=self.entity_name_to_id,
                    batch_size=batch_size,
                )
                if count > 0:
                    logger.warning(
                        "⚠️  全量扫描成功：读取了 %d 个实体。"
                        "请确认 --graph-id 参数是否正确（当前: %s）。"
                        "实体的 graphId 字段值可能与索引中不一致。",
                        count, graph_id,
                    )
                else:
                    logger.warning(
                        "⚠️  全量扫描仍然为空！"
                        "请使用 --resolve-debug 查看索引文档详情，"
                        "或通过 --entity-id-field / --entity-name-field 指定正确字段名。",
                    )

        if debug:
            samples = self._sample_index_docs(entity_index, size=3)
            logger.info(
                "[DEBUG] 实体索引 '%s' 样本文档字段: %s",
                entity_index,
                [list(d.keys()) for d in samples],
            )

        return count

    def build_relation_type_map(
            self,
            relation_type_index: str = "knowledge_entity_type_relation_index",
            ontology_id: Optional[str] = "979748419706068992",
            id_field: str = "relationTypeId",
            name_field: str = "name",
            batch_size: int = 5000,
            extra_query: Optional[dict] = None,
            debug: bool = False,
    ) -> int:
        """扫描关系类型索引，构建 relationTypeId → name 映射。

        Returns
        -------
        int
            加载的关系类型数量。
        """
        self.relation_type_map.clear()
        self.relation_name_to_id.clear()

        if extra_query:
            query = extra_query
        elif ontology_id:
            query = {"match": {"ontologyId": ontology_id}}
        else:
            query = {"match_all": {}}

        count = self._populate_map_from_scan(
            index=relation_type_index,
            query=query,
            id_field=id_field,
            name_field=name_field,
            target_map=self.relation_type_map,
            reverse_map=self.relation_name_to_id,
            batch_size=batch_size,
        )

        logger.info(
            "从索引 %s 读取了 %d 种关系类型 (ID↔名称, id_field=%s, name_field=%s)",
            relation_type_index, count, id_field, name_field,
        )

        # ---------- 空结果诊断与自动 fallback ----------
        if count == 0:
            self._diagnose_empty_map(
                relation_type_index, id_field, name_field,
                "ontologyId", str(ontology_id or "N/A"),
            )
            has_filter = bool(extra_query) or bool(ontology_id)
            if has_filter:
                logger.warning(
                    "⚠️  关系类型映射在 ontology_id 过滤下为空，尝试全量扫描 (match_all) ..."
                )
                count = self._populate_map_from_scan(
                    index=relation_type_index,
                    query={"match_all": {}},
                    id_field=id_field,
                    name_field=name_field,
                    target_map=self.relation_type_map,
                    reverse_map=self.relation_name_to_id,
                    batch_size=batch_size,
                )
                if count > 0:
                    logger.warning(
                        "⚠️  全量扫描成功：读取了 %d 种关系类型。"
                        "请确认 --ontology-id 参数是否正确（当前: %s）。"
                        "关系类型的 ontologyId 字段值可能与索引中不一致。",
                        count, ontology_id,
                    )
                else:
                    logger.warning(
                        "⚠️  全量扫描仍然为空！"
                        "请使用 --resolve-debug 查看索引文档详情，"
                        "或通过 --relation-type-id-field / --relation-type-name-field 指定正确字段名。",
                    )

        if debug:
            samples = self._sample_index_docs(relation_type_index, size=3)
            logger.info(
                "[DEBUG] 关系类型索引 '%s' 样本文档字段: %s",
                relation_type_index,
                [list(d.keys()) for d in samples],
            )

        return count

    def resolve_entity_names_to_ids(
            self,
            names: List[str],
    ) -> List[str]:
        """将实体名称反向解析为 ES 中的原始 entityId。

        用于 AsyncSubgraphSampler 子图查询：采样器持有的是名称，
        但 ES 关系索引 store 的是 ID，必须反解后才能构建 terms 查询。

        Parameters
        ----------
        names : List[str]
            实体名称列表。

        Returns
        -------
        List[str]
            对应的原始 entityId 列表（无法反解的保留原名作为 fallback）。
        """
        return [self.entity_name_to_id.get(n, n) for n in names]

    def resolve_triplets(
            self,
            triplets: List[dict],
    ) -> Tuple[List[dict], int, int]:
        """将原始 ID 三元组解析为名称三元组。

        对齐原 _load_relations_as_triples 的解析逻辑：
        - 头尾实体 ID → entity_map → 名称
        - 关系类型 ID → relation_type_map → 名称
        - 任一无法解析则跳过
        - 按 (head, relation, tail) 去重

        Parameters
        ----------
        triplets : List[dict]
            原始 ID 三元组列表，每项 {"head": id, "relation": id, "tail": id}。

        Returns
        -------
        Tuple[List[dict], int, int]
            (解析后的名称三元组列表, 解析成功数, 跳过数)
        """
        resolved: List[dict] = []
        skipped = 0
        seen: Set[Tuple[str, str, str]] = set()
        _unresolved_samples: List[Tuple[str, str, str]] = []

        for t in triplets:
            head_id = t["head"]
            tail_id = t["tail"]
            relation_id = t["relation"]

            head_name = self.entity_map.get(head_id)
            tail_name = self.entity_map.get(tail_id)
            relation_name = self.relation_type_map.get(relation_id)

            if head_name is None or tail_name is None or relation_name is None:
                skipped += 1
                if len(_unresolved_samples) < 3:
                    _unresolved_samples.append((head_id, relation_id, tail_id))
                continue

            # 去重
            dedup_key = (head_name, relation_name, tail_name)
            if dedup_key in seen:
                skipped += 1
                continue
            seen.add(dedup_key)

            resolved.append({
                "head": head_name,
                "relation": relation_name,
                "tail": tail_name,
            })

        if skipped > 0 and len(resolved) == 0:
            entity_keys_sample = list(self.entity_map.keys())[:5] if self.entity_map else []
            rel_keys_sample = list(self.relation_type_map.keys())[:5] if self.relation_type_map else []
            sample = _unresolved_samples[0] if _unresolved_samples else ("?", "?", "?")
            logger.warning(
                "⚠️  本批次 %d 条三元组全部无法解析为名称！\n"
                "  未解析样本: head_id=%s, relation_id=%s, tail_id=%s\n"
                "  entity_map 样本键(前5): %s\n"
                "  relation_type_map 样本键(前5): %s\n"
                "  → 检查 entity_map/relation_type_map 的 ID 是否与关系索引一致。",
                skipped,
                sample[0], sample[1], sample[2],
                entity_keys_sample,
                rel_keys_sample,
            )
        elif skipped > 0 and skipped >= len(resolved) * 2:
            logger.warning(
                "⚠️  本批次 %d 条跳过 (%.0f%%)，部分 ID 无法解析。"
                "样本: %s。可能 ID/名称字段不匹配。",
                skipped, skipped / max(skipped + len(resolved), 1) * 100,
                _unresolved_samples[:2] if _unresolved_samples else "?",
            )

        return resolved, len(resolved), skipped

    @property
    def is_ready(self) -> bool:
        """两个映射均已构建完成。"""
        return bool(self.entity_map) and bool(self.relation_type_map)

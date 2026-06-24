"""ES 知识图谱读取器 —— scroll/scan 模式。

提供 ESKnowledgeGraphReader 类，通过三步查询法从 ES 读取实体/关系数据
并构建标准三元组。
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from src.core.config import ESConfig, BatchProgress, logger
from src.core.types import Triple
from .client import (
    create_es_client,
    ping_es,
    index_exists,
    list_indices,
    get_index_mapping,
    safe_get,
)


class ESKnowledgeGraphReader:
    """从 Elasticsearch 读取知识图谱数据并构建三元组。

    支持的索引模式：
    - 实体索引：包含节点 ID、名称、类型等字段
    - 关系索引：包含 head_id、relation、tail_id 等字段
    - 也可以从单索引中读取已组合的三元组数据

    使用示例::

        reader = ESKnowledgeGraphReader()
        triples = reader.fetch_triples(
            entity_index="kg_entities",
            relation_index="kg_relations",
        )
    """

    def __init__(self, config: Optional[ESConfig] = None) -> None:
        self.config = config or ESConfig()
        self._client: Optional[Elasticsearch] = None

    # ---- 连接管理 ----
    @property
    def client(self) -> Elasticsearch:
        if self._client is None:
            self._client = self._create_client()
            self._ping()
        return self._client

    def _create_client(self) -> Elasticsearch:
        return create_es_client(self.config)

    def _ping(self) -> None:
        """检测 ES 连通性，失败时警告但不阻断。"""
        if self._client is None:
            raise ConnectionError("ES 客户端未初始化")
        ping_es(self._client, self.config.host, self.config.port)

    def close(self) -> None:
        """关闭 ES 连接。"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Elasticsearch 连接已关闭")

    # ---- 索引发现 ----
    def list_indices(self, pattern: str = "*") -> List[str]:
        """列出匹配模式的所有索引。"""
        return list_indices(self.client, pattern)

    def get_index_mapping(self, index_name: str) -> dict:
        """获取索引的 mapping 信息，便于自动推断字段名。"""
        return get_index_mapping(self.client, index_name)

    # ---- 核心：从 ES 批量读取并构建三元组 ----
    def fetch_triples(
            self,
            entity_index: str = "knowledge_entity_index",
            relation_index: str = "knowledge_entity_relation_index",
            relation_type_index: str = "knowledge_entity_type_relation_index",
            batch_size: int = 5000,
            query: Optional[dict] = None,
            entity_id_field: str = "entityId",
            entity_name_field: str = "name",
            head_id_field: str = "srcEntityId",
            tail_id_field: str = "dstEntityId",
            relation_field: str = "relationTypeId",
            relation_type_id_field: str = "relationTypeId",
            relation_type_name_field: str = "name",
    ) -> List[Triple]:
        """从 ES 实体/关系索引读取数据并构建三元组列表。

        查询流程（三步）：
        1. 从 entity_index 读取实体，构建 entityId → name 映射
        2. 从 relation_type_index 读取关系类型，构建 relationTypeId → name 映射
        3. 从 relation_index 读取关系，用 entity_map + relation_type_map 解析

        Parameters
        ----------
        entity_index : str
            实体索引名称。
        relation_index : str
            关系索引名称。
        relation_type_index : str
            关系类型索引名称，用于将 relation_field 的 id 映射为可读名称。
        batch_size : int
            scan 每批文档数。
        query : Optional[dict]
            额外的 ES 查询过滤条件。
        entity_id_field : str
            实体文档中的 ID 字段名。
        entity_name_field : str
            实体文档中的名称字段名。
        head_id_field : str
            关系文档中的头实体 ID 字段名。
        tail_id_field : str
            关系文档中的尾实体 ID 字段名。
        relation_field : str
            关系文档中的关系类型 ID 字段名。
        relation_type_id_field : str
            关系类型索引中用作映射 key 的字段名。
        relation_type_name_field : str
            关系类型索引中用作映射 value 的字段名。

        Returns
        -------
        List[Triple]
            标准三元组列表。
        """
        logger.info("=" * 60)
        logger.info("开始从 ES 读取知识图谱数据...")
        logger.info("实体索引: %s, 关系索引: %s, 关系类型索引: %s",
                    entity_index, relation_index, relation_type_index)
        logger.info("=" * 60)

        # Step 1: 读取所有实体，构建 ID -> Name 映射
        entity_map = self._load_entity_map(
            entity_index=entity_index,
            batch_size=batch_size,
            query=query,
            id_field=entity_id_field,
            name_field=entity_name_field,
        )
        logger.info("实体映射构建完成，共 %d 个实体", len(entity_map))

        # Step 2: 读取关系类型，构建 relationTypeId -> Name 映射
        relation_type_map = self._load_relation_type_map(
            relation_type_index=relation_type_index,
            batch_size=batch_size,
            query=query,
            id_field=relation_type_id_field,
            name_field=relation_type_name_field,
        )
        logger.info("关系类型映射构建完成，共 %d 种关系类型", len(relation_type_map))

        # Step 3: 批量读取关系，用 entity_map + relation_type_map 解析为三元组
        triples, progress = self._load_relations_as_triples(
            relation_index=relation_index,
            entity_map=entity_map,
            relation_type_map=relation_type_map,
            batch_size=batch_size,
            query=query,
            head_id_field=head_id_field,
            tail_id_field=tail_id_field,
            relation_field=relation_field,
        )

        logger.info("进度统计: %s", progress.summary())
        if progress.errors > 0:
            logger.warning("处理过程中出现 %d 个错误，详情:", progress.errors)
            for detail in progress.details[-10:]:
                logger.warning("  - %s", detail)

        return triples

    def fetch_triples_from_single_index(
            self,
            index: str = "knowledge_entity_relation_index",
            relation_type_index: str = "knowledge_entity_type_relation_index",
            batch_size: int = 5000,
            query: Optional[dict] = None,
            head_field: str = "srcEntityId",
            relation_field: str = "relationId",
            tail_field: str = "dstEntityId",
            relation_type_id_field: str = "relationTypeId",
            relation_type_name_field: str = "name",
    ) -> List[Triple]:
        """从单个索引读取三元组，并通过 relation_type_index 将关系 ID 解析为名称。

        适用于 ES 中已按 (head, relation_id, tail) 结构存储数据的场景，
        其中 relation_id 需要通过 relation_type_index 二次查询获取可读名称。
        """
        logger.info("=" * 60)
        logger.info("从单索引 %s 读取三元组，关系类型索引: %s ...", index, relation_type_index)
        logger.info("=" * 60)

        # Step 1: 读取关系类型映射
        relation_type_map = self._load_relation_type_map(
            relation_type_index=relation_type_index,
            batch_size=batch_size,
            query=query,
            id_field=relation_type_id_field,
            name_field=relation_type_name_field,
        )
        logger.info("关系类型映射构建完成，共 %d 种关系类型", len(relation_type_map))

        progress = BatchProgress()
        triples: List[Triple] = []
        base_query = query or {"match_all": {}}
        seen = set()

        for doc in self._scan_index(index=index, query=base_query, batch_size=batch_size):
            progress.total_docs += 1
            try:
                source = doc.get("_source", doc)
                head = safe_get(source, head_field)
                relation_id = safe_get(source, relation_field)
                tail = safe_get(source, tail_field)

                if not head or not relation_id or not tail:
                    progress.skipped_docs += 1
                    continue

                # 二次查询：将 relation ID 映射为关系名称
                relation_name = relation_type_map.get(relation_id, relation_id)
                if relation_name == relation_id:
                    logger.debug("关系类型 ID %s 未在 %s 中找到映射，使用原 ID",
                                 relation_id, relation_type_index)

                dedup_key = (head, relation_name, tail)
                if dedup_key in seen:
                    progress.skipped_docs += 1
                    continue
                seen.add(dedup_key)

                triples.append(Triple(head=head, relation=relation_name, tail=tail))
                progress.valid_triples += 1

            except Exception as e:
                progress.errors += 1
                progress.details.append(f"文档 {doc.get('_id', '?')}: {e}")

        logger.info("进度统计: %s", progress.summary())
        return triples

    # ---- 内部方法 ----
    def _load_entity_map(
            self,
            entity_index: str,
            batch_size: int,
            query: Optional[dict],
            id_field: str,
            name_field: str,
    ) -> Dict[str, str]:
        """扫描实体索引，构建 ID->Name 映射。"""
        entity_map: Dict[str, str] = {}
        base_query = query or {
            "match": {
                "graphId": "980044155496734720"
            }
        }
        count = 0

        for doc in self._scan_index(index=entity_index, query=base_query, batch_size=batch_size):
            try:
                source = doc.get("_source", doc)
                eid = str(safe_get(source, id_field))
                name = str(safe_get(source, name_field))
                if eid and name:
                    entity_map[eid] = name
                    count += 1
            except Exception as e:
                logger.warning("解析实体文档 %s 失败: %s", doc.get("_id", "?"), e)

        logger.info("从索引 %s 读取了 %d 个实体", entity_index, count)
        return entity_map

    def _load_relation_type_map(
            self,
            relation_type_index: str,
            batch_size: int,
            query: Optional[dict],
            id_field: str,
            name_field: str,
    ) -> Dict[str, str]:
        """扫描关系类型索引，构建 relationTypeId -> Name 映射。"""
        relation_type_map: Dict[str, str] = {}
        base_query = query or {
            "match": {
                "ontologyId": "979748419706068992"
            }
        }
        count = 0

        for doc in self._scan_index(
                index=relation_type_index,
                query=base_query,
                batch_size=batch_size,
        ):
            try:
                source = doc.get("_source", doc)
                type_id = str(safe_get(source, id_field))
                type_name = str(safe_get(source, name_field))
                if type_id and type_name:
                    relation_type_map[type_id] = type_name
                    count += 1
            except Exception as e:
                logger.warning("解析关系类型文档 %s 失败: %s", doc.get("_id", "?"), e)

        logger.info("从索引 %s 读取了 %d 种关系类型", relation_type_index, count)
        return relation_type_map

    def _load_relations_as_triples(
            self,
            relation_index: str,
            entity_map: Dict[str, str],
            relation_type_map: Dict[str, str],
            batch_size: int,
            query: Optional[dict],
            head_id_field: str,
            tail_id_field: str,
            relation_field: str,
    ) -> Tuple[List[Triple], BatchProgress]:
        """扫描关系索引，用 entity_map + relation_type_map 解析为三元组。"""
        progress = BatchProgress()
        triples: List[Triple] = []
        base_query = query or {
            "match": {
                "graphId": "980044155496734720"
            }
        }
        seen = set()
        unresolved_relations: set = set()

        for doc in self._scan_index(index=relation_index, query=base_query, batch_size=batch_size):
            progress.total_docs += 1
            try:
                source = doc.get("_source", doc)
                head_id = str(safe_get(source, head_id_field))
                tail_id = str(safe_get(source, tail_id_field))
                relation_id = str(safe_get(source, relation_field))

                # 校验字段完整性
                if not head_id or not tail_id or not relation_id:
                    progress.skipped_docs += 1
                    continue

                # 二次查询：将 relation ID 映射为关系名称
                relation_name = relation_type_map.get(relation_id)
                if relation_name is None:
                    unresolved_relations.add(relation_id)
                    progress.skipped_docs += 1
                    logger.debug(
                        "关系类型 ID %s 在 relation_type_map 中未找到映射，跳过",
                        relation_id,
                    )
                    continue

                head_name = entity_map.get(head_id)
                tail_name = entity_map.get(tail_id)

                if head_name is None:
                    logger.debug("头实体 ID %s 在实体映射中未找到，跳过", head_id)
                    progress.skipped_docs += 1
                    continue
                if tail_name is None:
                    logger.debug("尾实体 ID %s 在实体映射中未找到，跳过", tail_id)
                    progress.skipped_docs += 1
                    continue

                # 去重
                dedup_key = (head_name, relation_name, tail_name)
                if dedup_key in seen:
                    progress.skipped_docs += 1
                    continue
                seen.add(dedup_key)

                triples.append(Triple(head=head_name, relation=relation_name, tail=tail_name))
                progress.valid_triples += 1

            except Exception as e:
                progress.errors += 1
                progress.details.append(
                    f"关系文档 {doc.get('_id', '?')}: {e}"
                )

        if unresolved_relations:
            logger.warning(
                "有 %d 个关系类型 ID 未能映射到名称: %s",
                len(unresolved_relations),
                sorted(unresolved_relations)[:20],
            )

        return triples, progress

    def _scan_index(
            self,
            index: str,
            query: dict,
            batch_size: int,
    ) -> Iterator[dict]:
        """使用 elasticsearch.helpers.scan 实现高效的批量滚动读取。"""
        if not index_exists(self.client, index):
            available = list_indices(self.client)
            raise RuntimeError(
                f"索引 '{index}' 不存在，请确认索引名称是否正确。"
                f"当前可用索引: {available}"
            )

        try:
            yield from scan(
                client=self.client,
                index=index,
                query={"query": query},
                size=batch_size,
                scroll="10m",
                preserve_order=False,
            )
        except Exception as e:
            logger.error("扫描索引 %s 时发生异常: %s", index, e)
            raise

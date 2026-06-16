"""
Elasticsearch 知识图谱数据读取与三元组构建模块。

功能：
1. 连接 Elasticsearch 并读取实体与关系数据
2. 解析数据提取 (头实体, 关系, 尾实体) 标准三元组
3. 将三元组转换为模型训练所需的数据结构
4. 批量读取与处理机制以应对工业级海量数据
5. 异常处理确保数据完整性
6. search_after 流式提取器（解决上亿数据深分页性能问题）
7. 异步流式采样 + PyG NeighborLoader "边查边训" 流水线
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ESKGReader")


# -------------------- ES 连接配置 --------------------
@dataclass(frozen=True)
class ESConfig:
    """Elasticsearch 连接配置。"""
    host: str = "10.1.13.30"
    port: int = 30920
    username: str = "elastic"
    password: str = "E1OfAx4Nf55513tU4i40eQbA"
    scheme: str = "http"
    timeout: int = 60
    max_retries: int = 3
    retry_on_timeout: bool = True


# -------------------- 三元组数据类（与 kg_fault_diagnosis.py 兼容） --------------------
@dataclass(frozen=True)
class Triple:
    """知识图谱三元组：头实体 --关系--> 尾实体。"""
    head: str
    relation: str
    tail: str


# -------------------- 批量进度回调 --------------------
@dataclass
class BatchProgress:
    """批量处理进度统计。"""
    total_docs: int = 0
    valid_triples: int = 0
    skipped_docs: int = 0
    errors: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        if self.total_docs == 0:
            return 0.0
        return self.valid_triples / self.total_docs

    def summary(self) -> str:
        return (
            f"总文档: {self.total_docs} | 有效三元组: {self.valid_triples} | "
            f"跳过: {self.skipped_docs} | 错误: {self.errors} | "
            f"有效率: {self.valid_ratio:.2%}"
        )


# -------------------- ES 知识图谱读取器 --------------------
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
        dataset = reader.build_dataset(triples, fault_types=["故障"])
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
        return Elasticsearch(
            hosts=[{
                "host": self.config.host,
                "port": self.config.port,
                "scheme": self.config.scheme,
            }],
            # elasticsearch-py 7.x 使用 http_auth 进行 Basic 认证
            http_auth=(self.config.username, self.config.password),
            request_timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_on_timeout=self.config.retry_on_timeout,
            # 关闭节点嗅探，避免非标准端口（如 K8s NodePort）探测失败
            sniff_on_start=False,
            sniff_on_connection_fail=False,
            # 对于非标准端口上的 http 连接，跳过节点验证
            verify_certs=False,
        )

    def _ping(self) -> None:
        """检测 ES 连通性，失败时警告但不阻断（部分代理/网关可能拦截 ping）。"""
        if self._client is None:
            raise ConnectionError("ES 客户端未初始化")
        try:
            if self._client.ping():
                logger.info("Elasticsearch 连接成功: %s:%s", self.config.host, self.config.port)
            else:
                logger.warning(
                    "Elasticsearch ping 返回 False（%s:%s），可能是代理/网关拦截，"
                    "后续索引操作仍会重试",
                    self.config.host,
                    self.config.port,
                )
        except Exception as e:
            logger.warning(
                "Elasticsearch ping 异常（%s:%s）: %s。"
                "将跳过连通性检测，实际读写时如失败会抛出异常",
                self.config.host,
                self.config.port,
                e,
            )

    def close(self) -> None:
        """关闭 ES 连接。"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Elasticsearch 连接已关闭")

    # ---- 索引发现 ----
    def list_indices(self, pattern: str = "*") -> List[str]:
        """列出匹配模式的所有索引。"""
        try:
            cat = self.client.cat.indices(index=pattern, format="json")
            return [item["index"] for item in cat]
        except Exception as e:
            logger.error("列出索引失败: %s", e)
            return []

    def get_index_mapping(self, index_name: str) -> dict:
        """获取索引的 mapping 信息，便于自动推断字段名。"""
        try:
            resp = self.client.indices.get_mapping(index=index_name)
            return resp.body if hasattr(resp, "body") else resp
        except Exception as e:
            logger.error("获取索引 %s mapping 失败: %s", index_name, e)
            return {}

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
        3. 从 relation_index 读取关系，用 entity_map 解析头尾实体名称，
           用 relation_map 将 relationId 解析为关系名称

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
            关系文档中的关系类型 ID 字段名（对应 relation_type_index 中的 relationTypeId）。
        relation_type_id_field : str
            关系类型索引中用作映射 key 的字段名。
        relation_type_name_field : str
            关系类型索引中用作映射 value（关系名称）的字段名。

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
                head = self._safe_get(source, head_field)
                relation_id = self._safe_get(source, relation_field)
                tail = self._safe_get(source, tail_field)

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
                eid = str(self._safe_get(source, id_field))
                name = str(self._safe_get(source, name_field))
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
        """扫描关系类型索引，构建 relationTypeId -> Name 映射。

        从 knowledge_entity_type_relation_index 中读取所有关系类型定义，
        建立 relationTypeId 到 name 的映射表，供 _load_relations_as_triples
        解析关系 ID 时使用。
        """
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
                type_id = str(self._safe_get(source, id_field))
                type_name = str(self._safe_get(source, name_field))
                if type_id and type_name:
                    relation_type_map[type_id] = type_name
                    count += 1
            except Exception as e:
                logger.warning(
                    "解析关系类型文档 %s 失败: %s", doc.get("_id", "?"), e
                )

        logger.info(
            "从索引 %s 读取了 %d 种关系类型", relation_type_index, count
        )
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
        """扫描关系索引，用 entity_map + relation_type_map 解析为三元组。

        将关系文档中的 srcEntityId/dstEntityId 通过 entity_map 解析为实体名称，
        将 relationId 通过 relation_type_map 解析为关系类型名称。
        """
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
                head_id = str(self._safe_get(source, head_id_field))
                tail_id = str(self._safe_get(source, tail_id_field))
                relation_id = str(self._safe_get(source, relation_field))

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
                        "关系类型 ID %s 在 %s 中未找到映射，跳过",
                        relation_id,
                        "relation_type_map",
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
        """使用 elasticsearch.helpers.scan 实现高效的批量滚动读取。

        内置索引存在性检测。
        """
        # 先验证索引是否存在，给出明确提示
        if not self._index_exists(index):
            available = self.list_indices()
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

    def _index_exists(self, index: str) -> bool:
        """检测索引是否存在。"""
        try:
            return self.client.indices.exists(index=index)
        except Exception:
            return False

    @staticmethod
    def _safe_get(source: dict, field: str, default: str = "") -> str:
        """安全地从文档中取值，支持嵌套字段（如 obj.nested.key）。"""
        if field in source:
            val = source[field]
            return str(val).strip() if val is not None else default
        # 尝试嵌套路径
        parts = field.split(".")
        current = source
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return str(current).strip() if current is not None else default


# -------------------- 三元组 -> 模型训练数据转换 --------------------
class TripleToDatasetConverter:
    """将三元组列表转换为 KGFaultDataset 所需的数据结构。

    与 kg_fault_diagnosis.py 中的 KGFaultDataset 完全兼容，
    生成的 Data 对象可直接用于 FaultGCN 模型训练。
    """

    def __init__(
            self,
            triples: List[Triple],
            fault_nodes: Optional[List[str]] = None,
            fault_type_relations: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        triples : List[Triple]
            知识图谱三元组列表。
        fault_nodes : Optional[List[str]]
            手动指定的故障节点列表。若为 None 则自动推断。
        fault_type_relations : Optional[List[str]]
            用于自动推断故障节点的关系类型列表，默认 ["类型为", "is_fault", "故障"]。
        """
        self.triples = triples
        self.fault_type_relations = fault_type_relations or ["类型为", "is_fault", "故障"]

        if fault_nodes is not None:
            self.fault_nodes = list(fault_nodes)
        else:
            self.fault_nodes = self._infer_fault_nodes()

        self.labels = self._build_labels()
        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.edge_index = self._build_edge_index()
        self.x = torch.eye(len(self.node_to_idx), dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(len(self.idx_to_node))]
        self.y = torch.tensor(
            [self.labels[node] for node in ordered_nodes], dtype=torch.long
        )

    def _infer_fault_nodes(self) -> List[str]:
        """通过关系类型自动推断故障节点。

        查找形如 (X --类型为--> 故障) 的三元组，将 X 标记为故障节点。
        """
        faults = []
        for t in self.triples:
            if t.relation in self.fault_type_relations:
                faults.append(t.head)
        if not faults:
            # 回退：收集所有关系为 "原因在于" 的尾实体
            for t in self.triples:
                if "原因" in t.relation:
                    if t.tail not in faults:
                        faults.append(t.tail)
        logger.info("自动推断故障节点: %s", faults)
        return sorted(set(faults))

    def _build_labels(self) -> Dict[str, int]:
        """构建节点标签：故障节点标记为 1，其余为 0。"""
        fault_set = set(self.fault_nodes)
        all_nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: 1 if node in fault_set else 0 for node in all_nodes}

    def _build_vocab(self) -> Dict[str, int]:
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: idx for idx, node in enumerate(nodes)}

    def _build_edge_index(self) -> torch.Tensor:
        edges = []
        for triple in self.triples:
            h = self.node_to_idx[triple.head]
            t = self.node_to_idx[triple.tail]
            edges.append([h, t])
            edges.append([t, h])  # 无向图
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def to_data(self) -> Data:
        """生成 PyG Data 对象。"""
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def statistics(self) -> dict:
        """返回数据集统计信息。"""
        fault_count = sum(v for v in self.labels.values())
        total = len(self.labels)
        return {
            "总节点数": total,
            "故障节点数": fault_count,
            "正常节点数": total - fault_count,
            "三元组数": len(self.triples),
            "边数（无向）": self.edge_index.shape[1],
            "关系类型数": len({t.relation for t in self.triples}),
            "故障节点": self.fault_nodes,
        }


# ============================================================
# 4. ID-名称解析器 —— 将 ES 中的原始 ID 映射为人类可读名称
# ============================================================
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
        self.entity_name_to_id: Dict[str, str] = {}  # name → entityId (反向映射，供采样器查询 ES)
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
            # 如果用了 graph_id 过滤但结果为空，尝试 match_all 并警告
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
        # 收集前几个无法解析的 ID 样本，用于诊断日志
        _unresolved_samples: List[Tuple[str, str, str]] = []

        for t in triplets:
            head_id = t["head"]
            tail_id = t["tail"]
            relation_id = t["relation"]

            # 二次查询：将 ID 映射为名称
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


# ============================================================
# 5. ES 流式三元组提取器（search_after 替代 scroll，解决上亿数据深分页问题）
# ============================================================
class ESTripletStreamer:
    """使用 search_after 替代 scroll/scan，避免深分页性能灾难。

    支持：
    - 全量流式遍历（离线构建全局 ID 映射）
    - 种子实体过滤子图（Mini-batch 训练）
    - 断点续传
    - 异步预取（生产者-消费者模式）

    search_after 相比 scroll 的优势：
    - 无状态分页，不占用 ES 服务端 scroll 上下文
    - 深分页性能稳定，不会随 offset 增大而退化
    - 天然支持断点续传
    """

    # 默认 ES 查询超时（秒），request_timeout 必须为 int/float
    DEFAULT_TIMEOUT = 120
    # 预取队列最大容量，防止内存溢出
    MAX_QUEUE_SIZE = 4

    def __init__(
            self,
            es_hosts: List[str],
            index_name: Union[str, List[str]],
            batch_size: int = 5000,
            prefetch: bool = True,
            checkpoint_dir: Optional[str] = None,
            resolver: Optional["IDNameResolver"] = None,
    ):
        """
        Parameters
        ----------
        es_hosts : List[str]
            ES 节点地址列表，如 ["http://host1:9200", "http://host2:9200"]。
        index_name : str or List[str]
            ES 索引名称，支持单个索引或索引列表（逗号分隔多索引查询）。
        batch_size : int
            每批返回的文档数。
        prefetch : bool
            是否启用异步预取（后台线程提前拉取下一批）。
        checkpoint_dir : Optional[str]
            断点续传目录，用于保存/恢复 progress。
        resolver : Optional[IDNameResolver]
            ID→名称解析器。若提供，stream_triplets_raw 将在产出前自动将
            原始 entityId/relationTypeId 解析为可读名称。
        """
        self.es = Elasticsearch(es_hosts)
        # 支持多索引：列表转逗号分隔字符串
        if isinstance(index_name, list):
            self.index = ",".join(index_name)
            self._index_label = "_".join(sorted(index_name))
        else:
            self.index = index_name
            self._index_label = index_name
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.resolver = resolver
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 断点续传 ----------
    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self._index_label}_progress.json"

    def _save_progress(self, search_after: Optional[List]) -> None:
        if self.checkpoint_dir is None:
            return
        with open(self._checkpoint_path(), "w") as f:
            json.dump({"search_after": search_after}, f)

    def _load_progress(self) -> Optional[List]:
        if self.checkpoint_dir is None:
            return None
        cp = self._checkpoint_path()
        if not cp.exists():
            return None
        with open(cp) as f:
            data = json.load(f)
            return data.get("search_after")

    def _clear_progress(self) -> None:
        if self.checkpoint_dir is None:
            return
        cp = self._checkpoint_path()
        if cp.exists():
            cp.unlink()

    # ---------- 核心：search_after 流式遍历 ----------
    def _build_query(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            tail_field: str = "tail_id",
            extra_filters: Optional[dict] = None,
    ) -> dict:
        """构建带排序的 search_after 查询体。"""
        if seed_entities:
            query = {"terms": {head_field: seed_entities}}
        elif extra_filters:
            query = extra_filters
        else:
            query = {"match_all": {}}

        return {
            "query": query,
            "sort": [
                {head_field: "asc"},
                {tail_field: "asc"},
                {"_id": "asc"},  # tie-breaker: 保证全局唯一排序
            ],
            "size": self.batch_size,
        }

    def stream_triplets_raw(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            resume: bool = True,
            extra_filters: Optional[dict] = None,
    ) -> Iterator[List[dict]]:
        """流式生成原始三元组批次（dict 格式）。

        若配置了 resolver，每批会在产出前自动将原始 ID 解析为可读名称；
        无法解析的三元组会被跳过，对齐原 fetch_triples 的处理逻辑。

        Yields
        ------
        List[dict]
            每批三元组列表，每个三元组为 {"head": ..., "relation": ..., "tail": ...}。
            字段值可能是原始 ID（无 resolver）或解析后的名称（有 resolver）。
        """
        query_body = self._build_query(
            seed_entities=seed_entities,
            head_field=head_field,
            tail_field=tail_field,
            extra_filters=extra_filters,
        )

        search_after = self._load_progress() if resume else None
        if search_after:
            logger.info(
                "断点续传: 从 search_after=%s 继续遍历索引 %s",
                search_after, self.index,
            )

        batch_count = 0
        total_docs = 0
        total_skipped = 0
        _resolver = self.resolver  # 局部引用，避免属性查找
        _resolver_warned = False  # 仅警告一次

        # 若配置了 resolver 但未就绪，在开始遍历时给出明确警告
        if _resolver and not _resolver.is_ready:
            logger.error(
                "❌ ID→名称解析器已配置但未就绪！"
                "实体名称映射=%d, 关系类型名称映射=%d。"
                "词汇表将存储原始 ID 而非名称！"
                "请检查 --graph-id / --ontology-id / --entity-name-field 参数。",
                len(_resolver.entity_map), len(_resolver.relation_type_map),
            )

        while True:
            if search_after:
                query_body["search_after"] = search_after

            try:
                resp = self.es.search(
                    index=self.index,
                    body=query_body,
                    request_timeout=self.DEFAULT_TIMEOUT,
                )
            except Exception as e:
                logger.error("ES search 失败 (batch=%d, search_after=%s): %s",
                             batch_count, search_after, e)
                raise

            hits = resp["hits"]["hits"]
            if not hits:
                break

            batch_count += 1
            total_docs += len(hits)

            # 提取三元组（原始 ID）
            triplets = []
            for hit in hits:
                src = hit["_source"]
                h = src.get(head_field)
                r = src.get(relation_field)
                t = src.get(tail_field)
                if h is not None and r is not None and t is not None:
                    triplets.append({
                        "head": str(h),
                        "relation": str(r),
                        "tail": str(t),
                    })

            # 若配置了 ID→名称解析器，将原始 ID 解析为可读名称
            if _resolver and _resolver.is_ready:
                triplets, _, skipped = _resolver.resolve_triplets(triplets)
                total_skipped += skipped

            if triplets:
                yield triplets

            # 更新游标
            search_after = hits[-1]["sort"]
            self._save_progress(search_after)

            if len(hits) < self.batch_size:
                break  # 最后一批

        self._clear_progress()
        logger.info(
            "search_after 遍历完成: 共 %d 批, %d 条文档, 已解析名称 (跳过 %d 条)",
            batch_count, total_docs, total_skipped,
        )

    def stream_triplets(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            resume: bool = True,
            extra_filters: Optional[dict] = None,
    ) -> Iterator[List[dict]]:
        """带异步预取的流式三元组迭代器（生产者-消费者模式）。

        生产者线程：从 ES search_after 拉取数据放入队列
        消费者（主线程）：从队列取出数据进行训练

        这使得 ES I/O 延迟被 GPU 计算完全掩盖，实现真正的"边查边训"。
        """
        if not self.prefetch:
            yield from self.stream_triplets_raw(
                seed_entities=seed_entities,
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
                resume=resume,
                extra_filters=extra_filters,
            )
            return

        # --- 异步预取模式 ---
        data_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        stop_sentinel = object()
        exception_holder: List[Exception] = []

        def producer() -> None:
            """生产者：在后台线程中从 ES 拉取数据。"""
            try:
                for batch in self.stream_triplets_raw(
                        seed_entities=seed_entities,
                        head_field=head_field,
                        relation_field=relation_field,
                        tail_field=tail_field,
                        resume=resume,
                        extra_filters=extra_filters,
                ):
                    data_queue.put(batch)  # 阻塞直到队列有空位
                data_queue.put(stop_sentinel)
            except Exception as e:
                exception_holder.append(e)
                data_queue.put(stop_sentinel)

        thread = threading.Thread(target=producer, daemon=True, name="es-producer")
        thread.start()

        while True:
            batch = data_queue.get()
            if batch is stop_sentinel:
                break
            yield batch

        thread.join(timeout=10)
        if exception_holder:
            raise exception_holder[0]


# ============================================================
# 6. 全局词汇表构建器（entity2idx / relation2idx）
# ============================================================
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
            streamer: ESTripletStreamer,
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
        - 以常见 ID 前缀开头的长串（如 "9800441..."、"src_"、"entity_"）
        - 不含任何中文字符且长度 > 20 的字母数字串
        """
        if not s:
            return False
        # 包含中文 → 肯定是名称
        if any('\u4e00' <= c <= '\u9fff' for c in s):
            return False
        # 全部是数字（大整数 ID）
        if s.isdigit() and len(s) >= 10:
            return True
        # 长字母数字组合（如 UUID、MongoDB ObjectId）
        if len(s) >= 20 and all(c.isalnum() or c in '-_' for c in s):
            return True
        return False

    def _warn_if_names_look_like_ids(self) -> None:
        """抽样检测词汇表中的键是否看起来像 ID，若是则发出明确警告。"""
        entity_keys = list(self.entity2idx.keys())
        relation_keys = list(self.relation2idx.keys())

        # 抽样前 20 个
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


# ============================================================
# 7. 异步子图采样器 —— 核心："边查边训"引擎
# ============================================================
class AsyncSubgraphSampler:
    """基于 ES + search_after 的异步子图采样器。

    结合 PyG NeighborLoader 的思路，但直接从 ES 动态构建子图，
    无需将全图加载到内存，适合上亿级知识图谱。

    工作流程：
    1. 给定一批种子节点，异步查询 ES 获取它们的 k-hop 邻居边
    2. 构建局部 HeteroData 子图
    3. 通过预取队列实现 ES I/O 与 GPU 计算的流水线重叠
    """

    def __init__(
            self,
            streamer: ESTripletStreamer,
            vocab: KGVocabulary,
            num_hops: int = 2,
            max_neighbors_per_hop: int = 100,
            prefetch_size: int = 2,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            embedding_dim: int = 64,
            resolver: Optional["IDNameResolver"] = None,
    ):
        """
        Parameters
        ----------
        streamer : ESTripletStreamer
            search_after 流式提取器。
        vocab : KGVocabulary
            全局实体/关系 ID 映射（存名称）。
        num_hops : int
            邻居采样的跳数。
        max_neighbors_per_hop : int
            每跳最大邻居数（防止度数爆炸）。
        prefetch_size : int
            预取子图数量，越大 IO 隐藏越好但内存占用越高。
        head_field / relation_field / tail_field : str
            ES 索引中的字段名。
        embedding_dim : int
            实体特征嵌入维度（实际特征应从 ES 另一索引或外部特征源获取）。
        resolver : Optional[IDNameResolver]
            ID↔名称解析器。用于将 seed 实体名称反解为 ES 查询所需的原始 ID。
        """
        self.streamer = streamer
        self.vocab = vocab
        self.num_hops = num_hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self.prefetch_size = prefetch_size
        self.head_field = head_field
        self.relation_field = relation_field
        self.tail_field = tail_field
        self.embedding_dim = embedding_dim
        self.resolver = resolver

        # 预取队列与线程
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        self._prefetch_error: Optional[Exception] = None

    def sample_subgraph(
            self,
            seed_entities: List[str],
            num_hops: Optional[int] = None,
    ) -> Optional[HeteroData]:
        """从 ES 动态构建以 seed_entities 为中心的局部子图。

        Parameters
        ----------
        seed_entities : List[str]
            种子节点实体 ID 列表。
        num_hops : Optional[int]
            采样跳数，默认使用初始化参数。

        Returns
        -------
        Optional[HeteroData]
            构建的异质子图，若无边则返回 None。
        """
        hops = num_hops or self.num_hops
        visited: Set[str] = set(seed_entities)
        frontier: Set[str] = set(seed_entities)
        edges: List[Tuple[int, int, int]] = []  # (head_idx, rel_idx, tail_idx)
        local_entities: Dict[str, int] = {}  # 子图内实体 → 局部索引

        # 若配置了名称→ID 反向映射，每次查询前将 frontier 回译为 ES 可识别的原始 ID
        _resolve_names = (
            self.resolver.resolve_entity_names_to_ids
            if self.resolver and self.resolver.is_ready
            else lambda names: names
        )

        for hop in range(hops):
            if not frontier:
                break

            # 查询当前 frontier 的所有出边
            frontier_list = list(frontier)
            next_frontier: Set[str] = set()

            # 分批查询（ES terms query 有长度限制）
            chunk_size = 1000
            for i in range(0, len(frontier_list), chunk_size):
                chunk_raw = frontier_list[i:i + chunk_size]
                # ↓ 关键：将实体名称回译为 ES 索引中的原始 ID
                chunk = _resolve_names(chunk_raw)

                for batch in self.streamer.stream_triplets(
                        seed_entities=chunk,
                        head_field=self.head_field,
                        relation_field=self.relation_field,
                        tail_field=self.tail_field,
                        resume=False,  # 子图查询不续传
                ):
                    for t in batch:
                        h, r, tail = t["head"], t["relation"], t["tail"]

                        # 分配局部索引
                        if h not in local_entities:
                            local_entities[h] = len(local_entities)
                        if tail not in local_entities:
                            local_entities[tail] = len(local_entities)

                        # 确保关系在全局词汇表中
                        global_rel_idx = self.vocab.add_relation(r)

                        edges.append((
                            local_entities[h],
                            global_rel_idx,
                            local_entities[tail],
                        ))

                        # 下一跳：尾实体（如果尚未访问且未超限）
                        if tail not in visited and len(next_frontier) < self.max_neighbors_per_hop:
                            next_frontier.add(tail)
                            visited.add(tail)

            frontier = next_frontier

        if not edges:
            return None

        # 构建 HeteroData
        data = HeteroData()
        data["entity"].num_nodes = len(local_entities)
        data["entity"].x = torch.randn(len(local_entities), self.embedding_dim)

        edge_array = np.array(edges, dtype=np.int64)
        data["entity", "to", "entity"].edge_index = torch.tensor(
            [edge_array[:, 0], edge_array[:, 2]], dtype=torch.long
        )
        data["entity", "to", "entity"].edge_type = torch.tensor(
            edge_array[:, 1], dtype=torch.long
        )

        # 附加元数据
        data.local_entity_ids = list(local_entities.keys())
        data.local2global = torch.tensor(
            [self.vocab.entity2idx.get(e, -1) for e in data.local_entity_ids],
            dtype=torch.long,
        )

        return data

    # ---------- 异步预取流水线 ----------
    def _prefetch_worker(
            self,
            seed_batches: Iterator[List[str]],
    ) -> None:
        """后台预取线程：持续从 ES 拉取子图并放入队列。"""
        try:
            for seeds in seed_batches:
                if self._stop_prefetch.is_set():
                    break
                subgraph = self.sample_subgraph(seeds)
                self._prefetch_queue.put((seeds, subgraph))
            self._prefetch_queue.put(None)  # 结束信号
        except Exception as e:
            self._prefetch_error = e
            self._prefetch_queue.put(None)

    def iter_subgraphs(
            self,
            seed_batches: List[List[str]],
    ) -> Iterator[Tuple[List[str], Optional[HeteroData]]]:
        """同步迭代器：逐个返回 (种子节点, 子图)。

        Parameters
        ----------
        seed_batches : List[List[str]]
            种子节点批次的列表。
        """
        for seeds in seed_batches:
            yield seeds, self.sample_subgraph(seeds)

    def iter_subgraphs_async(
            self,
            seed_batches: List[List[str]],
    ) -> Iterator[Tuple[List[str], Optional[HeteroData]]]:
        """异步预取迭代器：后台拉取 + 主线程消费。

        ES 查询与 GPU 训练流水线重叠，实现"边查边训"。
        """
        if self.prefetch_size <= 0:
            yield from self.iter_subgraphs(seed_batches)
            return

        self._stop_prefetch.clear()
        self._prefetch_error = None

        # 清空队列
        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break

        # 启动预取线程
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(iter(seed_batches),),
            daemon=True,
            name="es-subgraph-prefetch",
        )
        self._prefetch_thread.start()

        # 主线程消费
        received = 0
        while True:
            item = self._prefetch_queue.get()
            if item is None:
                break
            received += 1
            yield item

        self._prefetch_thread.join(timeout=30)
        if self._prefetch_error:
            raise self._prefetch_error

        logger.info("异步子图采样完成: 预取了 %d/%d 个子图", received, len(seed_batches))

    def shutdown(self) -> None:
        """关闭预取线程。"""
        self._stop_prefetch.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=10)


# ============================================================
# 8. PyG NeighborLoader 适配器（全图在内存时使用）
# ============================================================
class KGNeighborLoaderAdapter:
    """将 ES 流式构建的全图适配为 PyG NeighborLoader。

    适用场景：知识图谱足够小（边数 < 数千万），可以全量加载到内存。
    此时利用 NeighborLoader 的高效采样，配合 CUDA 加速训练。

    使用流程：
    1. 用 ESTripletStreamer 全量遍历构建全局图
    2. 用本适配器创建 NeighborLoader
    3. 标准 PyG 训练循环
    """

    def __init__(
            self,
            vocab: KGVocabulary,
            edge_index: torch.Tensor,
            edge_type: Optional[torch.Tensor] = None,
            node_features: Optional[torch.Tensor] = None,
            embedding_dim: int = 64,
    ):
        """
        Parameters
        ----------
        vocab : KGVocabulary
            全局词汇表。
        edge_index : torch.Tensor
            边索引 [2, num_edges]。
        edge_type : Optional[torch.Tensor]
            边类型索引 [num_edges]。
        node_features : Optional[torch.Tensor]
            节点特征，若为 None 则使用随机初始化的可学习嵌入。
        embedding_dim : int
            当 node_features 为 None 时的嵌入维度。
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self.data = HeteroData()
        self.data["entity"].num_nodes = vocab.num_entities

        if node_features is not None:
            self.data["entity"].x = node_features
        else:
            self.data["entity"].x = torch.randn(vocab.num_entities, embedding_dim)

        self.data["entity", "to", "entity"].edge_index = edge_index
        if edge_type is not None:
            self.data["entity", "to", "entity"].edge_type = edge_type

    def create_loader(
            self,
            input_nodes: torch.Tensor,
            num_neighbors: List[int] = None,
            batch_size: int = 128,
            shuffle: bool = True,
            **kwargs,
    ) -> NeighborLoader:
        """创建 PyG NeighborLoader。

        Parameters
        ----------
        input_nodes : torch.Tensor
            训练节点索引。
        num_neighbors : List[int]
            每跳采样邻居数，如 [25, 10] 表示第一跳采样25个，第二跳10个。
        batch_size : int
            每批节点数。
        shuffle : bool
            是否打乱。
        """
        if num_neighbors is None:
            num_neighbors = [25, 10]

        return NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=("entity", input_nodes),
            shuffle=shuffle,
            **kwargs,
        )

    @classmethod
    def from_streamer(
            cls,
            streamer: ESTripletStreamer,
            vocab: Optional[KGVocabulary] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            embedding_dim: int = 64,
            node_features: Optional[torch.Tensor] = None,
    ) -> "KGNeighborLoaderAdapter":
        """从 ESTripletStreamer 一键构建全图 + NeighborLoader 适配器。

        内部会做一次全量遍历来构建 edge_index。
        """
        if vocab is None:
            vocab = KGVocabulary()

        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        total = 0

        for batch in streamer.stream_triplets(
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
        ):
            for t in batch:
                h_idx = vocab.add_entity(t["head"])
                t_idx = vocab.add_entity(t["tail"])
                r_idx = vocab.add_relation(t["relation"])
                edges.append((h_idx, t_idx))
                edge_types.append(r_idx)
                total += 1

            if total % 500000 == 0:
                logger.info(
                    "全图构建中... 实体: %d, 边: %d",
                    vocab.num_entities, total,
                )

        if not edges:
            raise ValueError("未从 ES 读取到任何三元组")

        edge_array = np.array(edges, dtype=np.int64)
        edge_index = torch.tensor(edge_array.T, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        logger.info(
            "全图构建完成: 实体=%d, 关系=%d, 边=%d",
            vocab.num_entities, vocab.num_relations, total,
        )

        return cls(
            vocab=vocab,
            edge_index=edge_index,
            edge_type=edge_type,
            node_features=node_features,
            embedding_dim=embedding_dim,
        )


# ============================================================
# 9. "边查边训" 训练流水线
# ============================================================
class StreamingTrainingPipeline:
    """将异步 ES 子图采样与 PyG 训练循环集成的流水线。

    两种模式：
    1. 全图模式（图小）：先全量构建 graph → NeighborLoader → 训练
    2. 流式模式（图大）：AsyncSubgraphSampler 动态采样 → 训练

    模式 2 真正实现"边查边训"：ES 查询与 GPU 前向/反向传播流水线重叠。
    """

    def __init__(
            self,
            streamer: ESTripletStreamer,
            vocab: Optional[KGVocabulary] = None,
            mode: str = "streaming",
            **sampler_kwargs,
    ):
        """
        Parameters
        ----------
        streamer : ESTripletStreamer
            search_after 流式提取器。
        vocab : Optional[KGVocabulary]
            词汇表，None 则自动构建。
        mode : str
            "full" 或 "streaming"。
        **sampler_kwargs
            传递给 AsyncSubgraphSampler 的参数。
        """
        self.streamer = streamer
        self.mode = mode
        self.vocab = vocab or KGVocabulary()
        self.sampler_kwargs = sampler_kwargs

        self._sampler: Optional[AsyncSubgraphSampler] = None
        self._loader_adapter: Optional[KGNeighborLoaderAdapter] = None

    def build_vocab(
            self,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
    ) -> KGVocabulary:
        """第一阶段：构建全局词汇表（只需一次）。"""
        self.vocab.build_from_streamer(
            streamer=self.streamer,
            head_field=head_field,
            relation_field=relation_field,
            tail_field=tail_field,
        )
        return self.vocab

    def prepare(
            self,
            seed_batches: Optional[List[List[str]]] = None,
            input_nodes: Optional[torch.Tensor] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
    ) -> Iterator:
        """准备训练数据迭代器。

        根据 mode 返回不同迭代器：
        - "full": NeighborLoader 迭代器
        - "streaming": AsyncSubgraphSampler 异步迭代器

        Returns
        -------
        Iterator
            每次迭代返回一个 mini-batch 用于训练。
        """
        if self.mode == "full":
            return self._prepare_full_mode(
                input_nodes=input_nodes,
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
            )
        else:
            return self._prepare_streaming_mode(
                seed_batches=seed_batches,
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
            )

    def _prepare_full_mode(
            self,
            input_nodes: Optional[torch.Tensor],
            head_field: str,
            relation_field: str,
            tail_field: str,
    ) -> NeighborLoader:
        """全图模式：构建完整图 + NeighborLoader。"""
        logger.info("全图模式: 开始构建全局图...")
        self._loader_adapter = KGNeighborLoaderAdapter.from_streamer(
            streamer=self.streamer,
            vocab=self.vocab,
            head_field=head_field,
            relation_field=relation_field,
            tail_field=tail_field,
        )

        if input_nodes is None:
            input_nodes = torch.arange(self.vocab.num_entities, dtype=torch.long)

        loader = self._loader_adapter.create_loader(
            input_nodes=input_nodes,
            num_neighbors=[25, 10],
            batch_size=128,
        )
        logger.info("全图模式: NeighborLoader 就绪")
        return loader

    def _prepare_streaming_mode(
            self,
            seed_batches: Optional[List[List[str]]],
            head_field: str,
            relation_field: str,
            tail_field: str,
    ) -> AsyncSubgraphSampler:
        """流式模式：创建异步子图采样器。"""
        if seed_batches is None:
            raise ValueError("流式模式需要提供 seed_batches (种子节点批次列表)")

        logger.info("流式模式: 初始化异步子图采样器...")
        self._sampler = AsyncSubgraphSampler(
            streamer=self.streamer,
            vocab=self.vocab,
            head_field=head_field,
            relation_field=relation_field,
            tail_field=tail_field,
            **self.sampler_kwargs,
        )
        logger.info("流式模式: 异步子图采样器就绪 (num_hops=%d)", self._sampler.num_hops)
        return self._sampler

    def shutdown(self) -> None:
        """清理资源。"""
        if self._sampler:
            self._sampler.shutdown()


# ============================================================
# 10. 故障标签构建器 —— 从 ES 数据中识别故障节点
# ============================================================
class FaultLabelBuilder:
    """从已解析的知识图谱中自动识别故障节点并构建训练标签。

    识别策略：
    1. 基于关系模式匹配：查找 (实体 --关系类型--> 尾实体) 中
       relation ∈ fault_relations 且 tail ∈ fault_tails 的实体，
       将其标记为故障节点。
    2. 默认匹配："类型为" → "故障" 模式，覆盖常见工业故障分类。
    """

    # 默认故障类型指示关系（中文 KG 常见模式）
    DEFAULT_FAULT_RELATIONS = [
        "类型为", "type", "rdf:type", "类别为", "分类为",
        "is_fault", "故障类型", "fault_type",
    ]
    # 默认故障类目名
    DEFAULT_FAULT_TAILS = ["故障", "fault", "Failure", "异常", "失效"]

    def __init__(
            self,
            vocab: KGVocabulary,
            fault_relations: Optional[List[str]] = None,
            fault_tails: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        vocab : KGVocabulary
            全局词汇表，已包含所有实体和关系类型。
        fault_relations : Optional[List[str]]
            指示故障分类的关系名称列表，默认使用 DEFAULT_FAULT_RELATIONS。
        fault_tails : Optional[List[str]]
            故障类别的尾实体名称列表，默认使用 DEFAULT_FAULT_TAILS。
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
        将 head 实体标记为故障节点。遍历结束后为所有实体生成 0/1 标签张量。

        Parameters
        ----------
        streamer : ESTripletStreamer
            已配置解析器的流式提取器（需启用 ID→名称解析）。
        head_field / relation_field / tail_field : str
            ES 索引中的字段名。
        extra_filters : Optional[dict]
            额外的 ES 查询过滤条件。

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

        # 构建标签张量
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


# ============================================================
# 11. 训练 + 推理集成管线
# ============================================================
class KGTrainInferPipeline:
    """端到端：ES 数据读取 → 训练 → 评估 → 故障推理。

    封装完整的 GCN 训练和 Top-K 故障诊断流程，可与 ES 全图/流式
    两种数据模式配合使用。

    使用示例::

        pipeline = KGTrainInferPipeline()
        pipeline.train(
            loader=neighbor_loader,
            y=labels,
            epochs=200,
        )
        results = pipeline.infer_topk(
            model=pipeline.model,
            data=full_graph_data,
            symptoms=["振动过高", "温度过高"],
            node_to_idx=vocab.entity2idx,
            fault_nodes=["轴承磨损", "定子故障"],
        )
    """

    def __init__(
            self,
            in_dim: int = 64,
            hidden_dim: int = 32,
            num_classes: int = 2,
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        in_dim : int
            输入特征维度（应与节点嵌入维度一致）。
        hidden_dim : int
            隐藏层维度。
        num_classes : int
            分类类别数（默认 2: 正常/故障）。
        lr : float
            学习率。
        weight_decay : float
            L2 正则化权重衰减。
        device : Optional[str]
            训练设备，None 则自动选择 cuda/cpu。
        """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional["FaultGCN"] = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _init_model(self) -> "FaultGCN":
        from kg_fault_diagnosis import FaultGCN
        model = FaultGCN(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        if self.num_classes != 2:
            model.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        return model

    # ---- 全图模式训练（Data / HeteroData） ----
    def train_full_graph(
            self,
            data: Union[Data, HeteroData],
            y: torch.Tensor,
            train_mask: torch.Tensor,
            val_mask: torch.Tensor,
            epochs: int = 200,
            log_interval: int = 50,
            verbose: bool = True,
    ) -> Dict[str, list]:
        """在全图上训练 FaultGCN（使用 train/val mask）。

        Parameters
        ----------
        data : Data or HeteroData
            包含 x, edge_index 的图数据。
        y : torch.Tensor
            所有节点的标签 [num_nodes]。
        train_mask : torch.Tensor
            训练集 mask [num_nodes]。
        val_mask : torch.Tensor
            验证集 mask [num_nodes]。
        epochs : int
            训练轮数。
        log_interval : int
            日志输出间隔。
        verbose : bool
            是否打印训练日志。

        Returns
        -------
        Dict[str, list]
            包含 epoch, train_loss, val_acc 的历史记录。
        """
        model = self._init_model()
        data = data.to(self.device)
        y = y.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        history = {"epoch": [], "train_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            self.optimizer.zero_grad()
            logits = model(data)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    pred = model(data).argmax(dim=1)
                    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_acc)
                if verbose:
                    logger.info(
                        "Epoch %03d | loss=%.4f | val_acc=%.4f",
                        epoch, train_loss, val_acc,
                    )

        return history

    # ---- NeighborLoader 模式训练 ----
    def train_with_loader(
            self,
            loader,
            epochs: int = 10,
            log_interval: int = 1,
            verbose: bool = True,
    ) -> Dict[str, list]:
        """使用 PyG NeighborLoader 进行 mini-batch 训练。

        Parameters
        ----------
        loader : NeighborLoader
            PyG NeighborLoader 迭代器。
        epochs : int
            训练轮数。
        log_interval : int
            日志输出间隔。
        verbose : bool
            是否打印训练日志。

        Returns
        -------
        Dict[str, list]
            训练历史记录。
        """
        model = self._init_model()
        history = {"epoch": [], "train_loss": []}

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                logits = model(batch)

                # 从 batch 中提取标签（需要预先设置）
                if hasattr(batch, "y") and batch.y is not None:
                    target = batch.y[: logits.shape[0]]
                else:
                    target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / max(batch_count, 1)
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            if verbose and (epoch % log_interval == 0 or epoch == 1 or epoch == epochs):
                logger.info("Epoch %03d | avg_loss=%.4f | batches=%d", epoch, avg_loss, batch_count)

        return history

    # ---- 评估 ----
    @staticmethod
    def evaluate(
            data: Union[Data, HeteroData],
            y: torch.Tensor,
            test_mask: torch.Tensor,
            model: Optional["FaultGCN"] = None,
    ) -> Dict[str, float]:
        """在测试集上评估模型。

        Returns
        -------
        Dict[str, float]
            {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
        """
        if model is None:
            raise ValueError("model 不能为 None，请先训练或加载模型")

        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        y = y.to(device)
        test_mask = test_mask.to(device)

        with torch.no_grad():
            pred = model(data).argmax(dim=1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        test_true = y[test_mask].cpu().numpy()
        test_pred = pred[test_mask].cpu().numpy()

        return {
            "accuracy": float(accuracy_score(test_true, test_pred)),
            "precision": float(precision_score(test_true, test_pred, zero_division=0)),
            "recall": float(recall_score(test_true, test_pred, zero_division=0)),
            "f1": float(f1_score(test_true, test_pred, zero_division=0)),
        }

    # ---- Top-K 故障诊断推理 ----
    @staticmethod
    def infer_topk(
            model: "FaultGCN",
            data: Union[Data, HeteroData],
            symptoms: List[str],
            node_to_idx: Dict[str, int],
            fault_nodes: List[str],
            top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """给定症状节点，基于嵌入余弦相似度推理最可能的故障根因。

        Parameters
        ----------
        model : FaultGCN
            已训练好的 GCN 模型。
        data : Data or HeteroData
            图数据。
        symptoms : List[str]
            症状节点名称列表。
        node_to_idx : Dict[str, int]
            节点名称到索引的映射。
        fault_nodes : List[str]
            候选故障节点名称列表。
        top_k : int
            返回的 Top-K 结果数。

        Returns
        -------
        List[Tuple[str, float]]
            按相似度降序排列的 (故障节点, 相似度) 列表。
        """
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            node_emb = model.encode(data)

            # 症状节点嵌入取平均作为查询向量
            symptom_indices = [
                node_to_idx[s] for s in symptoms if s in node_to_idx
            ]
            if not symptom_indices:
                raise ValueError(f"所有症状节点均不在图中: {symptoms}")

            query_emb = node_emb[symptom_indices].mean(dim=0, keepdim=True)

            # 候选故障节点嵌入
            candidate_indices = [
                node_to_idx[f] for f in fault_nodes if f in node_to_idx
            ]
            if not candidate_indices:
                raise ValueError(f"所有候选故障节点均不在图中: {fault_nodes}")

            candidate_emb = node_emb[candidate_indices]
            # 有效候选节点名称（对齐索引）
            valid_fault_names = [
                f for f in fault_nodes if f in node_to_idx
            ]

            # 余弦相似度
            query_norm = F.normalize(query_emb, p=2, dim=-1)
            cand_norm = F.normalize(candidate_emb, p=2, dim=-1)
            scores = (cand_norm * query_norm).sum(dim=-1)

            ranked = sorted(
                zip(valid_fault_names, scores.cpu().tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            return ranked[:top_k]

    # ---- 模型保存 / 加载 ----
    def save_model(self, filepath: str) -> None:
        """保存模型参数到磁盘。"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型，请先训练")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
        }, path)
        logger.info("模型已保存至 %s", filepath)

    def load_model(self, filepath: str) -> "FaultGCN":
        """从磁盘加载模型参数。"""
        from kg_fault_diagnosis import FaultGCN
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        model = FaultGCN(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        if checkpoint.get("num_classes", 2) != 2:
            model.classifier = nn.Linear(checkpoint["hidden_dim"], checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        self.in_dim = checkpoint["in_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_classes = checkpoint.get("num_classes", 2)
        self.model = model
        logger.info("模型已从 %s 加载", filepath)
        return model


# -------------------- 便捷函数 --------------------
def build_pipeline(
        entity_index: str = "knowledge_entity_index",
        relation_index: str = "knowledge_entity_relation_index",
        relation_type_index: str = "knowledge_entity_type_relation_index",
        batch_size: int = 5000,
        fault_nodes: Optional[List[str]] = None,
        **field_mapping,
) -> TripleToDatasetConverter:
    """一键构建：从 ES 读取 -> 构建三元组 -> 转换数据集。

    Returns
    -------
    TripleToDatasetConverter
        包含 node_to_idx, edge_index, x, y 等可直接用于 GCN 训练的属性。
    """
    reader = ESKnowledgeGraphReader()
    try:
        triples = reader.fetch_triples(
            entity_index=entity_index,
            relation_index=relation_index,
            relation_type_index=relation_type_index,
            batch_size=batch_size,
            **field_mapping,
        )
        if not triples:
            raise ValueError("未能从 ES 提取到任何三元组，请检查索引名与字段映射")

        converter = TripleToDatasetConverter(triples, fault_nodes=fault_nodes)
        logger.info("数据集构建完成:\n%s", converter.statistics())
        return converter

    finally:
        reader.close()


# -------------------- 示例主程序 --------------------
def main() -> None:
    """示例：支持 3 种模式的知识图谱读取与训练。

    模式 A: 传统 scroll/scan 模式（兼容旧代码）
    模式 B: search_after 全图模式 → NeighborLoader 训练（图较小，内存可容纳）
    模式 C: search_after 流式模式 → 异步子图采样训练（图巨大，"边查边训"）
    """
    import argparse

    parser = argparse.ArgumentParser(description="ES 知识图谱读取与训练")
    parser.add_argument(
        "--mode", choices=["legacy", "full", "streaming"], default="streaming",
        help="legacy: 传统scroll模式 | full: NeighborLoader全图模式 | streaming: 异步流式边查边训",
    )
    parser.add_argument(
        "--index", default=["knowledge_entity_relation_index"], nargs="+",
        help="ES 索引名（支持多个）",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prefetch", type=int, default=2, help="异步预取队列大小")
    # ES 字段映射（对齐原代码的索引 schema）
    parser.add_argument("--head-field", default="srcEntityId")
    parser.add_argument("--relation-field", default="relationTypeId")
    parser.add_argument("--tail-field", default="dstEntityId")

    parser.add_argument("--graph-id", default="986946166448234496", help="按 graphId 过滤（如 980044155496734720）")
    parser.add_argument("--ontology-id", default="986946166448234496", help="关系类型 ontologyId")

    # ID→名称解析索引（对齐原 fetch_triples 三步逻辑）
    parser.add_argument("--entity-index", default="knowledge_entity_index", help="实体索引名称")
    parser.add_argument("--entity-id-field", default="entityId", help="实体索引中的 ID 字段名")
    parser.add_argument("--entity-name-field", default="name",
                        help="实体索引中的名称字段名（如 name / entityName / displayName）")
    parser.add_argument("--relation-type-index", default="knowledge_entity_type_relation_index",
                        help="关系类型索引名称")
    parser.add_argument("--relation-type-id-field", default="relationTypeId", help="关系类型索引中的 ID 字段名")
    parser.add_argument("--relation-type-name-field", default="name",
                        help="关系类型索引中的名称字段名（如 name / typeName / label）")
    parser.add_argument("--no-resolve", action="store_true", help="禁用 ID 到名称的解析（直接使用原始 ID）")
    parser.add_argument("--resolve-debug", action="store_true", help="启用解析器调试模式：采样索引文档并输出字段名")
    # 故障标签 & 推理参数
    parser.add_argument("--fault-relations", nargs="+",
                        default=["类型为", "type", "rdf:type", "类别为", "is_fault", "故障类型"],
                        help="标识故障分类的关系名称列表")
    parser.add_argument("--fault-tails", nargs="+",
                        default=["故障", "fault", "Failure", "异常", "失效"],
                        help="故障类目尾实体名称列表")
    parser.add_argument("--gcn-epochs", type=int, default=200, help="GCN 训练轮数")
    parser.add_argument("--no-infer", action="store_true", help="跳过推理阶段")
    parser.add_argument("--symptoms", nargs="+", default=None,
                        help="推理时输入的症状节点名称（多个用空格分隔）")
    parser.add_argument("--model-save", default="./models/kg_fault_model.pt", help="模型保存路径")
    args = parser.parse_args()

    config = ESConfig()

    # ========== 模式 A: 传统模式（兼容旧代码 + 推理） ==========
    if args.mode == "legacy":
        reader = ESKnowledgeGraphReader(config)
        try:
            logger.info("运行模式: legacy (scroll/scan)")
            logger.info("发现索引: %s", reader.list_indices())

            triples = reader.fetch_triples(
                entity_index="knowledge_entity_index",
                relation_index="knowledge_entity_relation_index",
                relation_type_index="knowledge_entity_type_relation_index",
                batch_size=args.batch_size,
            )

            if not triples:
                logger.error("未提取到三元组")
                return

            logger.info("共提取 %d 个三元组，预览前 5 条:", len(triples))
            for t in triples[:5]:
                logger.info("  (%s) --[%s]--> (%s)", t.head, t.relation, t.tail)

            converter = TripleToDatasetConverter(triples)
            data = converter.to_data()

            from kg_fault_diagnosis import FaultGCN, split_masks, train, evaluate, topk_fault_diagnosis
            data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)
            model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
            train(model, data)
            evaluate(model, data)

            # 推理：Top-K 故障诊断
            if not args.no_infer and converter.fault_nodes:
                symptoms = args.symptoms
                if symptoms is None:
                    # 自动选取示例症状
                    normal_nodes = [
                        n for n, lbl in converter.labels.items() if lbl == 0
                    ][:3]
                    symptoms = normal_nodes if normal_nodes else ["泵_01"]
                    logger.info("未指定 --symptoms，自动选取: %s", symptoms)

                logger.info("\n执行 Top-K 故障诊断推理 ...")
                try:
                    results = topk_fault_diagnosis(
                        model, data, converter, symptoms, top_k=min(5, len(converter.fault_nodes)),
                    )
                    logger.info("=" * 50)
                    logger.info("  Top-K 故障诊断结果")
                    logger.info("  输入症状: %s", ", ".join(symptoms))
                    for rank, (fault, score) in enumerate(results, start=1):
                        logger.info("  %d. %-30s similarity=%.4f", rank, fault, score)
                    logger.info("=" * 50)
                except ValueError as e:
                    logger.warning("推理失败: %s", e)
        finally:
            reader.close()
        return

    # ========== 模式 B & C: search_after 流式架构 ==========
    # 多索引支持：传入列表即可跨索引查询
    index_names = args.index if len(args.index) > 1 else args.index[0]
    logger.info("目标索引: %s", index_names)

    # ---------- 第一阶段：构建 ID→名称解析器 ----------
    resolver = None
    if not args.no_resolve:
        logger.info("Phase 0: 构建 ID→名称解析器...")
        if args.resolve_debug:
            logger.info("[DEBUG] 解析器调试模式已启用")
        es_client = Elasticsearch([f"{config.scheme}://{config.host}:{config.port}"])
        resolver = IDNameResolver(es_client)
        resolver.build_entity_map(
            entity_index=args.entity_index,
            graph_id=args.graph_id,
            id_field=args.entity_id_field,
            name_field=args.entity_name_field,
            batch_size=args.batch_size,
            extra_query={"match": {"graphId": args.graph_id}} if args.graph_id else None,
            debug=args.resolve_debug,
        )
        resolver.build_relation_type_map(
            relation_type_index=args.relation_type_index,
            ontology_id=args.ontology_id,
            id_field=args.relation_type_id_field,
            name_field=args.relation_type_name_field,
            batch_size=args.batch_size,
            debug=args.resolve_debug,
        )
        logger.info(
            "ID→名称解析器就绪: 实体=%d 种, 关系类型=%d 种",
            len(resolver.entity_map), len(resolver.relation_type_map),
        )
        if not resolver.is_ready:
            logger.error(
                "❌ 解析器未就绪 (entity_map=%d, relation_type_map=%d)！"
                "词汇表中的实体/关系名称将保持为原始 ID。"
                "请运行 --resolve-debug 查看索引详情，"
                "或检查 --graph-id / --ontology-id / --entity-name-field 等参数。",
                len(resolver.entity_map), len(resolver.relation_type_map),
            )

    streamer = ESTripletStreamer(
        es_hosts=[f"{config.scheme}://{config.host}:{config.port}"],
        index_name=index_names,
        batch_size=args.batch_size,
        prefetch=True,
        checkpoint_dir="./es_checkpoints",
        resolver=resolver,
    )

    # ---------- 第二阶段：构建全局词汇表 ----------
    logger.info("Phase 1: 构建全局实体/关系词汇表 (已解析为名称)...")
    if resolver is None:
        logger.warning(
            "⚠️  未启用 ID→名称解析！词汇表中将存储原始 ID。"
            "请确认是否需要名称解析（检查 --no-resolve 参数）"
        )
    else:
        logger.info(
            "解析器配置: 实体索引=%s (id=%s, name=%s), 关系类型索引=%s (id=%s, name=%s)",
            args.entity_index, args.entity_id_field, args.entity_name_field,
            args.relation_type_index, args.relation_type_id_field, args.relation_type_name_field,
        )

    vocab = KGVocabulary(checkpoint_dir="./vocab_checkpoints")
    # 注入索引名称提示（供警告信息使用）
    vocab._entity_index_hint = args.entity_index
    vocab._relation_type_index_hint = args.relation_type_index

    # 构建 graphId 过滤条件（对齐原代码默认行为）
    extra_filters = None
    if args.graph_id:
        extra_filters = {"match": {"graphId": args.graph_id}}
        logger.info("启用 graphId 过滤: %s", args.graph_id)

    vocab.build_from_streamer(
        streamer,
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        extra_filters=extra_filters,
    )
    vocab.save("./vocab_checkpoints/vocab.json")

    # 预览解析后的名称（前 5 个实体 + 前 3 个关系类型）
    sample_entities = list(vocab.entity2idx.keys())[:5]
    sample_relations = list(vocab.relation2idx.keys())[:3]
    logger.info("解析后实体示例: %s", sample_entities)
    logger.info("解析后关系类型示例: %s", sample_relations)

    if args.mode == "full":
        # ========== 模式 B: 全图模式 → 真实标签训练 + 推理 ==========
        logger.info("模式 B: 全图模式 → 构建 edge_index + 全图 GCN 训练 + 推理")

        # Phase 2: 构建全图 edge_index（复用 vocab 中的实体/关系索引）
        adapter = KGNeighborLoaderAdapter.from_streamer(
            streamer=streamer,
            vocab=vocab,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
        )
        data = adapter.data

        # Phase 3: 构建故障标签
        logger.info("Phase 3: 识别故障节点并构建标签 ...")
        label_builder = FaultLabelBuilder(
            vocab=vocab,
            fault_relations=args.fault_relations,
            fault_tails=args.fault_tails,
        )
        y, fault_nodes = label_builder.build_from_streamer(
            streamer=streamer,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
            extra_filters=extra_filters,
        )
        data.y = y

        if not fault_nodes:
            logger.error(
                "未识别到任何故障节点，无法训练。"
                "请通过 --fault-relations / --fault-tails 指定正确的故障分类关系。"
            )
            return

        # Phase 4: 划分 train/val/test
        from kg_fault_diagnosis import split_masks
        data.train_mask, data.val_mask, data.test_mask = split_masks(data["entity"].num_nodes)
        logger.info(
            "数据集划分: train=%d, val=%d, test=%d",
            data.train_mask.sum().item(),
            data.val_mask.sum().item(),
            data.test_mask.sum().item(),
        )

        # Phase 5: 训练
        logger.info("Phase 4: 全图 GCN 训练 (epochs=%d) ...", args.gcn_epochs)
        pipeline = KGTrainInferPipeline(in_dim=adapter.embedding_dim, hidden_dim=32)
        history = pipeline.train_full_graph(
            data=data,
            y=y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            epochs=args.gcn_epochs,
            verbose=True,
        )

        # Phase 6: 评估
        logger.info("Phase 5: 测试集评估 ...")
        metrics = pipeline.evaluate(
            data=data,
            y=y,
            test_mask=data.test_mask,
            model=pipeline.model,
        )
        logger.info(
            "测试集结果: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
        )

        # Phase 7: 推理 (Top-K 故障诊断)
        if not args.no_infer and fault_nodes:
            symptoms = args.symptoms
            if symptoms is None:
                # 自动选取非故障的节点作为示例症状
                all_entity_names = list(vocab.entity2idx.keys())
                non_fault_set = set(all_entity_names) - set(fault_nodes)
                symptoms = list(non_fault_set)[:3] if non_fault_set else all_entity_names[:2]
                logger.info("未指定 --symptoms，自动选取示例: %s", symptoms)

            logger.info("Phase 6: Top-K 故障诊断推理 (symptoms=%s) ...", symptoms)
            try:
                results = pipeline.infer_topk(
                    model=pipeline.model,
                    data=data,
                    symptoms=symptoms,
                    node_to_idx=vocab.entity2idx,
                    fault_nodes=fault_nodes,
                    top_k=min(5, len(fault_nodes)),
                )
                logger.info("=" * 50)
                logger.info("  Top-K 故障诊断结果")
                logger.info("  输入症状: %s", ", ".join(symptoms))
                for rank, (fault, score) in enumerate(results, start=1):
                    logger.info("  %d. %-30s similarity=%.4f", rank, fault, score)
                logger.info("=" * 50)
            except ValueError as e:
                logger.warning("推理失败: %s", e)

        # 保存模型
        if args.model_save:
            pipeline.save_model(args.model_save)

    else:
        # ========== 模式 C: 流式模式 → 先建全图再训练推理 ==========
        logger.info("模式 C: 流式模式 → 全量构建图 + GCN 训练 + 推理")

        # Phase 2: 流式构建全图 edge_index
        logger.info("Phase 2: 流式构建全图 (search_after 逐批加载)...")
        adapter = KGNeighborLoaderAdapter.from_streamer(
            streamer=streamer,
            vocab=vocab,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
        )
        data = adapter.data

        # Phase 3: 构建故障标签
        logger.info("Phase 3: 识别故障节点并构建标签 ...")
        label_builder = FaultLabelBuilder(
            vocab=vocab,
            fault_relations=args.fault_relations,
            fault_tails=args.fault_tails,
        )
        y, fault_nodes = label_builder.build_from_streamer(
            streamer=streamer,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
            extra_filters=extra_filters,
        )
        data.y = y

        if not fault_nodes:
            logger.error(
                "未识别到任何故障节点，无法训练。"
                "请通过 --fault-relations / --fault-tails 指定正确的故障分类关系。"
            )
            return

        # Phase 4: 划分 train/val/test
        from kg_fault_diagnosis import split_masks
        data.train_mask, data.val_mask, data.test_mask = split_masks(data["entity"].num_nodes)
        logger.info(
            "数据集划分: train=%d, val=%d, test=%d",
            data.train_mask.sum().item(),
            data.val_mask.sum().item(),
            data.test_mask.sum().item(),
        )

        # Phase 5: 训练
        logger.info("Phase 4: GCN 训练 (epochs=%d) ...", args.gcn_epochs)
        pipeline = KGTrainInferPipeline(in_dim=adapter.embedding_dim, hidden_dim=32)
        history = pipeline.train_full_graph(
            data=data,
            y=y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            epochs=args.gcn_epochs,
            verbose=True,
        )

        # Phase 6: 评估
        logger.info("Phase 5: 测试集评估 ...")
        metrics = pipeline.evaluate(
            data=data,
            y=y,
            test_mask=data.test_mask,
            model=pipeline.model,
        )
        logger.info(
            "测试集结果: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
        )

        # Phase 7: 推理
        if not args.no_infer and fault_nodes:
            symptoms = args.symptoms
            if symptoms is None:
                all_entity_names = list(vocab.entity2idx.keys())
                non_fault_set = set(all_entity_names) - set(fault_nodes)
                symptoms = list(non_fault_set)[:3] if non_fault_set else all_entity_names[:2]
                logger.info("未指定 --symptoms，自动选取示例: %s", symptoms)

            logger.info("Phase 6: Top-K 故障诊断推理 ...")
            try:
                results = pipeline.infer_topk(
                    model=pipeline.model,
                    data=data,
                    symptoms=symptoms,
                    node_to_idx=vocab.entity2idx,
                    fault_nodes=fault_nodes,
                    top_k=min(5, len(fault_nodes)),
                )
                logger.info("=" * 50)
                logger.info("  Top-K 故障诊断结果")
                logger.info("  输入症状: %s", ", ".join(symptoms))
                for rank, (fault, score) in enumerate(results, start=1):
                    logger.info("  %d. %-30s similarity=%.4f", rank, fault, score)
                logger.info("=" * 50)
            except ValueError as e:
                logger.warning("推理失败: %s", e)

        # 保存模型
        if args.model_save:
            pipeline.save_model(args.model_save)


if __name__ == "__main__":
    main()

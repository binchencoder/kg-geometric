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
# 4. ES 流式三元组提取器（search_after 替代 scroll，解决上亿数据深分页问题）
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
        """流式生成原始三元组批次（dict 格式，不做 ID 映射）。

        Yields
        ------
        List[dict]
            每批三元组列表，每个三元组为 {"head": ..., "relation": ..., "tail": ...}。
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

            # 提取三元组
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

            yield triplets

            # 更新游标
            search_after = hits[-1]["sort"]
            self._save_progress(search_after)

            if len(hits) < self.batch_size:
                break  # 最后一批

        self._clear_progress()
        logger.info("search_after 遍历完成: 共 %d 批, %d 条文档", batch_count, total_docs)

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
# 5. 全局词汇表构建器（entity2idx / relation2idx）
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
        return total

    def save(self, filepath: str) -> None:
        """持久化词汇表到磁盘。"""
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
# 6. 异步子图采样器 —— 核心："边查边训"引擎
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
    ):
        """
        Parameters
        ----------
        streamer : ESTripletStreamer
            search_after 流式提取器。
        vocab : KGVocabulary
            全局实体/关系 ID 映射。
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

        for hop in range(hops):
            if not frontier:
                break

            # 查询当前 frontier 的所有出边
            frontier_list = list(frontier)
            next_frontier: Set[str] = set()

            # 分批查询（ES terms query 有长度限制）
            chunk_size = 1000
            for i in range(0, len(frontier_list), chunk_size):
                chunk = frontier_list[i:i + chunk_size]

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
# 7. PyG NeighborLoader 适配器（全图在内存时使用）
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
# 8. "边查边训" 训练流水线
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
    parser.add_argument("--graph-id", default=None, help="按 graphId 过滤（如 980044155496734720）")
    args = parser.parse_args()

    config = ESConfig()

    # ========== 模式 A: 传统模式（兼容旧代码） ==========
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

            from kg_fault_diagnosis import FaultGCN, split_masks, train, evaluate
            data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)
            model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
            train(model, data)
            evaluate(model, data)
        finally:
            reader.close()
        return

    # ========== 模式 B & C: search_after 流式架构 ==========
    # 多索引支持：传入列表即可跨索引查询
    index_names = args.index if len(args.index) > 1 else args.index[0]
    logger.info("目标索引: %s", index_names)

    streamer = ESTripletStreamer(
        es_hosts=[f"{config.scheme}://{config.host}:{config.port}"],
        index_name=index_names,
        batch_size=args.batch_size,
        prefetch=True,
        checkpoint_dir="./es_checkpoints",
    )

    # ---------- 第一阶段：构建全局词汇表 ----------
    logger.info("Phase 1: 构建全局实体/关系词汇表...")
    vocab = KGVocabulary(checkpoint_dir="./vocab_checkpoints")

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

    if args.mode == "full":
        # ========== 模式 B: 全图 NeighborLoader ==========
        logger.info("模式 B: 全图模式 → 构建 edge_index + NeighborLoader")
        adapter = KGNeighborLoaderAdapter.from_streamer(
            streamer=streamer,
            vocab=vocab,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
        )

        # 创建 NeighborLoader
        all_nodes = torch.arange(vocab.num_entities, dtype=torch.long)
        loader = adapter.create_loader(
            input_nodes=all_nodes,
            num_neighbors=[25, 10],
            batch_size=128,
            shuffle=True,
        )

        # 标准 PyG 训练循环
        from kg_fault_diagnosis import FaultGCN
        from sklearn.metrics import accuracy_score

        model = FaultGCN(in_dim=64, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        logger.info("开始训练 (NeighborLoader 模式)...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.zeros(batch["entity"].num_nodes, dtype=torch.long)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("Epoch %d | loss=%.4f", epoch, total_loss)

    else:
        # ========== 模式 C: 流式"边查边训" ==========
        logger.info("模式 C: 流式模式 → 异步子图采样 + 边查边训")

        sampler = AsyncSubgraphSampler(
            streamer=streamer,
            vocab=vocab,
            num_hops=2,
            max_neighbors_per_hop=100,
            prefetch_size=args.prefetch,
            head_field=args.head_field,
            relation_field=args.relation_field,
            tail_field=args.tail_field,
        )

        # 模拟种子节点批次（实际场景中由业务 Sampler 产生）
        sample_entities = list(vocab.entity2idx.keys())[:200]
        seed_batches = [
            sample_entities[i:i + 32]
            for i in range(0, len(sample_entities), 32)
        ]

        from kg_fault_diagnosis import FaultGCN

        model = FaultGCN(in_dim=64, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        logger.info("开始训练 (流式 边查边训 模式)...")
        for epoch in range(args.epochs):
            model.train()
            batch_count = 0

            for seeds, subgraph in sampler.iter_subgraphs_async(seed_batches):
                if subgraph is None:
                    continue

                batch_count += 1
                optimizer.zero_grad()
                logits = model(subgraph)
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    torch.zeros(subgraph["entity"].num_nodes, dtype=torch.long),
                )
                loss.backward()
                optimizer.step()

            logger.info(
                "Epoch %d | batches=%d | entities=%d | relations=%d",
                epoch,
                batch_count,
                vocab.num_entities,
                vocab.num_relations,
            )

        sampler.shutdown()
        logger.info("流式训练完成")


if __name__ == "__main__":
    main()

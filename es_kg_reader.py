"""
Elasticsearch 知识图谱数据读取与三元组构建模块。

功能：
1. 连接 Elasticsearch 并读取实体与关系数据
2. 解析数据提取 (头实体, 关系, 尾实体) 标准三元组
3. 将三元组转换为模型训练所需的数据结构
4. 批量读取与处理机制以应对工业级海量数据
5. 异常处理确保数据完整性
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from torch_geometric.data import Data

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
    """示例：连接 ES，读取数据，构建训练数据集。"""
    # ---------- 1. 连接 ES ----------
    reader = ESKnowledgeGraphReader()

    try:
        # ---------- 2. 发现可用索引 ----------
        logger.info("发现索引: %s", reader.list_indices())

        # ---------- 3. 读取三元组 ----------
        # 场景 A：实体+关系+关系类型三索引模式（完整链路）
        # fetch_triples 内部自动执行两步查询：
        #   ① relation_field(id) → knowledge_entity_type_relation_index.relationTypeId → name
        #   ② srcEntityId/dstEntityId → knowledge_entity_index.entityId → name
        triples = reader.fetch_triples(
            entity_index="knowledge_entity_index",
            relation_index="knowledge_entity_relation_index",
            relation_type_index="knowledge_entity_type_relation_index",
            batch_size=5000,
        )

        # 场景 B（备选）：单索引+关系类型映射模式
        # triples = reader.fetch_triples_from_single_index(
        #     index="knowledge_entity_relation_index",
        #     relation_type_index="knowledge_entity_type_relation_index",
        # )

        if not triples:
            logger.error("未提取到三元组，请确认索引名称和字段映射是否正确。")
            logger.info("使用 list_indices() 查看可用索引，使用 get_index_mapping() 查看字段结构。")
            return

        # ---------- 4. 预览三元组 ----------
        logger.info("共提取 %d 个三元组，预览前 10 条:", len(triples))
        for t in triples[:10]:
            logger.info("  (%s) --[%s]--> (%s)", t.head, t.relation, t.tail)

        # ---------- 5. 转换为训练数据集 ----------
        converter = TripleToDatasetConverter(triples)
        data = converter.to_data()

        logger.info("模型训练数据准备完毕:")
        for k, v in converter.statistics().items():
            logger.info("  %s: %s", k, v)

        # 可以直接传递给 kg_fault_diagnosis.py 中的 FaultGCN:
        from kg_fault_diagnosis import FaultGCN, split_masks, train, evaluate
        data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)
        model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
        train(model, data)
        evaluate(model, data)

    finally:
        reader.close()


if __name__ == "__main__":
    main()

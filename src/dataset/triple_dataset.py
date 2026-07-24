"""知识图谱数据集 —— 车辆故障知识图谱。

优先从 Elasticsearch 加载真实三元组数据，失败时回退到内置 CVFFAD 示例数据。
包含故障现象、故障原因、维修措施、所需工具等多关系类型。

支持 R-GCN 所需的关系类型索引和图遍历查询。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from src.core.config import (
    logger,
    ESConfig,
    KnowledgeGraphSchema,
    RelationMappingConfig,
)
from src.core.types import Triple


class KGTripleDataset:
    """知识图谱三元组数据集。

    数据源优先级：ES 实时查询 → 内置 CVFFAD 示例数据。

    Parameters
    ----------
    es_entity_index : str
        ES 实体索引名称。
    es_relation_index : str
        ES 关系索引名称。
    es_relation_type_index : str
        ES 关系类型索引名称。
    es_batch_size : int
        ES scan 批次大小。
    es_head_field : str
        关系文档中头实体 ID 字段名。
    es_tail_field : str
        关系文档中尾实体 ID 字段名。
    es_relation_field : str
        关系文档中关系类型 ID 字段名。
    es_entity_id_field : str
        实体文档中 ID 字段名。
    es_entity_name_field : str
        实体文档中名称字段名。
    es_relation_type_id_field : str
        关系类型文档中 ID 字段名。
    es_relation_type_name_field : str
        关系类型文档中名称字段名。
    use_builtin_example : bool
        是否使用内置 CVFFAD 示例数据（默认 False）。

    图遍历 API:
    - get_forward(head, relation) → List[tail]
    - get_backward(tail, relation) → List[head]
    - get_symptom_nodes() → List[str]
    - get_fault_category_nodes() → List[str]
    """

    def __init__(
            self,
            es_config: ESConfig,
            graph_id: Optional[str] = None,
            ontology_id: Optional[str] = None,
            es_entity_index: Optional[str] = None,
            es_relation_index: Optional[str] = None,
            es_relation_type_index: Optional[str] = None,
            es_batch_size: Optional[int] = None,
            es_head_field: Optional[str] = None,
            es_relation_field: Optional[str] = None,
            es_tail_field: Optional[str] = None,
            es_entity_id_field: Optional[str] = None,
            es_entity_name_field: Optional[str] = None,
            es_relation_type_id_field: Optional[str] = None,
            es_relation_type_name_field: Optional[str] = None,
            use_builtin_example: bool = False,
            default_relation_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        # 从 config/config.yaml 加载默认值，显式传入将覆盖 YAML 配置
        schema = KnowledgeGraphSchema.default().override(
            graph_id=graph_id,
            ontology_id=ontology_id,
            batch_size=es_batch_size,
            entity_index=es_entity_index,
            relation_index=es_relation_index,
            relation_type_index=es_relation_type_index,
            entity_id_field=es_entity_id_field,
            entity_name_field=es_entity_name_field,
            head_id_field=es_head_field,
            tail_id_field=es_tail_field,
            relation_field=es_relation_field,
            relation_type_id_field=es_relation_type_id_field,
            relation_type_name_field=es_relation_type_name_field,
        )

        # 语义角色→关系名映射：优先使用显式传入，否则从 config.yaml 读取
        if default_relation_mapping is None:
            default_relation_mapping = RelationMappingConfig.default().mapping
        self._default_relation_mapping: Dict[str, str] = dict(default_relation_mapping)

        if not use_builtin_example:
            self.triples = self._load_from_es(
                es_config=es_config,
                graph_id=schema.graph_id,
                ontology_id=schema.ontology_id,
                entity_index=schema.entity_index,
                relation_index=schema.relation_index,
                relation_type_index=schema.relation_type_index,
                batch_size=schema.batch_size,
                head_field=schema.head_id_field,
                relation_field=schema.relation_field,
                tail_field=schema.tail_id_field,
                entity_id_field=schema.entity_id_field,
                entity_name_field=schema.entity_name_field,
                relation_type_id_field=schema.relation_type_id_field,
                relation_type_name_field=schema.relation_type_name_field,
            )
            self._data_source = "elasticsearch"
            logger.info("从 ES 加载三元组完成: %d 条", len(self.triples))
        else:
            self.triples = self._hardcoded_triples()
            self._data_source = "builtin"

        # ---- 1. 构建节点词汇表 ----
        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # ---- 2. 构建关系词汇表 ----
        self._build_relation_vocab()

        # ---- 3. 构建边索引和边类型（用于 R-GCN） ----
        self.edge_index, self.edge_type = self._build_rgcn_edges()

        # ---- 4. 构建图遍历映射（用于推理管线） ----
        self._build_traversal_maps()

        # ---- 5. 构建标签 ----
        # 整图训练：所有节点平等参与，不指定故障节点类型
        all_nodes = {t.head for t in self.triples} | {t.tail for t in self.triples}
        self.fault_nodes = sorted(all_nodes)
        self.labels: Dict[str, int] = {
            node: 1 for node in all_nodes
        }

        # ---- 6. 构建特征和标签张量 ----
        self.num_nodes = len(self.node_to_idx)
        self.x = torch.eye(self.num_nodes, dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(self.num_nodes)]
        self.y = torch.tensor(
            [self.labels[node] for node in ordered_nodes], dtype=torch.long
        )

    @property
    def default_relation_mapping(self) -> Dict[str, str]:
        """语义角色 → 关系名映射。

        优先使用构造/反序列化时保存的 ``_default_relation_mapping``；
        兼容旧版 checkpoint：若对象被反序列化且缺少该属性（在
        relation_mapping 配置引入之前保存的模型），自动回退到当前
        ``config.yaml`` 的 ``relation_mapping`` 配置，避免 AttributeError。
        """
        mapping = getattr(self, "_default_relation_mapping", None)
        if mapping is None:
            logger.warning(
                "数据集缺少 default_relation_mapping 属性（旧版 checkpoint），"
                "回退到 config.yaml 的 relation_mapping 配置"
            )
            mapping = RelationMappingConfig.default().mapping
        return mapping

    @default_relation_mapping.setter
    def default_relation_mapping(self, value: Dict[str, str]) -> None:
        self._default_relation_mapping = dict(value)

    # ================================================================
    # 数据加载
    # ================================================================

    @staticmethod
    def _load_from_es(
            es_config: ESConfig,
            graph_id: str,
            ontology_id: str,
            batch_size: Optional[int] = None,
            entity_index: Optional[str] = None,
            relation_index: Optional[str] = None,
            relation_type_index: Optional[str] = None,
            head_field: Optional[str] = None,
            relation_field: Optional[str] = None,
            tail_field: Optional[str] = None,
            entity_id_field: Optional[str] = None,
            entity_name_field: Optional[str] = None,
            relation_type_id_field: Optional[str] = None,
            relation_type_name_field: Optional[str] = None,
    ) -> List[Triple]:
        """从 Elasticsearch 加载三元组。

        索引名/字段名默认从 ``config/config.yaml`` 加载，显式传入将覆盖 YAML 配置。
        """
        from src.es.reader import ESKnowledgeGraphReader

        with ESKnowledgeGraphReader(es_config) as reader:
            triples = reader.fetch_triples(
                graph_id=graph_id,
                ontology_id=ontology_id,
                batch_size=batch_size,
                entity_index=entity_index,
                relation_index=relation_index,
                relation_type_index=relation_type_index,
                entity_id_field=entity_id_field,
                entity_name_field=entity_name_field,
                head_id_field=head_field,
                tail_id_field=tail_field,
                relation_field=relation_field,
                relation_type_id_field=relation_type_id_field,
                relation_type_name_field=relation_type_name_field,
            )
            if not triples:
                raise ValueError("ES 查询结果为空（无三元组）")
            return triples

    @staticmethod
    def _hardcoded_triples() -> List[Triple]:
        """从同目录的 JSON 知识库加载内置示例三元组。

        JSON 结构: [{"head": "...", "relation": "...", "tail": "..."}, ...]
        文件路径: ``src/dataset/fault_knowledge_base.json``
        """
        kb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fault_knowledge_base.json",
        )
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "知识库文件不存在: %s，返回空列表", kb_path,
            )
            return []
        except json.JSONDecodeError as e:
            logger.warning("知识库 JSON 解析失败: %s", e)
            return []

        triples: List[Triple] = []
        for item in raw:
            try:
                triples.append(
                    Triple(str(item["head"]), str(item["relation"]), str(item["tail"]))
                )
            except KeyError as e:
                logger.warning("跳过缺失字段 %s 的三元组: %s", e, item)
                continue
        logger.info("从 %s 加载内置示例三元组: %d 条", kb_path, len(triples))
        return triples

    # ================================================================
    # 构建方法
    # ================================================================

    def _build_vocab(self) -> Dict[str, int]:
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: idx for idx, node in enumerate(nodes)}

    def _build_relation_vocab(self) -> None:
        """构建关系类型词汇表。

        为每条关系分配正向索引（0..R-1）和反向索引（R..2R-1）。
        R-GCN 需要区分正向和反向边以正确学习方向性语义。
        """
        unique_relations = sorted({t.relation for t in self.triples})
        self.relation_list = unique_relations
        self.num_original_relations = len(unique_relations)
        # 总关系数 = 正向 + 反向
        self.num_relations = self.num_original_relations * 2

        self.relation_to_idx: Dict[str, int] = {
            rel: idx for idx, rel in enumerate(unique_relations)
        }
        self.idx_to_relation: Dict[int, str] = {
            idx: rel for rel, idx in self.relation_to_idx.items()
        }
        # 反向关系名称映射（仅供调试）
        self.idx_to_relation_rev: Dict[int, str] = {
            idx + self.num_original_relations: f"←{rel}"
            for rel, idx in self.relation_to_idx.items()
        }

    def _build_rgcn_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建 R-GCN 所需的边索引和边类型张量。

        每条三元组生成两条边：
        - 正向边 (head → tail, type=relation_idx)
        - 反向边 (tail → head, type=relation_idx + num_original_relations)
        """
        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        R = self.num_original_relations

        for triple in self.triples:
            h = self.node_to_idx[triple.head]
            t = self.node_to_idx[triple.tail]
            r = self.relation_to_idx[triple.relation]

            # 正向边
            edges.append((h, t))
            edge_types.append(r)
            # 反向边（类型偏移 R，让 R-GCN 学习方向性）
            edges.append((t, h))
            edge_types.append(r + R)

        return (
            torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(edge_types, dtype=torch.long),
        )

    def _build_traversal_maps(self) -> None:
        """构建图遍历映射，支持高效的推理查询。

        两种映射：
        - _forward:  relation → {head → [tails]}   正向查询
        - _backward: relation → {tail → [heads]}   反向查询
        """
        self._forward: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._backward: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for triple in self.triples:
            self._forward[triple.relation][triple.head].append(triple.tail)
            self._backward[triple.relation][triple.tail].append(triple.head)

        # 将 defaultdict 转为普通 dict（避免意外创建空条目）
        self._forward = {k: dict(v) for k, v in self._forward.items()}
        self._backward = {k: dict(v) for k, v in self._backward.items()}

    # ================================================================
    # 图遍历 API（用于推理管线）
    # ================================================================

    def get_forward(self, head: str, relation: str) -> List[str]:
        """获取与 head 通过 relation 关系相连的所有 tail 节点（正向）。

        示例:
            dataset.get_forward("发动机动力不足", "由...引起")
            → ["空气滤清器堵塞", "涡轮增压器故障", ...]
        """
        return list(self._forward.get(relation, {}).get(head, []))

    def get_backward(self, tail: str, relation: str) -> List[str]:
        """获取通过 relation 关系指向 tail 的所有 head 节点（反向）。

        示例:
            dataset.get_backward("加速迟缓，超车困难", "表现为")
            → ["发动机动力不足"]
        """
        return list(self._backward.get(relation, {}).get(tail, []))

    def get_symptom_nodes(self, relation: Optional[str] = None) -> List[str]:
        """获取所有症状描述节点（指定关系的 tail）。

        Parameters
        ----------
        relation : Optional[str]
            用于识别症状节点的关系名称（默认取自
            config.yaml 的 relation_mapping.symptoms，回退为 "表现为"）。
        """
        if relation is None:
            relation = self.default_relation_mapping.get("symptoms")
        symptoms = set()
        for triple in self.triples:
            if triple.relation == relation:
                symptoms.add(triple.tail)
        return sorted(symptoms)

    def get_fault_category_nodes(self) -> List[str]:
        """获取所有故障类别节点（三元组 head 中唯一的故障根节点名称）。

        这些是知识图谱中的核心故障概念（如"发动机动力不足"），
        它们通过症状关系（config.yaml 的 relation_mapping.symptoms）连接症状、
        通过"由...引起"连接具体原因。
        """
        symptom_relation = self.default_relation_mapping.get("symptoms")
        fault_categories = set()
        for triple in self.triples:
            if triple.relation == symptom_relation:
                fault_categories.add(triple.head)
        return sorted(fault_categories)

    def get_node_relations(
            self,
            node: str,
            relation_mapping: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """以给定节点为中心，按关系映射检索其正向相连的尾实体。

        Parameters
        ----------
        node : str
            中心节点名称。
        relation_mapping : Dict[str, str]
            输出字段名 → 关系名的映射。传入几个关系就输出几个关系的结果。

        Returns
        -------
        Dict[str, List[str]]
            {关系名: [尾实体, ...]}，key 为 relation_mapping 中的实际关系名。
        """
        result: Dict[str, List[str]] = {}
        for key, relation in relation_mapping.items():
            if self.has_relation(relation):
                result[key] = self.get_forward(node, relation)

        return result

    def has_relation(self, relation: str) -> bool:
        """判断图谱中是否存在指定关系类型。"""
        return relation in self._forward

    def get_out_neighbors(self, node: str) -> List[str]:
        """获取节点的所有正向邻居（不区分关系类型）。"""
        neighbors: set = set()
        for relation in self._forward.values():
            neighbors.update(relation.get(node, []))
        return sorted(neighbors)

    def get_in_neighbors(self, node: str) -> List[str]:
        """获取节点的所有反向邻居（不区分关系类型）。"""
        neighbors: set = set()
        for relation in self._backward.values():
            neighbors.update(relation.get(node, []))
        return sorted(neighbors)

    def get_all_neighbors(self, node: str) -> List[str]:
        """获取节点的所有邻居（双向、不区分关系类型）。"""
        return sorted(set(self.get_out_neighbors(node)) | set(self.get_in_neighbors(node)))

    # ================================================================
    # PyG Data 导出
    # ================================================================

    def to_data(self):
        """生成标准 PyG Data 对象（用于 GCNModel 兼容）。"""
        from torch_geometric.data import Data
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def to_data_with_types(self):
        """生成带 edge_type 的 PyG Data 对象（用于 R-GCN）。

        Returns
        -------
        Data
            包含 x, edge_index, edge_type, y 的 PyG Data。
        """
        from torch_geometric.data import Data
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            y=self.y,
        )

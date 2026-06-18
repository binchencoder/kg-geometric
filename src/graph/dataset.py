"""三元组到模型训练数据的转换模块。

包含：
- TripleToDatasetConverter: 将三元组列表转换为 PyG Data
- build_pipeline: 一键从 ES 读取 → 构建三元组 → 转换数据集
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from ..core.config import logger
from ..core.types import Triple


class TripleToDatasetConverter:
    """将三元组列表转换为 KGFaultDataset 所需的数据结构。

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
        """通过关系类型自动推断故障节点。"""
        faults = []
        for t in self.triples:
            if t.relation in self.fault_type_relations:
                faults.append(t.head)
        if not faults:
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
            edges.append([t, h])
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


def build_pipeline(
        entity_index: str = "knowledge_entity_index",
        relation_index: str = "knowledge_entity_relation_index",
        relation_type_index: str = "knowledge_entity_type_relation_index",
        batch_size: int = 5000,
        fault_nodes: Optional[List[str]] = None,
        **field_mapping,
) -> TripleToDatasetConverter:
    """一键构建：从 ES 读取 → 构建三元组 → 转换数据集。

    Returns
    -------
    TripleToDatasetConverter
        包含 node_to_idx, edge_index, x, y 等可直接用于 GCN 训练的属性。
    """
    from ..es.reader import ESKnowledgeGraphReader

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

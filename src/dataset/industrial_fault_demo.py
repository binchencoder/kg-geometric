"""内置示例知识图谱数据集 —— 小型工业故障知识图谱。

包含泵/电机/齿轮箱/压缩机的故障模式演示数据，
用于快速测试和原型验证。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.core.types import Triple


class KGFaultDemoDataset:
    """小型工业故障知识图谱，包含 13 条三元组。

    覆盖泵、电机、齿轮箱、压缩机的常见故障模式。
    用于快速原型验证，无需连接 ES。
    """

    def __init__(self) -> None:
        self.triples: List[Triple] = [
            Triple("泵_01", "存在症状", "振动过高"),
            Triple("泵_01", "存在症状", "温度过高"),
            Triple("泵_01", "原因在于", "轴承磨损"),
            Triple("电机_02", "存在症状", "电流过高"),
            Triple("电机_02", "原因在于", "定子故障"),
            Triple("齿轮箱_03", "存在症状", "噪音异常"),
            Triple("齿轮箱_03", "原因在于", "齿轮磨损"),
            Triple("压缩机_04", "存在症状", "压力过低"),
            Triple("压缩机_04", "原因在于", "阀门泄漏"),
            Triple("轴承磨损", "类型为", "故障"),
            Triple("定子故障", "类型为", "故障"),
            Triple("齿轮磨损", "类型为", "故障"),
            Triple("阀门泄漏", "类型为", "故障"),
        ]

        self.fault_nodes = ["轴承磨损", "定子故障", "齿轮磨损", "阀门泄漏"]
        self.labels: Dict[str, int] = {
            "轴承磨损": 1, "定子故障": 1, "齿轮磨损": 1, "阀门泄漏": 1, "故障": 1,
            "泵_01": 0, "电机_02": 0, "齿轮箱_03": 0, "压缩机_04": 0,
            "振动过高": 0, "温度过高": 0, "电流过高": 0, "噪音异常": 0, "压力过低": 0,
        }

        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.edge_index = self._build_edge_index()
        self.x = torch.eye(len(self.node_to_idx), dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(len(self.idx_to_node))]
        self.y = torch.tensor([self.labels[node] for node in ordered_nodes], dtype=torch.long)

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

    def to_data(self):
        """生成 PyG Data 对象。"""
        from torch_geometric.data import Data
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

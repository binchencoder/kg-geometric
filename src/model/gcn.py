"""GCN 模型 —— 用于知识图谱节点分类的 2 层图卷积网络。

提供 GCNModel 类，支持 Data 和 HeteroData 两种图数据结构。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    """2 层 GCN + 线性分类器，可用于知识图谱故障节点分类。

    可同时接受 PyG Data 和 HeteroData 两种图格式。
    通过 encode() 方法可获取中间节点嵌入，用于相似度推理。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _extract(data) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容 Data 和 HeteroData 两种类型，提取 (x, edge_index)。"""
        if isinstance(data, HeteroData):
            x = data["entity"].x
            edge_index = data["entity", "to", "entity"].edge_index
        else:
            x = data.x
            edge_index = data.edge_index
        return x, edge_index

    def encode(self, data) -> torch.Tensor:
        """前向传播获取节点嵌入（不含分类器）。

        用于 Top-K 静态链接预测中的余弦相似度计算。
        """
        x, edge_index = self._extract(data)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)

    def forward(self, data) -> torch.Tensor:
        """完整前向传播：嵌入 + 分类 logits。"""
        return self.classifier(self.encode(data))

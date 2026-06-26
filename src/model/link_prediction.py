"""链接预测模型 —— GCN编码器 + DistMult解码器。

用于知识图谱链接预测任务：给定(head, relation, ?)，预测最可能的 tail 实体。
架构：GCN编码器生成节点嵌入 → DistMult解码器对三元组进行评分。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv


class LinkPredictionGCN(nn.Module):
    """2层GCN编码器 + DistMult解码器，用于知识图谱链接预测。

    编码器：GCNConv × 2 → 节点嵌入
    解码器：DistMult（h * r * t 点积求和）对三元组评分

    支持 Data 和 HeteroData 两种图格式。
    支持关系嵌入维度可配，关系数量自动从数据中推断或手动指定。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_relations: int = 1,
        relation_dim: Optional[int] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        in_dim : int
            输入特征维度（通常为节点数，one-hot 或随机嵌入）。
        hidden_dim : int
            GCN 隐藏层维度，同时也是输出嵌入维度。
        num_relations : int
            知识图谱关系类型总数。
        relation_dim : Optional[int]
            关系嵌入维度，默认与 hidden_dim 相同。
        dropout : float
            Dropout 比率。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim or hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout

        # GCN 编码器：2层图卷积
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 关系嵌入表：每种关系类型一个可学习向量
        self.relation_emb = nn.Embedding(num_relations, self.relation_dim)

        # 将 hidden_dim 投影到 relation_dim（如果需要）
        if hidden_dim != self.relation_dim:
            self.proj = nn.Linear(hidden_dim, self.relation_dim)
        else:
            self.proj = None

    @staticmethod
    def _extract(
        data,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容 Data 和 HeteroData，提取 (x, edge_index)。"""
        if isinstance(data, HeteroData):
            x = data["entity"].x
            edge_index = data["entity", "to", "entity"].edge_index
        else:
            x = data.x
            edge_index = data.edge_index
        return x, edge_index

    def encode(self, data) -> torch.Tensor:
        """GCN前向传播，获取所有节点的嵌入向量。

        Returns
        -------
        torch.Tensor
            形状 [num_nodes, hidden_dim] 的节点嵌入矩阵。
        """
        x, edge_index = self._extract(data)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, hidden_dim]

    def decode(
        self,
        head_emb: torch.Tensor,
        rel_idx: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """DistMult 解码器：对三元组 (head, relation, tail) 进行评分。

        评分公式：sum(head_emb * relation_emb * tail_emb, dim=-1)

        Parameters
        ----------
        head_emb : torch.Tensor
            头实体嵌入 [batch_size, hidden_dim]。
        rel_idx : torch.Tensor
            关系类型索引 [batch_size] 或标量。
        tail_emb : torch.Tensor
            尾实体嵌入 [batch_size, hidden_dim] 或 [num_nodes, hidden_dim]。

        Returns
        -------
        torch.Tensor
            三元组评分 [batch_size] 或 [batch_size, num_nodes]（当 tail_emb 为全量时）。
        """
        # 投影到 relation_dim
        if self.proj is not None:
            head_for_score = self.proj(head_emb)
        else:
            head_for_score = head_emb

        # 获取关系嵌入
        rel_emb = self.relation_emb(rel_idx)  # [batch_size, relation_dim]

        # DistMult 评分: 逐元素乘积后求和
        if tail_emb.dim() == 2 and tail_emb.size(0) != head_for_score.size(0):
            # tail_emb 是全量实体嵌入 [num_nodes, relation_dim]，需要扩展为批量评分
            # head_for_score: [B, D], rel_emb: [B, D], tail_emb: [N, D]
            # 结果: [B, N]
            if self.proj is not None:
                tail_for_score = self.proj(tail_emb)
            else:
                tail_for_score = tail_emb
            # (head * rel) @ tail.T 等价于 DistMult
            return torch.mm(
                head_for_score * rel_emb, tail_for_score.t()
            )  # [B, N]
        else:
            # 逐对评分
            if self.proj is not None and tail_emb.dim() == 2:
                tail_for_score = self.proj(tail_emb)
            else:
                tail_for_score = tail_emb
            return torch.sum(
                head_for_score * rel_emb * tail_for_score, dim=-1
            )  # [B]

    def forward(
        self,
        data,
        head_idx: torch.Tensor,
        rel_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """完整前向传播：编码 + 解码。

        Parameters
        ----------
        data : Data or HeteroData
            图数据。
        head_idx : torch.Tensor
            头实体索引 [batch_size]。
        rel_idx : torch.Tensor
            关系类型索引 [batch_size]。
        tail_idx : torch.Tensor
            尾实体索引 [batch_size]。

        Returns
        -------
        torch.Tensor
            三元组评分 [batch_size]。
        """
        node_emb = self.encode(data)  # [num_nodes, hidden_dim]
        head_emb = node_emb[head_idx]
        tail_emb = node_emb[tail_idx]
        return self.decode(head_emb, rel_idx, tail_emb)

    def predict_all_tails(
        self,
        data,
        head_idx: torch.Tensor,
        rel_idx: torch.Tensor,
    ) -> torch.Tensor:
        """给定头实体和关系，对所有候选尾实体进行评分。

        用于链接预测推理：已知 (h, r, ?)，对所有实体评分并排序。

        Parameters
        ----------
        data : Data or HeteroData
            图数据。
        head_idx : torch.Tensor
            头实体索引 [batch_size]。
        rel_idx : torch.Tensor
            关系类型索引 [batch_size]。

        Returns
        -------
        torch.Tensor
            评分矩阵 [batch_size, num_nodes]，每行对应一个查询的全实体评分。
        """
        node_emb = self.encode(data)
        head_emb = node_emb[head_idx]
        # 使用全量节点嵌入作为候选尾实体
        return self.decode(head_emb, rel_idx, node_emb)

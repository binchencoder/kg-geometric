"""R-GCN 模型 —— 关系图卷积网络用于知识图谱故障诊断。

使用 PyG 的 RGCNConv 实现多关系感知的节点嵌入学习，
能够区分"表现为"、"由...引起"、"维修措施"等不同关系类型的语义差异，
这是标准 GCN 无法做到的。

参考文献:
    Modeling Relational Data with Graph Convolutional Networks (Schlichtkrull et al., ESWC 2018)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class FaultRGCN(nn.Module):
    """多层 R-GCN + 分类器，用于知识图谱故障节点分类。

    关键特性：
    - 可学习节点嵌入（Embedding）作为初始特征
    - 多层 RGCNConv 进行关系感知消息传递
    - 每层可选 basis decomposition 减少参数量
    - Dropout 正则化防止过拟合
    - Linear 分类器输出故障/正常二分类 logits

    Parameters
    ----------
    num_nodes : int
        图中实体节点总数。
    num_relations : int
        关系类型总数（含正向+反向边的关系类型）。
    hidden_dim : int
        隐藏层维度（节点嵌入维度），默认 64。
    num_layers : int
        R-GCN 层数，默认 2。
    dropout : float
        Dropout 比例，默认 0.3。
    num_bases : Optional[int]
        Basis decomposition 的基数量。None 表示不使用。
        对于关系类型较多的场景，可设为较小的值（如 4-8）以减少参数。
    """

    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            num_bases: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 可学习节点嵌入（替代 one-hot / 随机特征）
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # 多层 RGCNConv
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            is_first = (layer_idx == 0)
            in_dim = hidden_dim  # 所有层使用相同维度
            self.convs.append(
                RGCNConv(
                    in_dim,
                    hidden_dim,
                    num_relations,
                    num_bases=num_bases,
                )
            )

        # 节点分类器
        self.classifier = nn.Linear(hidden_dim, 2)

    def encode(
            self,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """编码所有节点，输出嵌入向量 [num_nodes, hidden_dim]。

        用于相似度计算和下游语义匹配。
        """
        x = self.node_emb.weight  # [num_nodes, hidden_dim]
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def forward(
            self,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，输出节点分类 logits [num_nodes, 2]。

        Returns
        -------
        torch.Tensor
            每个节点的二分类 logits，[:, 0] = normal, [:, 1] = fault。
        """
        emb = self.encode(edge_index, edge_type)
        return self.classifier(emb)

    def link_score(
            self,
            edge_index: torch.Tensor,
            edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算边的评分（DistMult 解码器）。

        对给定的候选边 (head, tail) 计算 DistMult 分数：
        score = sum(head_emb * rel_emb * tail_emb)

        用于推理阶段的候选故障节点排序。

        Parameters
        ----------
        edge_index : torch.Tensor [2, num_edges]
            候选边索引。
        edge_type : Optional[torch.Tensor] [num_edges]
            边类型索引（当前未使用，保留接口兼容）。

        Returns
        -------
        torch.Tensor [num_edges]
            每条候选边的评分。
        """
        _ = edge_type  # 当前版本使用内积评分，不依赖 edge_type
        head_emb = self.node_emb(edge_index[0])
        tail_emb = self.node_emb(edge_index[1])
        # 内积评分
        scores = (head_emb * tail_emb).sum(dim=-1)
        return scores

    def get_node_embedding(self, node_idx: int) -> torch.Tensor:
        """获取单个节点的嵌入向量（含 R-GCN 编码）。"""
        return self.node_emb(torch.tensor(node_idx, device=self.node_emb.weight.device))

    def get_all_embeddings(self) -> torch.Tensor:
        """获取原始嵌入（不经 R-GCN 编码，仅 Embedding 层输出）。"""
        return self.node_emb.weight

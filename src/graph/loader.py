"""PyG NeighborLoader 适配器 —— 将 ES 流式构建的全图适配为 PyG NeighborLoader。

适用场景：知识图谱足够小（边数 < 数千万），可以全量加载到内存。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.core.config import logger


class KGNeighborLoaderAdapter:
    """将 ES 流式构建的全图适配为 PyG NeighborLoader。

    使用流程：
    1. 用 ESTripletStreamer 全量遍历构建全局图
    2. 用本适配器创建 NeighborLoader
    3. 标准 PyG 训练循环
    """

    def __init__(
            self,
            vocab,
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
            每跳采样邻居数，如 [25, 10]。
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
            streamer,
            vocab=None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            embedding_dim: int = 64,
            node_features: Optional[torch.Tensor] = None,
    ) -> "KGNeighborLoaderAdapter":
        """从 ESTripletStreamer 一键构建全图 + NeighborLoader 适配器。"""
        if vocab is None:
            from ..es.vocabulary import KGVocabulary
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

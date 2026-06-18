"""流式训练流水线 —— "边查边训"引擎。

将异步 ES 子图采样与 PyG 训练循环集成，支持全图和流式两种模式。
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from torch_geometric.loader import NeighborLoader

from ..core.config import logger
from ..es.streamer import ESTripletStreamer
from ..es.vocabulary import KGVocabulary
from ..graph.loader import KGNeighborLoaderAdapter
from ..graph.sampler import AsyncSubgraphSampler


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

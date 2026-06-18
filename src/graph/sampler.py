"""异步子图采样器 —— 基于 ES + search_after 动态构建局部异质子图。

结合 PyG NeighborLoader 的思路，但直接从 ES 动态构建子图，
无需将全图加载到内存，适合上亿级知识图谱。
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..core.config import logger

if TYPE_CHECKING:
    from ..es.resolver import IDNameResolver
    from ..es.streamer import ESTripletStreamer
    from ..es.vocabulary import KGVocabulary


class AsyncSubgraphSampler:
    """基于 ES + search_after 的异步子图采样器。

    工作流程：
    1. 给定一批种子节点，异步查询 ES 获取它们的 k-hop 邻居边
    2. 构建局部 HeteroData 子图
    3. 通过预取队列实现 ES I/O 与 GPU 计算的流水线重叠
    """

    def __init__(
            self,
            streamer: "ESTripletStreamer",
            vocab: "KGVocabulary",
            num_hops: int = 2,
            max_neighbors_per_hop: int = 100,
            prefetch_size: int = 2,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            embedding_dim: int = 64,
            resolver: Optional["IDNameResolver"] = None,
    ):
        """
        Parameters
        ----------
        streamer : ESTripletStreamer
            search_after 流式提取器。
        vocab : KGVocabulary
            全局实体/关系 ID 映射（存名称）。
        num_hops : int
            邻居采样的跳数。
        max_neighbors_per_hop : int
            每跳最大邻居数（防止度数爆炸）。
        prefetch_size : int
            预取子图数量。
        head_field / relation_field / tail_field : str
            ES 索引中的字段名。
        embedding_dim : int
            实体特征嵌入维度。
        resolver : Optional[IDNameResolver]
            ID↔名称解析器，用于将 seed 实体名称反解为 ES 查询所需的原始 ID。
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
        self.resolver = resolver

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
        edges: List[Tuple[int, int, int]] = []
        local_entities: Dict[str, int] = {}

        _resolve_names = (
            self.resolver.resolve_entity_names_to_ids
            if self.resolver and self.resolver.is_ready
            else lambda names: names
        )

        for hop in range(hops):
            if not frontier:
                break

            frontier_list = list(frontier)
            next_frontier: Set[str] = set()

            chunk_size = 1000
            for i in range(0, len(frontier_list), chunk_size):
                chunk_raw = frontier_list[i:i + chunk_size]
                chunk = _resolve_names(chunk_raw)

                for batch in self.streamer.stream_triplets(
                        seed_entities=chunk,
                        head_field=self.head_field,
                        relation_field=self.relation_field,
                        tail_field=self.tail_field,
                        resume=False,
                ):
                    for t in batch:
                        h, r, tail = t["head"], t["relation"], t["tail"]

                        if h not in local_entities:
                            local_entities[h] = len(local_entities)
                        if tail not in local_entities:
                            local_entities[tail] = len(local_entities)

                        global_rel_idx = self.vocab.add_relation(r)

                        edges.append((
                            local_entities[h],
                            global_rel_idx,
                            local_entities[tail],
                        ))

                        if tail not in visited and len(next_frontier) < self.max_neighbors_per_hop:
                            next_frontier.add(tail)
                            visited.add(tail)

            frontier = next_frontier

        if not edges:
            return None

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
            self._prefetch_queue.put(None)
        except Exception as e:
            self._prefetch_error = e
            self._prefetch_queue.put(None)

    def iter_subgraphs(
            self,
            seed_batches: List[List[str]],
    ) -> Iterator[Tuple[List[str], Optional[HeteroData]]]:
        """同步迭代器：逐个返回 (种子节点, 子图)。"""
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

        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break

        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(iter(seed_batches),),
            daemon=True,
            name="es-subgraph-prefetch",
        )
        self._prefetch_thread.start()

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

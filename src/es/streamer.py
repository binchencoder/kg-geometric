"""ES 流式三元组提取器 —— search_after 模式。

使用 search_after 替代 scroll/scan，避免深分页性能灾难。
支持全量流式遍历、种子实体过滤、断点续传、异步预取。
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Set, Tuple, Union

from elasticsearch import Elasticsearch

from ..core.config import logger

if TYPE_CHECKING:
    from .resolver import IDNameResolver


class ESTripletStreamer:
    """使用 search_after 替代 scroll/scan，避免深分页性能灾难。

    search_after 相比 scroll 的优势：
    - 无状态分页，不占用 ES 服务端 scroll 上下文
    - 深分页性能稳定，不会随 offset 增大而退化
    - 天然支持断点续传
    """

    # 默认 ES 查询超时（秒）
    DEFAULT_TIMEOUT = 120
    # 预取队列最大容量
    MAX_QUEUE_SIZE = 4

    def __init__(
            self,
            es_hosts: List[str],
            index_name: Union[str, List[str]],
            batch_size: int = 5000,
            prefetch: bool = True,
            checkpoint_dir: Optional[str] = None,
            resolver: Optional["IDNameResolver"] = None,
    ):
        """
        Parameters
        ----------
        es_hosts : List[str]
            ES 节点地址列表，如 ["http://host1:9200", "http://host2:9200"]。
        index_name : str or List[str]
            ES 索引名称，支持单个索引或索引列表。
        batch_size : int
            每批返回的文档数。
        prefetch : bool
            是否启用异步预取（后台线程提前拉取下一批）。
        checkpoint_dir : Optional[str]
            断点续传目录，用于保存/恢复 progress。
        resolver : Optional[IDNameResolver]
            ID→名称解析器。若提供，stream_triplets_raw 将在产出前自动将
            原始 entityId/relationTypeId 解析为可读名称。
        """
        self.es = Elasticsearch(es_hosts)
        if isinstance(index_name, list):
            self.index = ",".join(index_name)
            self._index_label = "_".join(sorted(index_name))
        else:
            self.index = index_name
            self._index_label = index_name
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.resolver = resolver
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 断点续传 ----------
    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self._index_label}_progress.json"

    def _save_progress(self, search_after: Optional[List]) -> None:
        if self.checkpoint_dir is None:
            return
        with open(self._checkpoint_path(), "w") as f:
            json.dump({"search_after": search_after}, f)

    def _load_progress(self) -> Optional[List]:
        if self.checkpoint_dir is None:
            return None
        cp = self._checkpoint_path()
        if not cp.exists():
            return None
        with open(cp) as f:
            data = json.load(f)
            return data.get("search_after")

    def _clear_progress(self) -> None:
        if self.checkpoint_dir is None:
            return
        cp = self._checkpoint_path()
        if cp.exists():
            cp.unlink()

    # ---------- 核心：search_after 流式遍历 ----------
    def _build_query(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            tail_field: str = "tail_id",
            extra_filters: Optional[dict] = None,
    ) -> dict:
        """构建带排序的 search_after 查询体。"""
        if seed_entities:
            query = {"terms": {head_field: seed_entities}}
        elif extra_filters:
            query = extra_filters
        else:
            query = {"match_all": {}}

        return {
            "query": query,
            "sort": [
                {head_field: "asc"},
                {tail_field: "asc"},
                {"_id": "asc"},
            ],
            "size": self.batch_size,
        }

    def stream_triplets_raw(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            resume: bool = True,
            extra_filters: Optional[dict] = None,
    ) -> Iterator[List[dict]]:
        """流式生成原始三元组批次（dict 格式）。

        若配置了 resolver，每批会在产出前自动将原始 ID 解析为可读名称；
        无法解析的三元组会被跳过。

        Yields
        ------
        List[dict]
            每批三元组列表，每个三元组为 {"head": ..., "relation": ..., "tail": ...}。
        """
        query_body = self._build_query(
            seed_entities=seed_entities,
            head_field=head_field,
            tail_field=tail_field,
            extra_filters=extra_filters,
        )

        search_after = self._load_progress() if resume else None
        if search_after:
            logger.info(
                "断点续传: 从 search_after=%s 继续遍历索引 %s",
                search_after, self.index,
            )

        batch_count = 0
        total_docs = 0
        total_skipped = 0
        _resolver = self.resolver

        # 若配置了 resolver 但未就绪，在开始遍历时给出明确警告
        if _resolver and not _resolver.is_ready:
            logger.error(
                "❌ ID→名称解析器已配置但未就绪！"
                "实体名称映射=%d, 关系类型名称映射=%d。"
                "词汇表将存储原始 ID 而非名称！"
                "请检查 --graph-id / --ontology-id / --entity-name-field 参数。",
                len(_resolver.entity_map), len(_resolver.relation_type_map),
            )

        while True:
            if search_after:
                query_body["search_after"] = search_after

            try:
                resp = self.es.search(
                    index=self.index,
                    body=query_body,
                    request_timeout=self.DEFAULT_TIMEOUT,
                )
            except Exception as e:
                logger.error("ES search 失败 (batch=%d, search_after=%s): %s",
                             batch_count, search_after, e)
                raise

            hits = resp["hits"]["hits"]
            if not hits:
                break

            batch_count += 1
            total_docs += len(hits)

            # 提取三元组（原始 ID）
            triplets = []
            for hit in hits:
                src = hit["_source"]
                h = src.get(head_field)
                r = src.get(relation_field)
                t = src.get(tail_field)
                if h is not None and r is not None and t is not None:
                    triplets.append({
                        "head": str(h),
                        "relation": str(r),
                        "tail": str(t),
                    })

            # 若配置了 ID→名称解析器，将原始 ID 解析为可读名称
            if _resolver and _resolver.is_ready:
                triplets, _, skipped = _resolver.resolve_triplets(triplets)
                total_skipped += skipped

            if triplets:
                yield triplets

            # 更新游标
            search_after = hits[-1]["sort"]
            self._save_progress(search_after)

            if len(hits) < self.batch_size:
                break

        self._clear_progress()
        logger.info(
            "search_after 遍历完成: 共 %d 批, %d 条文档, 已解析名称 (跳过 %d 条)",
            batch_count, total_docs, total_skipped,
        )

    def stream_triplets(
            self,
            seed_entities: Optional[List[str]] = None,
            head_field: str = "head_id",
            relation_field: str = "relation",
            tail_field: str = "tail_id",
            resume: bool = True,
            extra_filters: Optional[dict] = None,
    ) -> Iterator[List[dict]]:
        """带异步预取的流式三元组迭代器（生产者-消费者模式）。

        生产者线程：从 ES search_after 拉取数据放入队列
        消费者（主线程）：从队列取出数据进行训练

        这使得 ES I/O 延迟被 GPU 计算完全掩盖，实现真正的"边查边训"。
        """
        if not self.prefetch:
            yield from self.stream_triplets_raw(
                seed_entities=seed_entities,
                head_field=head_field,
                relation_field=relation_field,
                tail_field=tail_field,
                resume=resume,
                extra_filters=extra_filters,
            )
            return

        # --- 异步预取模式 ---
        data_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        stop_sentinel = object()
        exception_holder: List[Exception] = []

        def producer() -> None:
            """生产者：在后台线程中从 ES 拉取数据。"""
            try:
                for batch in self.stream_triplets_raw(
                        seed_entities=seed_entities,
                        head_field=head_field,
                        relation_field=relation_field,
                        tail_field=tail_field,
                        resume=resume,
                        extra_filters=extra_filters,
                ):
                    data_queue.put(batch)
                data_queue.put(stop_sentinel)
            except Exception as e:
                exception_holder.append(e)
                data_queue.put(stop_sentinel)

        thread = threading.Thread(target=producer, daemon=True, name="es-producer")
        thread.start()

        while True:
            batch = data_queue.get()
            if batch is stop_sentinel:
                break
            yield batch

        thread.join(timeout=10)
        if exception_holder:
            raise exception_holder[0]

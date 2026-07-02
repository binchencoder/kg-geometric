"""图数据层：知识图谱子图采样与加载。

包含：
- AsyncSubgraphSampler: 基于 ES 的异步子图采样器
- KGNeighborLoaderAdapter: PyG NeighborLoader 适配器

注意：数据集相关类已迁移至 src.dataset 模块，此处保留重导出以保持向后兼容。
"""

# 从新位置重导出数据集相关类（向后兼容）
from src.dataset import (
    TripleToDatasetConverter,
    build_pipeline,
    KGFaultDataset,
    LinkPredictionData,
    LinkPredictionStreamingData,
)
from .loader import KGNeighborLoaderAdapter
from .sampler import AsyncSubgraphSampler

__all__ = [
    "TripleToDatasetConverter",
    "build_pipeline",
    "KGFaultDataset",
    "KGNeighborLoaderAdapter",
    "AsyncSubgraphSampler",
    "LinkPredictionData",
    "LinkPredictionStreamingData",
]

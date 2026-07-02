"""图数据层：知识图谱子图构建、采样与加载。

包含：
- TripleToDatasetConverter: 三元组列表 → PyG Data 转换器
- KGFaultDataset: 内置小型示例故障知识图谱
- AsyncSubgraphSampler: 基于 ES 的异步子图采样器
- KGNeighborLoaderAdapter: PyG NeighborLoader 适配器
- LinkPredictionData: 链接预测数据集（全量模式，边划分 + 负采样）
- LinkPredictionStreamingData: 流式链接预测数据集（海量三元组，边查边训）
"""

from .dataset import TripleToDatasetConverter, build_pipeline
from .kg_fault_demo import KGFaultDataset
from .link_prediction_dataset import LinkPredictionData, LinkPredictionStreamingData
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

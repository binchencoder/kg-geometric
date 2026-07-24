"""知识图谱几何学习 —— 工业静态链接预测与链接预测工具包。

基于 Elasticsearch 知识图谱 + GCN 的端到端静态链接预测与链接预测系统。

子包结构：
- core/  : 全局配置、基础数据类型
- es/    : ES 连接、读取、流式提取、ID 解析、词汇表
- graph/ : 图数据构建、子图采样、NeighborLoader 适配、链接预测数据集
- model/ : GCN 模型、故障标签、训练工具
- pipeline/: 训练流水线与 Top-K 静态链接预测推理
"""

from .core import ESConfig, Triple, BatchProgress, logger
from .es import (
    create_es_client, ping_es, index_exists, list_indices,
    get_index_mapping, safe_get,
    ESKnowledgeGraphReader,
    IDNameResolver,
    ESTripletStreamer,
    KGVocabulary,
)
from .graph import (
    TripleToDatasetConverter, build_pipeline,
    KGTripleDataset, KGNeighborLoaderAdapter, AsyncSubgraphSampler,
    LinkPredictionData, LinkPredictionStreamingData,
)
from .model import (
    GCNModel, FaultLabelBuilder, split_masks, train, evaluate,
)
from .pipeline import (
    StreamingTrainingPipeline, KGTrainInferPipeline,
    topk_static_link_prediction, print_topk_static_link_prediction,
)

__all__ = [
    # core
    "ESConfig", "Triple", "BatchProgress", "logger",
    # es
    "create_es_client", "ping_es", "index_exists", "list_indices",
    "get_index_mapping", "safe_get",
    "ESKnowledgeGraphReader", "IDNameResolver",
    "ESTripletStreamer", "KGVocabulary",
    # graph
    "TripleToDatasetConverter", "build_pipeline",
    "KGTripleDataset", "KGNeighborLoaderAdapter", "AsyncSubgraphSampler",
    "LinkPredictionData", "LinkPredictionStreamingData",
    # model
    "GCNModel", "FaultLabelBuilder", "split_masks", "train", "evaluate",
    # pipeline
    "StreamingTrainingPipeline", "KGTrainInferPipeline",
    "topk_static_link_prediction", "print_topk_static_link_prediction",
]

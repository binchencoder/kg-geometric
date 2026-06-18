"""知识图谱几何学习 —— 工业故障诊断工具包。

基于 Elasticsearch 知识图谱 + GCN 的端到端故障诊断系统。

子包结构：
- core/  : 全局配置、基础数据类型
- es/    : ES 连接、读取、流式提取、ID 解析、词汇表
- graph/ : 图数据构建、子图采样、NeighborLoader 适配
- model/ : GCN 模型、故障标签、训练工具
- pipeline/: 训练流水线与 Top-K 故障诊断推理
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
    KGFaultDataset, KGNeighborLoaderAdapter, AsyncSubgraphSampler,
)
from .model import FaultGCN, FaultLabelBuilder, split_masks, train, evaluate
from .pipeline import (
    StreamingTrainingPipeline, KGTrainInferPipeline,
    topk_fault_diagnosis, print_topk_diagnosis,
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
    "KGFaultDataset", "KGNeighborLoaderAdapter", "AsyncSubgraphSampler",
    # model
    "FaultGCN", "FaultLabelBuilder", "split_masks", "train", "evaluate",
    # pipeline
    "StreamingTrainingPipeline", "KGTrainInferPipeline",
    "topk_fault_diagnosis", "print_topk_diagnosis",
]

"""数据集模块 —— 知识图谱数据集定义与转换。

包含：
- TripleToDatasetConverter: 三元组列表 → PyG Data 转换器
- build_pipeline: 一键从 ES 读取 → 构建三元组 → 转换数据集
- KGFaultDataset: 车辆故障诊断知识图谱（内置示例数据集）
- KGFaultDemoDataset: 工业故障小型演示数据集
- LinkPredictionData: 链接预测数据集（全量模式，边划分 + 负采样）
- LinkPredictionStreamingData: 流式链接预测数据集（海量三元组，边查边训）
"""

from .converter import TripleToDatasetConverter, build_pipeline
from .kg_fault_demo import KGFaultDataset
from .industrial_fault_demo import KGFaultDemoDataset
from .link_prediction import LinkPredictionData, LinkPredictionStreamingData

__all__ = [
    "TripleToDatasetConverter",
    "build_pipeline",
    "KGFaultDataset",
    "KGFaultDemoDataset",
    "LinkPredictionData",
    "LinkPredictionStreamingData",
]

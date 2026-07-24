"""数据集模块 —— 知识图谱数据集定义与转换。

包含：
- TripleToDatasetConverter: 三元组列表 → PyG Data 转换器
- build_pipeline: 一键从 ES 读取 → 构建三元组 → 转换数据集
- KGTripleDataset: 车辆静态链接预测知识图谱（内置示例数据集）
- KGFaultDemoDataset: 工业故障小型演示数据集
- LinkPredictionData: 链接预测数据集（全量模式，边划分 + 负采样）
- LinkPredictionStreamingData: 流式链接预测数据集（海量三元组，边查边训）
- TransformerTemporalKG: 电力变压器时序异构图知识图谱（ETTh1/ETTh2 CSV 加载）
"""

from src.dataset.converter import TripleToDatasetConverter, build_pipeline
from src.dataset.triple_dataset import KGTripleDataset
from src.dataset.industrial_fault_demo import KGFaultDemoDataset
from src.dataset.link_prediction import LinkPredictionData, LinkPredictionStreamingData
from src.dataset.temporal_dataset import (
    KGTemporalDataset,
    load_and_preprocess_dataset,
    build_transformer_kg,
)

__all__ = [
    "KGTripleDataset",
    "KGTemporalDataset",
    "KGFaultDemoDataset",
    "TripleToDatasetConverter",
    "build_pipeline",
    "LinkPredictionData",
    "LinkPredictionStreamingData",
    "load_and_preprocess_dataset",
    "build_transformer_kg",
]
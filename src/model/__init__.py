"""模型层：GCN/R-GCN/TGN 网络、故障标签构建、训练工具。

包含：
- FaultGCN: 2 层 GCN + 线性分类器（节点分类，简单基线）
- FaultRGCN: 多层 R-GCN + 分类器（关系感知，推荐使用）
- TGN / TGNOilTemperaturePredict: 时序异构图网络（油温预测 + 风险评分）
- FaultLabelBuilder: 基于关系模式自动识别故障节点
- split_masks / train / evaluate: 训练辅助函数（支持 7:1:2 划分）
- LinkPredictionGCN: GCN编码器 + DistMult解码器（链接预测）
"""

from .gcn import FaultGCN
from .rgcn import FaultRGCN
from .tgn import TGN, TGNOilTemperaturePredict
from .labels import FaultLabelBuilder
from .link_prediction import LinkPredictionGCN
from .link_prediction_training import (
    train_link_prediction,
    evaluate_link_prediction,
    predict_top_k,
    print_link_prediction_results,
    train_link_prediction_streaming,
    evaluate_link_prediction_streaming,
    predict_top_k_streaming,
)
from .training import split_masks, train, evaluate, train_rgcn

__all__ = [
    "FaultGCN",
    "FaultRGCN",
    "TGN",
    "TGNOilTemperaturePredict",
    "FaultLabelBuilder",
    "split_masks",
    "train",
    "train_rgcn",
    "evaluate",
    "LinkPredictionGCN",
    "train_link_prediction",
    "evaluate_link_prediction",
    "predict_top_k",
    "print_link_prediction_results",
    "train_link_prediction_streaming",
    "evaluate_link_prediction_streaming",
    "predict_top_k_streaming",
]
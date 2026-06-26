"""模型层：GCN 网络、故障标签构建、训练工具与链接预测。

包含：
- FaultGCN: 2 层 GCN + 线性分类器（节点分类）
- FaultLabelBuilder: 基于关系模式自动识别故障节点
- split_masks / train / evaluate: 节点分类训练辅助函数
- LinkPredictionGCN: GCN编码器 + DistMult解码器（链接预测）
- train_link_prediction / evaluate_link_prediction / predict_top_k: 全图模式训练与推理
- train_link_prediction_streaming / evaluate_link_prediction_streaming / predict_top_k_streaming: 流式子图采样训练与推理
"""

from .gcn import FaultGCN
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
from .training import split_masks, train, evaluate

__all__ = [
    "FaultGCN",
    "FaultLabelBuilder",
    "split_masks",
    "train",
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

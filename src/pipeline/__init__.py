"""推理流水线层：训练流水线与静态链接预测推理。

包含：
- StreamingTrainingPipeline: "边查边训"流水线
- KGTrainInferPipeline: 端到端训练+推理管线
- topk_static_link_prediction / print_topk_static_link_prediction: Top-K 静态链接预测
"""

from .slp import topk_static_link_prediction, print_topk_static_link_prediction
from .pipeline import StreamingTrainingPipeline
from .train_infer import KGTrainInferPipeline

__all__ = [
    "StreamingTrainingPipeline",
    "KGTrainInferPipeline",
    "topk_static_link_prediction",
    "print_topk_static_link_prediction",
]

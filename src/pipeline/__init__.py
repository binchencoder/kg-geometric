"""推理流水线层：训练流水线与故障诊断推理。

包含：
- StreamingTrainingPipeline: "边查边训"流水线
- KGTrainInferPipeline: 端到端训练+推理管线
- topk_fault_diagnosis / print_topk_diagnosis: Top-K 故障诊断
"""

from .diagnosis import topk_fault_diagnosis, print_topk_diagnosis
from .pipeline import StreamingTrainingPipeline
from .train_infer import KGTrainInferPipeline

__all__ = [
    "StreamingTrainingPipeline",
    "KGTrainInferPipeline",
    "topk_fault_diagnosis",
    "print_topk_diagnosis",
]

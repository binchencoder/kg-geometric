"""模型层：GCN 网络、故障标签构建与训练工具。

包含：
- FaultGCN: 2 层 GCN + 线性分类器
- FaultLabelBuilder: 基于关系模式自动识别故障节点
- split_masks / train / evaluate: 训练辅助函数
"""

from .gcn import FaultGCN
from .labels import FaultLabelBuilder
from .training import split_masks, train, evaluate

__all__ = [
    "FaultGCN",
    "FaultLabelBuilder",
    "split_masks",
    "train",
    "evaluate",
]

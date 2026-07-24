"""模型层：GCN/R-GCN/TGN 网络、故障标签构建、训练工具。

包含：
- GCNModel: 2 层 GCN + 线性分类器（节点分类，简单基线）
- FaultRGCN: 多层 R-GCN + 分类器（关系感知，推荐使用）
- TGN / TGNModel: 时序异构图网络（油温预测 + 风险评分）
- FaultLabelBuilder: 基于关系模式自动识别故障节点
- split_masks / train / evaluate: 训练辅助函数（支持 7:1:2 划分）
"""

from .gcn import GCNModel
from .rgcn import FaultRGCN
from .tgn import TGN, TGNModel
from .labels import FaultLabelBuilder
from .training import split_masks, train, evaluate, train_rgcn

__all__ = [
    "GCNModel",
    "FaultRGCN",
    "TGN",
    "TGNModel",
    "FaultLabelBuilder",
    "split_masks",
    "train",
    "train_rgcn",
    "evaluate",
]
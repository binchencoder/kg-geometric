"""训练工具函数 —— 数据集划分、GCN 训练与评估。

提供 split_masks, train, evaluate 三个核心训练辅助函数。
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from .gcn import FaultGCN


def split_masks(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将节点索引按 6:2:2 划分为 train/val/test mask。

    Parameters
    ----------
    num_nodes : int
        总节点数。

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (train_mask, val_mask, test_mask)，每个为 bool 张量 [num_nodes]。
    """
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_cut = max(1, int(num_nodes * 0.6))
    val_cut = max(train_cut + 1, int(num_nodes * 0.8))

    train_idx = indices[:train_cut]
    val_idx = indices[train_cut:val_cut]
    test_idx = indices[val_cut:]

    def make_mask(idxs: List[int]) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if idxs:
            mask[idxs] = True
        return mask

    return make_mask(train_idx), make_mask(val_idx), make_mask(test_idx)


def train(model: nn.Module, data, epochs: int = 200, lr: float = 0.01,
          weight_decay: float = 5e-4, log_interval: int = 50) -> None:
    """标准全图 GCN 训练循环。

    Parameters
    ----------
    model : nn.Module
        FaultGCN 模型实例。
    data : Data
        包含 x, edge_index, y, train_mask, val_mask 的图数据。
    epochs : int
        训练轮数。
    lr : float
        学习率。
    weight_decay : float
        L2 正则化权重衰减。
    log_interval : int
        日志输出间隔（轮）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={val_acc:.3f}")


def evaluate(model: nn.Module, data) -> None:
    """在测试集上评估模型并打印分类报告。

    Parameters
    ----------
    model : nn.Module
        已训练的 FaultGCN 模型。
    data : Data
        包含 test_mask 的图数据。
    """
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)

    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = pred[data.test_mask].cpu().numpy()

    print("\nTest accuracy:", accuracy_score(test_true, test_pred))
    print(
        classification_report(
            test_true,
            test_pred,
            target_names=["normal", "fault"],
            zero_division=0,
        )
    )

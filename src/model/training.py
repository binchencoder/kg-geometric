"""训练工具函数 —— 数据集划分、R-GCN 训练与评估。

支持：
- 灵活比例的数据集划分（7:1:2 三元组划分）
- R-GCN 全图训练循环（关系感知）
- 标准 GCN 训练循环（保持向后兼容）
- 测试集评估与分类报告
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .gcn import FaultGCN
from .rgcn import FaultRGCN
from src.core.config import logger


def split_masks(
        num_nodes: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将节点索引按指定比例划分为 train/val/test mask。

    Parameters
    ----------
    num_nodes : int
        总节点数。
    train_ratio : float
        训练集比例，默认 0.7。
    val_ratio : float
        验证集比例，默认 0.1。
    test_ratio : float
        测试集比例，默认 0.2。

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (train_mask, val_mask, test_mask)，每个为 bool 张量 [num_nodes]。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和应为 1.0，当前: {train_ratio}+{val_ratio}+{test_ratio}"

    rng = random.Random(seed)
    indices = list(range(num_nodes))
    rng.shuffle(indices)

    train_cut = max(1, int(num_nodes * train_ratio))
    val_cut = max(train_cut + 1, int(num_nodes * (train_ratio + val_ratio)))

    def make_mask(idxs: List[int]) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if idxs:
            mask[idxs] = True
        return mask

    return (
        make_mask(indices[:train_cut]),
        make_mask(indices[train_cut:val_cut]),
        make_mask(indices[val_cut:]),
    )


def train_rgcn(
        model: FaultRGCN,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        log_interval: int = 30,
        patience: int = 50,
        verbose: bool = True,
) -> dict:
    """R-GCN 全图训练循环（关系感知）。

    核心逻辑：
    1. 使用所有边进行 R-GCN 消息传递（全图结构）
    2. 仅对训练集节点计算分类损失
    3. 在验证集上监控准确率，支持早停

    Parameters
    ----------
    model : FaultRGCN
        R-GCN 模型实例。
    edge_index : torch.Tensor [2, num_edges]
        边索引。
    edge_type : torch.Tensor [num_edges]
        边类型索引。
    y : torch.Tensor [num_nodes]
        节点标签。
    train_mask : torch.Tensor [num_nodes]
        训练节点 mask。
    val_mask : torch.Tensor [num_nodes]
        验证节点 mask。
    epochs : int
        最大训练轮数。
    lr : float
        学习率（默认 1e-3）。
    weight_decay : float
        L2 正则化权重衰减（默认 5e-4）。
    log_interval : int
        日志输出间隔。
    patience : int
        早停耐心值（验证准确率不再提升的最大轮数）。
    verbose : bool
        是否输出详细日志。

    Returns
    -------
    dict
        训练历史: {"epoch": [...], "train_loss": [...], "val_acc": [...]}.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- 训练步骤 ----
        model.train()
        optimizer.zero_grad()
        logits = model(edge_index, edge_type)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # ---- 验证步骤 ----
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(edge_index, edge_type).argmax(dim=1)
                val_correct = (pred[val_mask] == y[val_mask]).sum().item()
                val_total = val_mask.sum().item()
                val_acc = val_correct / max(val_total, 1)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                logger.info(
                    "Epoch %03d | loss=%.4f | val_acc=%.4f",
                    epoch, train_loss, val_acc,
                )

            # ---- 早停检查 ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= patience and patience > 0:
            logger.info("早停触发: val_acc 连续 %d 轮未提升 (best=%.4f)", patience, best_val_acc)
            break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("已恢复最佳模型 (val_acc=%.4f)", best_val_acc)

    return history


def train(
        model: nn.Module,
        data,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        log_interval: int = 50,
) -> None:
    """标准全图 GCN 训练循环（保持向后兼容）。

    优先路由到 R-GCN 训练（如果 model 是 FaultRGCN），
    否则使用标准 GCN 训练逻辑。

    Parameters
    ----------
    model : nn.Module
        FaultGCN 或 FaultRGCN 模型实例。
    data : Data
        包含 x, edge_index, y, train_mask, val_mask 的图数据。
    epochs : int
        训练轮数。
    lr : float
        学习率。
    weight_decay : float
        L2 正则化权重衰减。
    log_interval : int
        日志输出间隔。
    """
    # R-GCN 训练路径
    if isinstance(model, FaultRGCN):
        if not hasattr(data, "edge_type"):
            logger.warning("Data 缺少 edge_type 字段，使用默认值 0")
            data.edge_type = torch.zeros(
                data.edge_index.shape[1], dtype=torch.long,
                device=data.edge_index.device,
            )
        train_rgcn(
            model=model,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            log_interval=log_interval,
        )
        return

    # 标准 GCN 训练路径（原逻辑）
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
                val_bool = pred[data.val_mask] == data.y[data.val_mask]
                val_acc = val_bool.float().mean().item()
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={val_acc:.3f}")


def evaluate(model: nn.Module, data) -> dict:
    """在测试集上评估模型并打印分类报告。

    兼容 FaultGCN 和 FaultRGCN。

    Parameters
    ----------
    model : nn.Module
        已训练的模型（FaultGCN 或 FaultRGCN）。
    data : Data
        包含 test_mask 的图数据。

    Returns
    -------
    dict
        {"accuracy": ..., "f1": ..., "precision": ..., "recall": ...}
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, FaultRGCN):
            edge_type = getattr(data, "edge_type", None)
            pred = model(data.edge_index, edge_type).argmax(dim=1)
        else:
            pred = model(data).argmax(dim=1)

    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = pred[data.test_mask].cpu().numpy()

    acc = float(accuracy_score(test_true, test_pred))
    f1 = float(f1_score(test_true, test_pred, zero_division=0))

    print(f"\n{'='*50}")
    print("  测试集评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'='*50}")
    print(
        classification_report(
            test_true,
            test_pred,
            target_names=["normal", "fault"],
            zero_division=0,
        )
    )

    from sklearn.metrics import precision_score, recall_score
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": float(precision_score(test_true, test_pred, zero_division=0)),
        "recall": float(recall_score(test_true, test_pred, zero_division=0)),
    }

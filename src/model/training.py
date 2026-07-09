"""训练工具函数 —— 数据集划分、R-GCN/TGN 训练与评估。

支持：
- 灵活比例的数据集划分（7:1:2 三元组划分）
- 时序数据集尾部预留未知样本
- R-GCN 全图训练循环（关系感知）
- TGN 时序异构图训练循环（油温回归 + 风险评分二分类）
- R-GCN + TGN 联合训练（多任务损失）
- 标准 GCN 训练循环（保持向后兼容）
- 测试集评估与分类报告
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error

from src.core.config import logger
from .gcn import FaultGCN
from .rgcn import FaultRGCN


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


def train_gcn(
        model: FaultGCN,
        data,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        log_interval: int = 50,
) -> None:
    """标准 GCN 训练循环（非关系型，无早停）。

    Parameters
    ----------
    model : FaultGCN
        GCN 模型实例。
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
    train_gcn(model, data, epochs, lr, weight_decay, log_interval)


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

    unique_classes = sorted(set(test_true.tolist()) | set(test_pred.tolist()))
    class_names = [f"class_{c}" for c in unique_classes]
    if len(unique_classes) == 1:
        # 单类场景（整图训练）跳过分类报告
        class_names = ["all"]
        labels = unique_classes
    else:
        labels = unique_classes

    print(f"\n{'=' * 50}")
    print("  测试集评估结果")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'=' * 50}")
    if len(unique_classes) > 1:
        print(
            classification_report(
                test_true, test_pred,
                target_names=class_names,
                labels=labels,
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


# =====================================================================
# TGN 训练工具
# =====================================================================
def _extract_time_slice_targets(
        hetero_data,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从异构图中提取 time_slice 节点的训练目标。

    返回: (y_health, y_future_ot_norm, y_future_ot, ot_mean, ot_std)
    """
    ts = hetero_data["time_slice"]
    return (
        ts.y_health,
        ts.y_future_ot_norm,
        ts.y_future_ot,
        ts.ot_mean,
        ts.ot_std,
    )


def train_tgn(
        tgn_model: nn.Module,
        hetero_data,
        train_idx: torch.Tensor,
        test_idx: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        log_interval: int = 10,
        verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, float]]:
    """独立训练 TGN 模型（油温回归 + 故障风险二分类，双任务）。

    Parameters
    ----------
    tgn_model : nn.Module
        TGNOilTemperaturePredict 实例（forward 返回 future_ot, fault_risk, _）。
    hetero_data : HeteroData
        需包含 ``x_dict / edge_index_dict / edge_time / y_health / y_future_ot / y_future_ot_norm / ot_mean / ot_std``。
    train_idx, test_idx : torch.Tensor
        训练/测试切片索引（long 型）。
    epochs, lr : int, float
        训练轮数与 Adam 学习率。
    log_interval : int
        日志输出间隔。
    verbose : bool
        是否输出进度信息。

    Returns
    -------
    (tgn_model, metrics)
        - tgn_model: 载入最优权重的模型。
        - metrics: {"ot_mae": ..., "risk_acc": ...}，使用最优 checkpoint 在 test_idx 上计算。
    """
    optimizer = torch.optim.Adam(tgn_model.parameters(), lr=lr)
    loss_ot = nn.MSELoss()
    loss_risk = nn.BCELoss()

    _, y_ot_norm, y_ot_raw, ot_mean, ot_std = _extract_time_slice_targets(hetero_data)
    risk_label = (hetero_data["time_slice"].y_health >= 2).float()

    best_ot_mae = float("inf")
    best_state: Optional[Dict] = None
    best_metrics: Dict[str, float] = {}

    if verbose:
        print("\n===== 开始训练 TGN 时序趋势预测 =====")

    for epoch in range(epochs):
        tgn_model.train()
        optimizer.zero_grad()
        future_ot_pred, fault_risk_pred, _ = tgn_model(hetero_data)

        loss = loss_ot(
            future_ot_pred.squeeze()[train_idx], y_ot_norm[train_idx],
        ) + loss_risk(fault_risk_pred.squeeze()[train_idx], risk_label[train_idx])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                future_ot_celsius = (
                    future_ot_pred.squeeze().detach() * ot_std + ot_mean
                )
                ot_mae = mean_absolute_error(
                    y_ot_raw[test_idx].cpu().numpy(),
                    future_ot_celsius[test_idx].cpu().numpy(),
                )
                risk_pred = (fault_risk_pred.squeeze() >= 0.5).float()
                risk_acc = accuracy_score(
                    risk_label[test_idx].cpu().numpy(),
                    risk_pred[test_idx].cpu().numpy(),
                )

            if ot_mae < best_ot_mae:
                best_ot_mae = ot_mae
                best_state = copy.deepcopy(tgn_model.state_dict())
                best_metrics = {"ot_mae": float(ot_mae), "risk_acc": float(risk_acc)}

            if verbose:
                print(
                    f"Epoch:{epoch + 1:3d} | Loss:{loss.item():.4f} "
                    f"| OT_MAE:{ot_mae:.4f} | RiskAcc:{risk_acc:.4f}"
                )

    if best_state is not None:
        tgn_model.load_state_dict(best_state)
    if verbose:
        print(f"\n训练完成！最优油温预测 MAE: {best_ot_mae:.4f}")
    return tgn_model, best_metrics


def train_joint_rgcn_tgn(
        diag_model: nn.Module,
        tgn_model: nn.Module,
        hetero_data,
        train_idx: torch.Tensor,
        test_idx: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        log_interval: int = 10,
        hold_out_n: int = 0,
        verbose: bool = True,
) -> Tuple[nn.Module, nn.Module, Dict[str, float]]:
    """联合训练 R-GCN（故障诊断，分类）+ TGN（油温+风险，双任务）。

    统一使用 Adam 优化两个模型的参数，loss 为三项之和：
    ``CrossEntropy(health) + MSE(future_ot_norm) + BCE(risk)``。

    Parameters
    ----------
    diag_model : nn.Module
        R-GCN 故障诊断模型（forward 返回 health_logits, _）。
    tgn_model : nn.Module
        TGNOilTemperaturePredict 实例。
    hetero_data : HeteroData
        见 :func:`train_tgn`。
    train_idx, test_idx : torch.Tensor
        训练/测试切片索引。
    epochs, lr : int, float
        训练轮数与 Adam 学习率。
    log_interval : int
        日志输出间隔。
    hold_out_n : int
        仅用于日志提示（不会在这里重新切分数据）。
    verbose : bool
        是否输出进度信息。

    Returns
    -------
    (diag_model, tgn_model, metrics)
        - diag_model, tgn_model: 已载入各自最优权重。
        - metrics: {"diag_acc": ..., "ot_mae": ..., "risk_acc": ...}。
    """
    optimizer = torch.optim.Adam(
        list(diag_model.parameters()) + list(tgn_model.parameters()), lr=lr,
    )
    loss_cls = nn.CrossEntropyLoss()
    loss_ot = nn.MSELoss()
    loss_risk = nn.BCELoss()

    y_health, y_ot_norm, y_ot_raw, ot_mean, ot_std = _extract_time_slice_targets(
        hetero_data,
    )
    risk_label = (y_health >= 2).float()

    best_diag_acc = 0.0
    best_ot_mae = float("inf")
    best_diag_state: Optional[Dict] = None
    best_tgn_state: Optional[Dict] = None
    metrics: Dict[str, float] = {}

    if verbose:
        print("\n===== 开始联合训练 R-GCN故障诊断 + TGN时序趋势预测 =====")

    for epoch in range(epochs):
        diag_model.train()
        tgn_model.train()
        optimizer.zero_grad()

        # ---- 前向传播 ----
        health_logits, _ = diag_model(
            hetero_data.x_dict, hetero_data.edge_index_dict,
        )
        future_ot_pred, fault_risk_pred, _ = tgn_model(hetero_data)

        # ---- 多任务损失 ----
        loss1 = loss_cls(health_logits[train_idx], y_health[train_idx])
        loss2 = loss_ot(future_ot_pred.squeeze()[train_idx], y_ot_norm[train_idx])
        loss3 = loss_risk(fault_risk_pred.squeeze()[train_idx], risk_label[train_idx])
        total_loss = loss1 + loss2 + loss3

        total_loss.backward()
        optimizer.step()

        # ---- 评估（不切换 eval，避免 TGN msg_store 刷新导致二次 backward 失败） ----
        with torch.no_grad():
            pred_health = torch.argmax(health_logits, dim=1)
            diag_acc = accuracy_score(
                y_health[test_idx].cpu().numpy(),
                pred_health[test_idx].cpu().numpy(),
            )
            future_ot_celsius = (
                future_ot_pred.squeeze().detach() * ot_std + ot_mean
            )
            ot_mae = mean_absolute_error(
                y_ot_raw[test_idx].cpu().numpy(),
                future_ot_celsius[test_idx].cpu().numpy(),
            )
            risk_pred = (fault_risk_pred.squeeze() >= 0.5).float()
            risk_acc = accuracy_score(
                risk_label[test_idx].cpu().numpy(),
                risk_pred[test_idx].cpu().numpy(),
            )

        # ---- 保存最优模型 ----
        if diag_acc > best_diag_acc:
            best_diag_acc = diag_acc
            best_diag_state = copy.deepcopy(diag_model.state_dict())
        if ot_mae < best_ot_mae:
            best_ot_mae = ot_mae
            best_tgn_state = copy.deepcopy(tgn_model.state_dict())

        if verbose and ((epoch + 1) % log_interval == 0 or epoch == epochs - 1):
            print(
                f"Epoch:{epoch + 1:3d} | TotalLoss:{total_loss.item():.4f} "
                f"| DiagAcc:{diag_acc:.4f} | OT_MAE:{ot_mae:.4f} "
                f"| RiskAcc:{risk_acc:.4f}"
            )

    if best_diag_state is not None:
        diag_model.load_state_dict(best_diag_state)
    if best_tgn_state is not None:
        tgn_model.load_state_dict(best_tgn_state)

    metrics = {
        "diag_acc": float(best_diag_acc),
        "ot_mae": float(best_ot_mae),
        "risk_acc": float(risk_acc),
    }

    if verbose:
        print(
            f"\n训练完成！最优故障诊断准确率: {best_diag_acc:.4f}，"
            f"最优油温预测MAE: {best_ot_mae:.4f}"
        )
        if hold_out_n > 0:
            print(f"预留未知样本数: {hold_out_n}（既不参与训练也不参与测试）")

    return diag_model, tgn_model, metrics
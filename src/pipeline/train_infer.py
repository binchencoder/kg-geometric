"""训练+推理集成管线 —— 端到端的 ES 数据读取 → 训练 → 评估 → 故障推理。

封装完整的 GCN 训练和 Top-K 静态链接预测流程，支持全图和 NeighborLoader 两种模式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from src.core.config import logger
from src.model.gcn import GCNModel


class KGTrainInferPipeline:
    """端到端：ES 数据读取 → 训练 → 评估 → 故障推理。

    封装完整的 GCN 训练和 Top-K 静态链接预测流程，可与 ES 全图/流式
    两种数据模式配合使用。

    使用示例::

        pipeline = KGTrainInferPipeline()
        pipeline.train_full_graph(data, y, train_mask, val_mask, epochs=200)
        results = pipeline.infer_topk(model, data, symptoms, node_to_idx, fault_nodes)
    """

    def __init__(
            self,
            in_dim: int = 64,
            hidden_dim: int = 32,
            num_classes: int = 2,
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        in_dim : int
            输入特征维度（应与节点嵌入维度一致）。
        hidden_dim : int
            隐藏层维度。
        num_classes : int
            分类类别数（默认 2: 正常/故障）。
        lr : float
            学习率。
        weight_decay : float
            L2 正则化权重衰减。
        device : Optional[str]
            训练设备，None 则自动选择 cuda/cpu。
        """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[GCNModel] = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _init_model(self) -> GCNModel:
        model = GCNModel(in_dim=self.in_dim, hidden_dim=self.hidden_dim)
        if self.num_classes != 2:
            model.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        model.to(self.device)
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        return model

    # ---- 全图模式训练（Data / HeteroData） ----
    def train_full_graph(
            self,
            data: Union[Data, HeteroData],
            y: torch.Tensor,
            train_mask: torch.Tensor,
            val_mask: torch.Tensor,
            epochs: int = 200,
            log_interval: int = 50,
            verbose: bool = True,
    ) -> Dict[str, list]:
        """在全图上训练 GCNModel（使用 train/val mask）。

        Returns
        -------
        Dict[str, list]
            包含 epoch, train_loss, val_acc 的历史记录。
        """
        model = self._init_model()
        data = data.to(self.device)
        y = y.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        history = {"epoch": [], "train_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            self.optimizer.zero_grad()
            logits = model(data)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    pred = model(data).argmax(dim=1)
                    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_acc)
                if verbose:
                    logger.info(
                        "Epoch %03d | loss=%.4f | val_acc=%.4f",
                        epoch, train_loss, val_acc,
                    )

        return history

    # ---- NeighborLoader 模式训练 ----
    def train_with_loader(
            self,
            loader,
            epochs: int = 10,
            log_interval: int = 1,
            verbose: bool = True,
    ) -> Dict[str, list]:
        """使用 PyG NeighborLoader 进行 mini-batch 训练。

        Returns
        -------
        Dict[str, list]
            训练历史记录。
        """
        model = self._init_model()
        history = {"epoch": [], "train_loss": []}

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                logits = model(batch)

                if hasattr(batch, "y") and batch.y is not None:
                    target = batch.y[: logits.shape[0]]
                else:
                    target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / max(batch_count, 1)
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            if verbose and (epoch % log_interval == 0 or epoch == 1 or epoch == epochs):
                logger.info("Epoch %03d | avg_loss=%.4f | batches=%d", epoch, avg_loss, batch_count)

        return history

    # ---- 评估 ----
    @staticmethod
    def evaluate(
            data: Union[Data, HeteroData],
            y: torch.Tensor,
            test_mask: torch.Tensor,
            model: Optional[GCNModel] = None,
    ) -> Dict[str, float]:
        """在测试集上评估模型。

        Returns
        -------
        Dict[str, float]
            {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
        """
        if model is None:
            raise ValueError("model 不能为 None，请先训练或加载模型")

        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        y = y.to(device)
        test_mask = test_mask.to(device)

        with torch.no_grad():
            pred = model(data).argmax(dim=1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        test_true = y[test_mask].cpu().numpy()
        test_pred = pred[test_mask].cpu().numpy()

        return {
            "accuracy": float(accuracy_score(test_true, test_pred)),
            "precision": float(precision_score(test_true, test_pred, zero_division=0)),
            "recall": float(recall_score(test_true, test_pred, zero_division=0)),
            "f1": float(f1_score(test_true, test_pred, zero_division=0)),
        }

    # ---- Top-K 静态链接预测推理 ----
    @staticmethod
    def infer_topk(
            model: GCNModel,
            data: Union[Data, HeteroData],
            symptoms: List[str],
            node_to_idx: Dict[str, int],
            fault_nodes: List[str],
            top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """给定症状节点，基于嵌入余弦相似度推理最可能的故障根因。

        Returns
        -------
        List[Tuple[str, float]]
            按相似度降序排列的 (故障节点, 相似度) 列表。
        """
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            node_emb = model.encode(data)

            symptom_indices = [
                node_to_idx[s] for s in symptoms if s in node_to_idx
            ]
            if not symptom_indices:
                raise ValueError(f"所有症状节点均不在图中: {symptoms}")

            query_emb = node_emb[symptom_indices].mean(dim=0, keepdim=True)

            candidate_indices = [
                node_to_idx[f] for f in fault_nodes if f in node_to_idx
            ]
            if not candidate_indices:
                raise ValueError(f"所有候选故障节点均不在图中: {fault_nodes}")

            candidate_emb = node_emb[candidate_indices]
            valid_fault_names = [
                f for f in fault_nodes if f in node_to_idx
            ]

            query_norm = F.normalize(query_emb, p=2, dim=-1)
            cand_norm = F.normalize(candidate_emb, p=2, dim=-1)
            scores = (cand_norm * query_norm).sum(dim=-1)

            ranked = sorted(
                zip(valid_fault_names, scores.cpu().tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            return ranked[:top_k]

    # ---- 模型保存 / 加载 ----
    def save_model(self, filepath: str) -> None:
        """保存模型参数到磁盘。"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型，请先训练")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
        }, path)
        logger.info("模型已保存至 %s", filepath)

    def load_model(self, filepath: str) -> GCNModel:
        """从磁盘加载模型参数。"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        model = GCNModel(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        if checkpoint.get("num_classes", 2) != 2:
            model.classifier = nn.Linear(checkpoint["hidden_dim"], checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        self.in_dim = checkpoint["in_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_classes = checkpoint.get("num_classes", 2)
        self.model = model
        logger.info("模型已从 %s 加载", filepath)
        return model

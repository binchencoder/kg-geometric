"""知识图谱故障诊断示例：基于症状到故障的 Top-K 推理。

本示例使用 PyTorch Geometric 对一个小型知识图谱进行编码，该图谱由以下三元组构成：

    泵_01 --存在症状--> 振动过高
    泵_01 --原因在于--> 轴承磨损

模型使用 GCN 学习节点嵌入，然后支持两个阶段：
1. 故障节点 vs 正常实体的节点分类。
2. 给定一个或多个症状，按相似度对候选故障节点进行排序。

运行方式：
    python kg_fault_diagnosis.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str


class KGFaultDataset:
    def __init__(self) -> None:
        self.triples: List[Triple] = [
            Triple("泵_01", "存在症状", "振动过高"),
            Triple("泵_01", "存在症状", "温度过高"),
            Triple("泵_01", "原因在于", "轴承磨损"),
            Triple("电机_02", "存在症状", "电流过高"),
            Triple("电机_02", "原因在于", "定子故障"),
            Triple("齿轮箱_03", "存在症状", "噪音异常"),
            Triple("齿轮箱_03", "原因在于", "齿轮磨损"),
            Triple("压缩机_04", "存在症状", "压力过低"),
            Triple("压缩机_04", "原因在于", "阀门泄漏"),
            Triple("轴承磨损", "类型为", "故障"),
            Triple("定子故障", "类型为", "故障"),
            Triple("齿轮磨损", "类型为", "故障"),
            Triple("阀门泄漏", "类型为", "故障"),
        ]

        self.fault_nodes = ["轴承磨损", "定子故障", "齿轮磨损", "阀门泄漏"]
        self.labels: Dict[str, int] = {
            "轴承磨损": 1,
            "定子故障": 1,
            "齿轮磨损": 1,
            "阀门泄漏": 1,
            "故障": 1,
            "泵_01": 0,
            "电机_02": 0,
            "齿轮箱_03": 0,
            "压缩机_04": 0,
            "振动过高": 0,
            "温度过高": 0,
            "电流过高": 0,
            "噪音异常": 0,
            "压力过低": 0,
        }

        self.node_to_idx = self._build_vocab()
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.edge_index = self._build_edge_index()
        self.x = torch.eye(len(self.node_to_idx), dtype=torch.float)
        ordered_nodes = [self.idx_to_node[i] for i in range(len(self.idx_to_node))]
        self.y = torch.tensor([self.labels[node] for node in ordered_nodes], dtype=torch.long)

    def _build_vocab(self) -> Dict[str, int]:
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        return {node: idx for idx, node in enumerate(nodes)}

    def _build_edge_index(self) -> torch.Tensor:
        edges = []
        for triple in self.triples:
            h = self.node_to_idx[triple.head]
            t = self.node_to_idx[triple.tail]
            edges.append([h, t])
            edges.append([t, h])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def to_data(self) -> Data:
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)


class FaultGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _extract(data) -> Tuple[torch.Tensor, torch.Tensor]:
        """兼容 Data 和 HeteroData 两种类型，提取 (x, edge_index)。"""
        if isinstance(data, HeteroData):
            x = data["entity"].x
            edge_index = data["entity", "to", "entity"].edge_index
        else:
            x = data.x
            edge_index = data.edge_index
        return x, edge_index

    def encode(self, data) -> torch.Tensor:
        x, edge_index = self._extract(data)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)

    def forward(self, data) -> torch.Tensor:
        return self.classifier(self.encode(data))


def split_masks(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def train(model: nn.Module, data: Data) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={val_acc:.3f}")


def evaluate(model: nn.Module, data: Data) -> None:
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


def topk_fault_diagnosis(
    model: FaultGCN,
    data: Data,
    dataset: KGFaultDataset,
    symptoms: Sequence[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Rank fault candidates for the given symptoms.

    We average the learned embeddings of symptom nodes, then score every fault
    candidate using cosine similarity.
    """
    model.eval()
    with torch.no_grad():
        node_emb = model.encode(data)
        symptom_indices = [dataset.node_to_idx[s] for s in symptoms if s in dataset.node_to_idx]
        if not symptom_indices:
            raise ValueError(f"None of the symptoms exist in the graph: {symptoms}")

        query_emb = node_emb[symptom_indices].mean(dim=0, keepdim=True)
        candidate_indices = [dataset.node_to_idx[f] for f in dataset.fault_nodes]
        candidate_emb = node_emb[candidate_indices]

        query_norm = F.normalize(query_emb, p=2, dim=-1)
        cand_norm = F.normalize(candidate_emb, p=2, dim=-1)
        scores = (cand_norm * query_norm).sum(dim=-1)

        ranked = sorted(
            zip(dataset.fault_nodes, scores.cpu().tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]


def print_topk_diagnosis(results: List[Tuple[str, float]], symptoms: Sequence[str]) -> None:
    print("\nInput symptoms:", ", ".join(symptoms))
    print("Top-K fault diagnosis results:")
    for rank, (fault, score) in enumerate(results, start=1):
        print(f"  {rank}. {fault:<15} score={score:.4f}")


def main() -> None:
    dataset = KGFaultDataset()
    data = dataset.to_data()
    data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)

    model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
    train(model, data)
    evaluate(model, data)

    example_symptoms = ["振动过高", "温度过高"]
    results = topk_fault_diagnosis(model, data, dataset, example_symptoms, top_k=3)
    print_topk_diagnosis(results, example_symptoms)

    example_symptoms_2 = ["电流过高"]
    results_2 = topk_fault_diagnosis(model, data, dataset, example_symptoms_2, top_k=3)
    print_topk_diagnosis(results_2, example_symptoms_2)


if __name__ == "__main__":
    main()

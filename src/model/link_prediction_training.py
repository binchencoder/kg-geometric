"""链接预测训练与评估工具。

提供：
- train_link_prediction: 标准负采样训练循环（全图模式）
- evaluate_link_prediction: MRR / Hits@K 评估（全图模式）
- predict_top_k: Top-K 链接推理
- train_link_prediction_streaming: 异步子图采样训练循环（流式模式）
- evaluate_link_prediction_streaming: 流式模式下的近似评估
- predict_top_k_streaming: 流式模式下的 Top-K 推理
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.core.config import logger

from .link_prediction import LinkPredictionGCN

if TYPE_CHECKING:
    from src.graph.sampler import AsyncSubgraphSampler
    from src.graph.link_prediction_dataset import LinkPredictionStreamingData

# ---------- 训练 ----------


def train_link_prediction(
    model: LinkPredictionGCN,
    data: Data,
    train_edges: List[Tuple[int, int, int]],
    val_edges: List[Tuple[int, int, int]],
    all_triples_set: set,
    num_nodes: int,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    num_negatives: int = 1,
    log_interval: int = 20,
    device: str = "cpu",
) -> LinkPredictionGCN:
    """链接预测标准训练循环（BCE Loss + 负采样）。

    Parameters
    ----------
    model : LinkPredictionGCN
        链接预测模型实例。
    data : Data
        图数据（仅包含训练边）。
    train_edges : List[Tuple[int, int, int]]
        训练边列表 [(h, r, t), ...]。
    val_edges : List[Tuple[int, int, int]]
        验证边列表。
    all_triples_set : set
        所有已知三元组的集合，用于负采样过滤。
    num_nodes : int
        节点总数。
    epochs : int
        训练轮数。
    lr : float
        学习率。
    weight_decay : float
        L2 正则化系数。
    batch_size : int
        每批正样本数。
    num_negatives : int
        每个正样本的负样本数。
    log_interval : int
        日志输出间隔。
    device : str
        训练设备。

    Returns
    -------
    LinkPredictionGCN
        训练后的模型。
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_edges_array = np.array(train_edges)
    # 避免 batch_size 大于训练边数
    actual_batch_size = min(batch_size, len(train_edges_array))
    best_mrr = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        # 随机采样一个 batch
        indices = np.random.choice(len(train_edges_array), size=actual_batch_size, replace=False)
        pos_batch = train_edges_array[indices]  # [batch_size, 3]

        pos_heads = torch.tensor(pos_batch[:, 0], dtype=torch.long, device=device)
        pos_rels = torch.tensor(pos_batch[:, 1], dtype=torch.long, device=device)
        pos_tails = torch.tensor(pos_batch[:, 2], dtype=torch.long, device=device)

        # 生成负样本：对每个正样本随机替换头或尾
        neg_heads, neg_rels, neg_tails = [], [], []
        for h, r, t in pos_batch:
            for _ in range(num_negatives):
                if np.random.random() < 0.5:
                    neg_h = np.random.randint(0, num_nodes)
                    while (neg_h, r, t) in all_triples_set:
                        neg_h = np.random.randint(0, num_nodes)
                    neg_heads.append(neg_h)
                    neg_rels.append(r)
                    neg_tails.append(t)
                else:
                    neg_t = np.random.randint(0, num_nodes)
                    while (h, r, neg_t) in all_triples_set:
                        neg_t = np.random.randint(0, num_nodes)
                    neg_heads.append(h)
                    neg_rels.append(r)
                    neg_tails.append(neg_t)

        neg_heads_t = torch.tensor(neg_heads, dtype=torch.long, device=device)
        neg_rels_t = torch.tensor(neg_rels, dtype=torch.long, device=device)
        neg_tails_t = torch.tensor(neg_tails, dtype=torch.long, device=device)

        # 合并正负样本
        all_heads = torch.cat([pos_heads, neg_heads_t])
        all_rels = torch.cat([pos_rels, neg_rels_t])
        all_tails = torch.cat([pos_tails, neg_tails_t])
        labels = torch.cat([
            torch.ones(pos_heads.size(0), device=device),
            torch.zeros(neg_heads_t.size(0), device=device),
        ])

        optimizer.zero_grad()
        scores = model(data, all_heads, all_rels, all_tails)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            mrr, hits = _evaluate_mrr(
                model, data, val_edges, all_triples_set, num_nodes, device
            )
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_mrr={mrr:.4f} | val_hits@10={hits[2]:.3f}"
            )

            if mrr > best_mrr:
                best_mrr = mrr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val MRR restored: {best_mrr:.4f}")

    return model


# ---------- 评估 ----------


def evaluate_link_prediction(
    model: LinkPredictionGCN,
    data: Data,
    test_edges: List[Tuple[int, int, int]],
    all_triples_set: set,
    num_nodes: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """在测试集上评估链接预测模型。

    对每个测试三元组 (h, r, t)：
    - 计算"尾实体排名"：(h, r, ?) 对所有候选实体评分，获取 t 的排名
    - 计算"头实体排名"：(?, r, t) 对所有候选实体评分，获取 h 的排名
    - 采用 filtered 设置（排除已知正样本）

    Returns
    -------
    Dict[str, float]
        包含 mrr, hits@1, hits@3, hits@10 的评估指标。
    """
    mrr, hits = _evaluate_mrr(model, data, test_edges, all_triples_set, num_nodes, device)
    return {
        "mrr": mrr,
        "hits@1": hits[0],
        "hits@3": hits[1],
        "hits@10": hits[2],
    }


def _evaluate_mrr(
    model: LinkPredictionGCN,
    data: Data,
    edges: List[Tuple[int, int, int]],
    all_triples_set: set,
    num_nodes: int,
    device: str = "cpu",
) -> Tuple[float, List[float]]:
    """计算 MRR 和 Hits@K 指标。

    对每个三元组分别评估尾实体预测和头实体预测，综合计算排名。
    """
    model.eval()
    ranks = []

    with torch.no_grad():
        node_emb = model.encode(data)  # [N, D]

        for h, r, t in edges:
            # ---- 尾实体预测 (h, r, ?) ----
            head_t = torch.tensor([h], dtype=torch.long, device=device)
            rel_t = torch.tensor([r], dtype=torch.long, device=device)

            # 评分所有候选尾实体
            if model.proj is not None:
                head_for_score = model.proj(node_emb[head_t])  # [1, relation_dim]
                tail_all = model.proj(node_emb)                # [N, relation_dim]
            else:
                head_for_score = node_emb[head_t]
                tail_all = node_emb

            rel_emb = model.relation_emb(rel_t)  # [1, relation_dim]
            scores = torch.mm(
                head_for_score * rel_emb, tail_all.t()
            ).squeeze(0)  # [N]

            # 过滤已知正样本（filtered setting），置为 -inf
            for cand_t in range(num_nodes):
                if cand_t != t and (h, r, cand_t) in all_triples_set:
                    scores[cand_t] = -float("inf")
            # 排除头实体自身（不能预测查询头实体为尾实体）
            scores[h] = -float("inf")

            rank_tail = (scores > scores[t]).sum().item() + 1
            ranks.append(rank_tail)

            # ---- 头实体预测 (?, r, t) ----
            tail_t = torch.tensor([t], dtype=torch.long, device=device)
            if model.proj is not None:
                tail_for_score = model.proj(node_emb[tail_t])
                head_all = model.proj(node_emb)
            else:
                tail_for_score = node_emb[tail_t]
                head_all = node_emb

            scores = torch.mm(
                head_all * rel_emb, tail_for_score.t()
            ).squeeze(1)  # [N]

            for cand_h in range(num_nodes):
                if cand_h != h and (cand_h, r, t) in all_triples_set:
                    scores[cand_h] = -float("inf")
            # 排除尾实体自身（不能预测尾实体为头实体）
            scores[t] = -float("inf")

            rank_head = (scores > scores[h]).sum().item() + 1
            ranks.append(rank_head)

    # 计算指标
    ranks = np.array(ranks, dtype=np.float32)
    mrr = float(np.mean(1.0 / ranks))
    hits_at_1 = float(np.mean(ranks <= 1))
    hits_at_3 = float(np.mean(ranks <= 3))
    hits_at_10 = float(np.mean(ranks <= 10))

    return mrr, [hits_at_1, hits_at_3, hits_at_10]


# ---------- 推理 ----------


def predict_top_k(
    model: LinkPredictionGCN,
    data: Data,
    head_name: str,
    relation_name: str,
    node_to_idx: dict,
    idx_to_node: dict,
    rel_to_idx: dict,
    all_triples_set: set,
    num_nodes: int,
    top_k: int = 5,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """给定头实体和关系，预测 Top-K 最可能的尾实体。

    适用于问答场景：(头实体, 关系, ?) 查询。

    Parameters
    ----------
    model : LinkPredictionGCN
        已训练的链接预测模型。
    data : Data
        图数据。
    head_name : str
        头实体名称。
    relation_name : str
        关系类型名称。
    node_to_idx : dict
        节点名 → 索引 映射。
    idx_to_node : dict
        索引 → 节点名 映射。
    rel_to_idx : dict
        关系名 → 索引 映射。
    all_triples_set : set
        已知三元组集合（用于 filtered 排名）。
    num_nodes : int
        节点总数。
    top_k : int
        返回的结果数。
    device : str
        推理设备。

    Returns
    -------
    List[Tuple[str, float]]
        按评分降序排列的 (候选尾实体名, 评分) 列表。
    """
    if head_name not in node_to_idx:
        raise ValueError(f"Unknown head entity: {head_name}")
    if relation_name not in rel_to_idx:
        raise ValueError(f"Unknown relation: {relation_name}")

    h = node_to_idx[head_name]
    r = rel_to_idx[relation_name]

    model.eval()
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        scores = model.predict_all_tails(
            data,
            torch.tensor([h], dtype=torch.long, device=device),
            torch.tensor([r], dtype=torch.long, device=device),
        ).squeeze(0)  # [N]

    # 过滤已知三元组（filtered setting）
    for cand in range(num_nodes):
        if (h, r, cand) in all_triples_set:
            scores[cand] = -float("inf")

    # 排除头实体自身（不能预测查询头实体为尾实体）
    scores[h] = -float("inf")

    # 取 Top-K
    top_indices = torch.topk(scores, k=min(top_k, num_nodes)).indices.cpu().tolist()
    top_scores = scores[top_indices].cpu().tolist()

    results = [(idx_to_node[idx], score) for idx, score in zip(top_indices, top_scores)]
    return results


def print_link_prediction_results(
    results: List[Tuple[str, float]],
    head_name: str,
    relation_name: str,
) -> None:
    """格式化输出链接预测结果。"""
    print(f"\n查询: ({head_name}, {relation_name}, ?)")
    print("Top-K 预测结果:")
    for rank, (entity, score) in enumerate(results, 1):
        print(f"  {rank}. {entity:<15} score={score:.4f}")


# ═══════════════════════════════════════════════════════════════════
# 流式链接预测训练（异步子图采样）
# ═══════════════════════════════════════════════════════════════════


def _build_subgraph_for_entities(
    sampler: "AsyncSubgraphSampler",
    entities: List[str],
    device: str,
    feature_dim: int,
) -> Optional[tuple]:
    """为一组实体名构建局部子图并返回编码所需的组件。

    Returns
    -------
    Optional[tuple]
        (subgraph_on_device, name_to_local_dict, local_size)
        若子图为空则返回 None。
    """
    subgraph = sampler.sample_subgraph(entities)
    if subgraph is None or subgraph["entity"].num_nodes < 3:
        return None

    L = subgraph["entity"].num_nodes
    local_ids = subgraph.local_entity_ids  # List[str]

    # 随机特征（固定维度），适配 GCN 可变大小子图
    subgraph["entity"].x = torch.randn(L, feature_dim)
    subgraph = subgraph.to(device)

    name_to_local = {name: i for i, name in enumerate(local_ids)}
    return subgraph, name_to_local, L


def _score_triples_local(
    node_emb: torch.Tensor,
    head_indices: torch.Tensor,
    rel_indices: torch.Tensor,
    tail_indices: torch.Tensor,
    model: LinkPredictionGCN,
) -> torch.Tensor:
    """在局部子图上对一批三元组进行 DistMult 评分。

    Parameters
    ----------
    node_emb : [L, hidden_dim]
        局部子图的节点嵌入。
    head_indices / rel_indices / tail_indices : [B]
        头/关系/尾索引（关系索引为全局，头尾为局部）。
    model : LinkPredictionGCN
        模型实例（用于 proj 和 relation_emb）。

    Returns
    -------
    torch.Tensor
        评分 [B]。
    """
    h_emb = node_emb[head_indices]
    t_emb = node_emb[tail_indices]
    if model.proj is not None:
        h_emb = model.proj(h_emb)
        t_emb = model.proj(t_emb)
    r_emb = model.relation_emb(rel_indices)
    return torch.sum(h_emb * r_emb * t_emb, dim=-1)


def train_link_prediction_streaming(
    model: LinkPredictionGCN,
    sampler: "AsyncSubgraphSampler",
    streaming_data: "LinkPredictionStreamingData",
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    num_negatives: int = 1,
    log_interval: int = 20,
    val_batches: int = 5,
    device: str = "cpu",
) -> LinkPredictionGCN:
    """异步子图采样式链接预测训练循环。

    每批训练：
    1. 从训练边中随机采样 B 条正样本
    2. 收集涉及的实体（头+尾），去重后作为种子
    3. AsyncSubgraphSampler 从 ES 动态拉取 k-hop 邻居 → 构建局部子图
    4. GCN 编码子图 → 局部节点嵌入
    5. 在子图内负采样 → DistMult 评分 → BCE Loss

    Parameters
    ----------
    model : LinkPredictionGCN
        链接预测模型（应使用 in_dim=hidden_dim 初始化）。
    sampler : AsyncSubgraphSampler
        异步子图采样器（已配置 streamer + vocab）。
    streaming_data : LinkPredictionStreamingData
        流式数据集（持有 vocab 与边划分）。
    epochs : int
        训练轮数。
    lr : float
        学习率。
    weight_decay : float
        L2 正则化系数。
    batch_size : int
        每批正样本数。
    num_negatives : int
        每个正样本的负样本数。
    log_interval : int
        日志输出间隔。
    val_batches : int
        验证时采样批次数。
    device : str
        训练设备。

    Returns
    -------
    LinkPredictionGCN
        训练后的模型。
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_edges = list(streaming_data.train_edges)
    if len(train_edges) == 0:
        raise ValueError("训练边为空！请检查 ES 连接与过滤参数。")

    actual_batch_size = min(batch_size, len(train_edges))
    best_val_loss = float("inf")
    best_state = None
    rng = np.random.RandomState(42)

    logger.info(
        "流式训练开始: epochs=%d, train_edges=%d, batch_size=%d, num_neg=%d",
        epochs, len(train_edges), actual_batch_size, num_negatives,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_edges)

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_edges), actual_batch_size):
            batch_end = min(batch_start + actual_batch_size, len(train_edges))
            batch_edges = train_edges[batch_start:batch_end]

            # 1. 收集唯一实体
            unique_entities: List[str] = list(
                {e for h, _, t in batch_edges for e in (h, t)}
            )

            # 2. 构建子图
            result = _build_subgraph_for_entities(
                sampler, unique_entities, device, model.hidden_dim,
            )
            if result is None:
                continue
            subgraph, name_to_local, L = result

            # 3. 提取正样本局部索引
            pos_heads, pos_rels, pos_tails = [], [], []
            for h_name, r_name, t_name in batch_edges:
                if h_name in name_to_local and t_name in name_to_local:
                    pos_heads.append(name_to_local[h_name])
                    pos_rels.append(streaming_data.relation_to_idx(r_name))
                    pos_tails.append(name_to_local[t_name])

            if len(pos_heads) < 2:
                continue

            B = len(pos_heads)

            # 4. GCN 编码子图
            node_emb = model.encode(subgraph)  # [L, hidden_dim]

            # 5. 构建子图内已知边集合（用于负采样过滤）
            subgraph_edges: set = set()
            if hasattr(subgraph["entity", "to", "entity"], "edge_index"):
                ei = subgraph["entity", "to", "entity"].edge_index
                for i in range(ei.size(1)):
                    subgraph_edges.add((int(ei[0, i]), int(ei[1, i])))

            # 6. 子图内负采样
            pos_h = torch.tensor(pos_heads, dtype=torch.long, device=device)
            pos_r = torch.tensor(pos_rels, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_tails, dtype=torch.long, device=device)

            neg_h_list, neg_r_list, neg_t_list = [], [], []
            for i in range(B):
                h_loc, r_glob, t_loc = pos_heads[i], pos_rels[i], pos_tails[i]
                for _ in range(num_negatives):
                    if rng.random() < 0.5:
                        neg_h = rng.randint(0, L - 1)
                        attempts = 0
                        while (neg_h, t_loc) in subgraph_edges and attempts < 20:
                            neg_h = rng.randint(0, L - 1)
                            attempts += 1
                        neg_h_list.append(neg_h)
                        neg_r_list.append(r_glob)
                        neg_t_list.append(t_loc)
                    else:
                        neg_t = rng.randint(0, L - 1)
                        attempts = 0
                        while (h_loc, neg_t) in subgraph_edges and attempts < 20:
                            neg_t = rng.randint(0, L - 1)
                            attempts += 1
                        neg_h_list.append(h_loc)
                        neg_r_list.append(r_glob)
                        neg_t_list.append(neg_t)

            neg_h = torch.tensor(neg_h_list, dtype=torch.long, device=device)
            neg_r = torch.tensor(neg_r_list, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_t_list, dtype=torch.long, device=device)

            # 7. 评分
            pos_scores = _score_triples_local(node_emb, pos_h, pos_r, pos_t, model)
            neg_scores = _score_triples_local(node_emb, neg_h, neg_r, neg_t, model)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B * num_negatives, device=device),
            ])

            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # ── 验证 ──
        if epoch % log_interval == 0:
            avg_loss = total_loss / max(n_batches, 1)
            val_loss = _evaluate_val_streaming(
                model, sampler, streaming_data,
                batch_size=actual_batch_size,
                num_negatives=num_negatives,
                num_batches=val_batches,
                device=device,
            )
            logger.info(
                "Epoch %03d | loss=%.4f | val_loss=%.4f | entities=%d",
                epoch, avg_loss, val_loss, streaming_data.num_entities,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Best val loss restored: %.4f", best_val_loss)

    return model


def _evaluate_val_streaming(
    model: LinkPredictionGCN,
    sampler: "AsyncSubgraphSampler",
    streaming_data: "LinkPredictionStreamingData",
    batch_size: int = 64,
    num_negatives: int = 1,
    num_batches: int = 5,
    device: str = "cpu",
) -> float:
    """子图采样模式下的验证损失计算。"""
    model.eval()
    val_edges = list(streaming_data.val_edges)
    if not val_edges:
        return 0.0

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    count = 0
    rng = np.random.RandomState(123)

    with torch.no_grad():
        for _ in range(num_batches):
            if len(val_edges) < batch_size:
                indices = rng.choice(len(val_edges), size=min(len(val_edges), batch_size), replace=True)
            else:
                indices = rng.choice(len(val_edges), size=batch_size, replace=False)
            batch_edges = [val_edges[i] for i in indices]

            unique_entities = list(
                {e for h, _, t in batch_edges for e in (h, t)}
            )

            result = _build_subgraph_for_entities(
                sampler, unique_entities, device, model.hidden_dim,
            )
            if result is None:
                continue
            subgraph, name_to_local, L = result

            pos_heads, pos_rels, pos_tails = [], [], []
            for h_name, r_name, t_name in batch_edges:
                if h_name in name_to_local and t_name in name_to_local:
                    pos_heads.append(name_to_local[h_name])
                    pos_rels.append(streaming_data.relation_to_idx(r_name))
                    pos_tails.append(name_to_local[t_name])

            if len(pos_heads) < 2:
                continue

            B = len(pos_heads)
            node_emb = model.encode(subgraph)

            subgraph_edges: set = set()
            if hasattr(subgraph["entity", "to", "entity"], "edge_index"):
                ei = subgraph["entity", "to", "entity"].edge_index
                for i in range(ei.size(1)):
                    subgraph_edges.add((int(ei[0, i]), int(ei[1, i])))

            pos_h = torch.tensor(pos_heads, dtype=torch.long, device=device)
            pos_r = torch.tensor(pos_rels, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_tails, dtype=torch.long, device=device)

            neg_h_list, neg_r_list, neg_t_list = [], [], []
            for i in range(B):
                h_loc, r_glob, t_loc = pos_heads[i], pos_rels[i], pos_tails[i]
                for _ in range(num_negatives):
                    if rng.random() < 0.5:
                        neg_h_val = rng.randint(0, L - 1)
                        neg_h_list.append(neg_h_val)
                        neg_r_list.append(r_glob)
                        neg_t_list.append(t_loc)
                    else:
                        neg_t_val = rng.randint(0, L - 1)
                        neg_h_list.append(h_loc)
                        neg_r_list.append(r_glob)
                        neg_t_list.append(neg_t_val)

            neg_h = torch.tensor(neg_h_list, dtype=torch.long, device=device)
            neg_r = torch.tensor(neg_r_list, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_t_list, dtype=torch.long, device=device)

            pos_scores = _score_triples_local(node_emb, pos_h, pos_r, pos_t, model)
            neg_scores = _score_triples_local(node_emb, neg_h, neg_r, neg_t, model)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B * num_negatives, device=device),
            ])
            loss = criterion(scores, labels)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def evaluate_link_prediction_streaming(
    model: LinkPredictionGCN,
    sampler: "AsyncSubgraphSampler",
    streaming_data: "LinkPredictionStreamingData",
    batch_size: int = 64,
    num_negatives: int = 1,
    num_batches: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """流式模式下的近似评估 —— 在子图内计算 Hit@K。

    对于海量知识图谱，无法计算全局 MRR（需要全实体评分），
    改为在随机采样的子图内计算近似 Hits@1/3/10。

    Returns
    -------
    Dict[str, float]
        val_loss, approx_hits@1, approx_hits@3, approx_hits@10。
    """
    model.eval()
    test_edges = list(streaming_data.test_edges)
    if not test_edges:
        return {"val_loss": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_hits = [0, 0, 0]  # @1, @3, @10
    total_queries = 0
    count = 0
    rng = np.random.RandomState(456)

    with torch.no_grad():
        for _ in range(num_batches):
            if len(test_edges) < batch_size:
                indices = rng.choice(len(test_edges), size=min(len(test_edges), batch_size), replace=True)
            else:
                indices = rng.choice(len(test_edges), size=batch_size, replace=False)
            batch_edges = [test_edges[i] for i in indices]

            unique_entities = list(
                {e for h, _, t in batch_edges for e in (h, t)}
            )

            result = _build_subgraph_for_entities(
                sampler, unique_entities, device, model.hidden_dim,
            )
            if result is None:
                continue
            subgraph, name_to_local, L = result

            pos_heads, pos_rels, pos_tails = [], [], []
            for h_name, r_name, t_name in batch_edges:
                if h_name in name_to_local and t_name in name_to_local:
                    pos_heads.append(name_to_local[h_name])
                    pos_rels.append(streaming_data.relation_to_idx(r_name))
                    pos_tails.append(name_to_local[t_name])

            if len(pos_heads) < 2:
                continue

            B = len(pos_heads)
            node_emb = model.encode(subgraph)

            # 负采样 + 损失
            subgraph_edges: set = set()
            if hasattr(subgraph["entity", "to", "entity"], "edge_index"):
                ei = subgraph["entity", "to", "entity"].edge_index
                for i in range(ei.size(1)):
                    subgraph_edges.add((int(ei[0, i]), int(ei[1, i])))

            pos_h = torch.tensor(pos_heads, dtype=torch.long, device=device)
            pos_r = torch.tensor(pos_rels, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_tails, dtype=torch.long, device=device)

            neg_h_list, neg_r_list, neg_t_list = [], [], []
            for i in range(B):
                h_loc, r_glob, t_loc = pos_heads[i], pos_rels[i], pos_tails[i]
                for _ in range(num_negatives):
                    neg_h_val = rng.randint(0, L - 1)
                    neg_h_list.append(neg_h_val)
                    neg_r_list.append(r_glob)
                    neg_t_list.append(t_loc)

            neg_h = torch.tensor(neg_h_list, dtype=torch.long, device=device)
            neg_r = torch.tensor(neg_r_list, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_t_list, dtype=torch.long, device=device)

            pos_scores = _score_triples_local(node_emb, pos_h, pos_r, pos_t, model)
            neg_scores = _score_triples_local(node_emb, neg_h, neg_r, neg_t, model)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B * num_negatives, device=device),
            ])
            total_loss += criterion(scores, labels).item()

            # ── 子图内近似 Hits@K ──
            # 对每个正样本 (h, r, t)，在子图内对所有尾实体评分，看 t 的排名
            for i in range(B):
                h_loc = pos_heads[i]
                r_glob = pos_rels[i]
                true_t = pos_tails[i]

                h_emb = node_emb[h_loc:h_loc + 1]
                if model.proj is not None:
                    h_emb = model.proj(h_emb)
                    all_t_emb = model.proj(node_emb)
                else:
                    all_t_emb = node_emb
                r_emb = model.relation_emb(torch.tensor([r_glob], device=device))
                all_scores = torch.mm(h_emb * r_emb, all_t_emb.t()).squeeze(0)  # [L]

                # 过滤子图内已知边
                for cand in range(L):
                    if cand != true_t and (h_loc, cand) in subgraph_edges:
                        all_scores[cand] = -float("inf")

                rank = (all_scores > all_scores[true_t]).sum().item() + 1
                for k_idx, k in enumerate([1, 3, 10]):
                    if rank <= k:
                        total_hits[k_idx] += 1
                total_queries += 1

            count += 1

    return {
        "val_loss": total_loss / max(count, 1),
        "hits@1": total_hits[0] / max(total_queries, 1),
        "hits@3": total_hits[1] / max(total_queries, 1),
        "hits@10": total_hits[2] / max(total_queries, 1),
    }


def predict_top_k_streaming(
    model: LinkPredictionGCN,
    sampler: "AsyncSubgraphSampler",
    head_name: str,
    relation_name: str,
    streaming_data: "LinkPredictionStreamingData",
    top_k: int = 5,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """流式模式下的 Top-K 链接推理。

    构建包含 head 实体的子图，在子图内对所有实体评分并排序。

    Parameters
    ----------
    model : LinkPredictionGCN
        已训练的链接预测模型。
    sampler : AsyncSubgraphSampler
        子图采样器。
    head_name : str
        头实体名称。
    relation_name : str
        关系类型名称。
    streaming_data : LinkPredictionStreamingData
        流式数据集。
    top_k : int
        返回的候选数。
    device : str
        推理设备。

    Returns
    -------
    List[Tuple[str, float]]
        (候选实体名, 评分) 列表。
    """
    if head_name not in streaming_data.vocab.entity2idx:
        raise ValueError(f"Unknown head entity: {head_name}")
    if relation_name not in streaming_data.vocab.relation2idx:
        raise ValueError(f"Unknown relation: {relation_name}")

    r_glob = streaming_data.relation_to_idx(relation_name)

    # 构建以 head 实体为中心的子图
    result = _build_subgraph_for_entities(
        sampler, [head_name], device, model.hidden_dim,
    )
    if result is None:
        logger.warning("子图为空，无法进行推理")
        return []

    subgraph, name_to_local, L = result
    local_ids = subgraph.local_entity_ids  # List[str]

    if head_name not in name_to_local:
        raise ValueError(f"Head entity '{head_name}' 不在子图中")

    h_loc = name_to_local[head_name]

    model.eval()
    with torch.no_grad():
        node_emb = model.encode(subgraph)  # [L, hidden_dim]

        h_emb = node_emb[h_loc:h_loc + 1]  # [1, D]
        if model.proj is not None:
            h_emb = model.proj(h_emb)
            all_t_emb = model.proj(node_emb)
        else:
            all_t_emb = node_emb

        r_emb = model.relation_emb(torch.tensor([r_glob], device=device))
        scores = torch.mm(h_emb * r_emb, all_t_emb.t()).squeeze(0)  # [L]

    # ── 过滤 ──
    # 1. 排除头实体自身（不能预测查询头实体为尾实体）
    scores[h_loc] = -float("inf")

    # 2. 过滤子图内与 h_loc 有直接边的已知实体（避免把已有连边当预测）
    if hasattr(subgraph["entity", "to", "entity"], "edge_index"):
        ei = subgraph["entity", "to", "entity"].edge_index
        connected_to_head = set()
        for i in range(ei.size(1)):
            u, v = int(ei[0, i]), int(ei[1, i])
            if u == h_loc:
                connected_to_head.add(v)
            elif v == h_loc:
                connected_to_head.add(u)
        for cand in connected_to_head:
            scores[cand] = -float("inf")

    # 取 Top-K
    k = min(top_k, L)
    top_indices = torch.topk(scores, k=k).indices.cpu().tolist()
    top_scores = scores[top_indices].cpu().tolist()

    results = [(local_ids[idx], score) for idx, score in zip(top_indices, top_scores)]
    return results

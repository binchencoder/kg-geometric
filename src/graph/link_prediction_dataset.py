"""链接预测数据集 —— 将知识图谱三元组转换为链接预测训练/验证/测试数据。

包含：
- LinkPredictionData: 全量加载模式 —— 初始化时对边进行 train/val/test 划分
- LinkPredictionStreamingData: 流式模式 —— 适用于海量三元组，边查边训
- 负采样：随机替换头实体或尾实体生成负样本
- 评估用数据：为每个测试三元组生成 (h,r,?) 和 (?,r,t) 候选集
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data

from src.core.config import logger
from src.core.types import Triple
from src.es.vocabulary import KGVocabulary

if TYPE_CHECKING:
    from src.es.streamer import ESTripletStreamer


class LinkPredictionData:
    """将三元组列表转换为链接预测所需的训练/验证/测试数据。

    工作流程：
    1. 构建节点/关系词汇表
    2. 按 train:val:test = 0.8:0.1:0.1 随机划分边
    3. 生成 PyG Data（用于 GCN 编码器，仅使用训练边构建图）
    4. 提供负采样和评估工具方法
    """

    def __init__(
        self,
        triples: List[Triple],
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        triples : List[Triple]
            知识图谱三元组列表。
        split_ratios : Tuple[float, float, float]
            (train_ratio, val_ratio, test_ratio)，三项之和应为 1.0。
        seed : int
            随机种子，确保划分可复现。
        """
        self.triples = triples
        self.split_ratios = split_ratios
        self.seed = seed

        # 构建词汇表
        self.node_to_idx, self.idx_to_node = self._build_node_vocab()
        self.rel_to_idx, self.idx_to_rel = self._build_rel_vocab()
        self.num_nodes = len(self.node_to_idx)
        self.num_relations = len(self.rel_to_idx)

        # 划分边集
        self.train_edges, self.val_edges, self.test_edges = self._split_edges()

        # 构建图（仅使用训练边）
        self.data = self._build_graph(self.train_edges)

        # 缓存已知三元组集合（用于负采样时排除正样本）
        self._all_triples_set: Set[Tuple[int, int, int]] = set()
        for t in triples:
            self._all_triples_set.add(
                (self.node_to_idx[t.head], self.rel_to_idx[t.relation], self.node_to_idx[t.tail])
            )

    def _build_node_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """构建节点名称 <-> 索引的双向映射。"""
        nodes = sorted({t.head for t in self.triples} | {t.tail for t in self.triples})
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        return node_to_idx, idx_to_node

    def _build_rel_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """构建关系名称 <-> 索引的双向映射。"""
        rels = sorted({t.relation for t in self.triples})
        rel_to_idx = {rel: idx for idx, rel in enumerate(rels)}
        idx_to_rel = {idx: rel for rel, idx in rel_to_idx.items()}
        return rel_to_idx, idx_to_rel

    def _split_edges(
        self,
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """按比例随机划分边为 train/val/test。

        策略：按照 paper "Convolutional 2D Knowledge Graph Embeddings" 的做法，
        对所有边进行随机 shuffle 后按比例切分。
        """
        rng = random.Random(self.seed)
        all_edges = [
            (self.node_to_idx[t.head], self.rel_to_idx[t.relation], self.node_to_idx[t.tail])
            for t in self.triples
        ]
        rng.shuffle(all_edges)

        total = len(all_edges)
        train_end = max(1, int(total * self.split_ratios[0]))
        val_end = max(train_end + 1, int(total * (self.split_ratios[0] + self.split_ratios[1])))

        train = all_edges[:train_end]
        val = all_edges[train_end:val_end]
        test = all_edges[val_end:]

        return train, val, test

    def _build_graph(self, edges: List[Tuple[int, int, int]]) -> Data:
        """使用指定边集构建 PyG Data（无向图，one-hot 特征）。"""
        # 构建无向边索引（忽略边类型，GCN仅用结构信息）
        edge_list = []
        for h, _, t in edges:
            edge_list.append([h, t])
            edge_list.append([t, h])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # one-hot 特征：单位矩阵
        x = torch.eye(self.num_nodes, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def negative_sample(
        self,
        batch_size: int,
        num_negatives: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从训练集中采样一批正样本并生成对应的负样本。

        负采样策略：对每个正样本，随机替换头实体或尾实体（各50%概率），
        确保生成的负样本不在已知三元组中。

        Parameters
        ----------
        batch_size : int
            正样本数量。
        num_negatives : int
            每个正样本生成的负样本数量。

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            (heads, rels, tails, labels)
            - heads: [batch_size * (1 + num_negatives)]
            - rels:  [batch_size * (1 + num_negatives)]
            - tails: [batch_size * (1 + num_negatives)]
            - labels: [batch_size * (1 + num_negatives)], 1.0 为正样本，0.0 为负样本
        """
        # 从训练边中随机采样
        indices = np.random.choice(len(self.train_edges), size=batch_size, replace=False)
        pos_heads, pos_rels, pos_tails = [], [], []

        for idx in indices:
            h, r, t = self.train_edges[idx]
            pos_heads.append(h)
            pos_rels.append(r)
            pos_tails.append(t)

        # 构建负样本
        neg_heads, neg_rels, neg_tails = [], [], []
        for i in range(batch_size):
            for _ in range(num_negatives):
                h, r, t = pos_heads[i], pos_rels[i], pos_tails[i]
                # 50% 概率替换头实体，50% 替换尾实体
                if random.random() < 0.5:
                    # 替换头实体
                    neg_h = random.randint(0, self.num_nodes - 1)
                    while (neg_h, r, t) in self._all_triples_set:
                        neg_h = random.randint(0, self.num_nodes - 1)
                    neg_heads.append(neg_h)
                    neg_rels.append(r)
                    neg_tails.append(t)
                else:
                    # 替换尾实体
                    neg_t = random.randint(0, self.num_nodes - 1)
                    while (h, r, neg_t) in self._all_triples_set:
                        neg_t = random.randint(0, self.num_nodes - 1)
                    neg_heads.append(h)
                    neg_rels.append(r)
                    neg_tails.append(neg_t)

        # 合并正负样本
        all_heads = pos_heads + neg_heads
        all_rels = pos_rels + neg_rels
        all_tails = pos_tails + neg_tails
        labels = [1.0] * (batch_size) + [0.0] * (batch_size * num_negatives)

        return (
            torch.tensor(all_heads, dtype=torch.long),
            torch.tensor(all_rels, dtype=torch.long),
            torch.tensor(all_tails, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )

    def get_eval_triples(
        self,
        split: str = "test",
    ) -> List[Tuple[int, int, int]]:
        """获取评估用的三元组列表。

        Parameters
        ----------
        split : str
            "val" 或 "test"。

        Returns
        -------
        List[Tuple[int, int, int]]
            (head_idx, rel_idx, tail_idx) 三元组列表。
        """
        if split == "val":
            return self.val_edges
        elif split == "test":
            return self.test_edges
        else:
            raise ValueError(f"Unknown split: {split}")

    def statistics(self) -> dict:
        """返回数据集统计信息。"""
        return {
            "总节点数": self.num_nodes,
            "关系类型数": self.num_relations,
            "三元组总数": len(self.triples),
            "训练边数": len(self.train_edges),
            "验证边数": len(self.val_edges),
            "测试边数": len(self.test_edges),
            "节点列表": list(self.node_to_idx.keys()),
            "关系列表": list(self.rel_to_idx.keys()),
        }


# ═══════════════════════════════════════════════════════════════════
# 流式链接预测数据（适用于海量三元组，不加载全量到内存）
# ═══════════════════════════════════════════════════════════════════


class LinkPredictionStreamingData:
    """流式链接预测数据集 —— 适用于海量三元组（上亿级）的"边查边训"模式。

    与 LinkPredictionData 的区别：
    - 不将全量三元组加载到内存
    - 通过一次 ES 流式扫描完成词汇表构建 + 边集划分
    - 训练时由 AsyncSubgraphSampler 按需从 ES 拉取局部子图

    工作流程：
    1. build_from_streamer(): 流式扫描 ES，构建 vocab，MD5 哈希划分边集
    2. 产出 train/val/test 边引用列表（存实体名称而非索引，供采样器使用）
    3. 训练时：sampler 按种子实体查 ES → 构建子图 → 本地索引映射 → 训练
    """

    def __init__(self):
        self.vocab = KGVocabulary()
        self.train_edges: List[Tuple[str, str, str]] = []   # (h_name, r_name, t_name)
        self.val_edges: List[Tuple[str, str, str]] = []
        self.test_edges: List[Tuple[str, str, str]] = []
        self._total_triples = 0

    # ── 属性 ──
    @property
    def num_entities(self) -> int:
        return self.vocab.num_entities

    @property
    def num_relations(self) -> int:
        return self.vocab.num_relations

    def entity_to_idx(self, name: str) -> int:
        return self.vocab.entity2idx.get(name, -1)

    def relation_to_idx(self, name: str) -> int:
        return self.vocab.relation2idx.get(name, -1)

    def idx_to_entity(self, idx: int) -> str:
        return self.vocab.idx2entity.get(idx, "?")

    def idx_to_relation(self, idx: int) -> str:
        return self.vocab.idx2relation.get(idx, "?")

    # ── 核心：从 ES 流式构建 ──

    def build_from_streamer(
        self,
        streamer: "ESTripletStreamer",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        head_field: str = "head_id",
        relation_field: str = "relation",
        tail_field: str = "tail_id",
        extra_filters: Optional[dict] = None,
        max_edges: Optional[int] = None,
    ) -> int:
        """从 ESTripletStreamer 流式构建词汇表并划分边集。

        使用 MD5 哈希做确定性分桶，不需要两次扫描。

        Parameters
        ----------
        streamer : ESTripletStreamer
            search_after 流式提取器。
        split_ratios : Tuple[float, float, float]
            (train, val, test) 比例，和应为 1.0。
        seed : int
            哈希种子，保证划分可复现。
        head_field / relation_field / tail_field : str
            ES 索引中的字段名。
        extra_filters : Optional[dict]
            额外 ES 过滤条件。
        max_edges : Optional[int]
            最大边数限制（调试用），None 表示不限制。

        Returns
        -------
        int
            总三元组数。
        """
        self.train_edges.clear()
        self.val_edges.clear()
        self.test_edges.clear()
        total = 0

        for batch in streamer.stream_triplets(
            head_field=head_field,
            relation_field=relation_field,
            tail_field=tail_field,
            extra_filters=extra_filters,
            resume=False,
        ):
            for t in batch:
                h, r, tail = t["head"], t["relation"], t["tail"]

                # 词汇表
                self.vocab.add_entity(h)
                self.vocab.add_entity(tail)
                self.vocab.add_relation(r)

                # MD5 哈希分桶
                key = f"{h}|{r}|{tail}|{seed}"
                bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000 / 10000.0
                edge = (h, r, tail)

                if bucket < split_ratios[0]:
                    self.train_edges.append(edge)
                elif bucket < split_ratios[0] + split_ratios[1]:
                    self.val_edges.append(edge)
                else:
                    self.test_edges.append(edge)

                total += 1
                if max_edges and total >= max_edges:
                    break

            if total % 100000 == 0 and total > 0:
                logger.info(
                    "流式数据构建中... 三元组=%d, 实体=%d, 关系=%d",
                    total, self.num_entities, self.num_relations,
                )

            if max_edges and total >= max_edges:
                break

        logger.info(
            "流式数据构建完成: 三元组=%d (train=%d/val=%d/test=%d), 实体=%d, 关系=%d",
            total,
            len(self.train_edges), len(self.val_edges), len(self.test_edges),
            self.num_entities, self.num_relations,
        )
        self._total_triples = total
        self.vocab._warn_if_names_look_like_ids()
        return total

    # ── 统计 ──

    def statistics(self) -> dict:
        return {
            "总三元组数": self._total_triples,
            "实体数": self.num_entities,
            "关系类型数": self.num_relations,
            "训练边": self.train_edges,
            "训练边总数": len(self.train_edges),
            "验证边": self.val_edges,
            "验证边总数": len(self.val_edges),
            "测试边": self.test_edges,
            "测试边总数": len(self.test_edges),
        }

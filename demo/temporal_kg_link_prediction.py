# -*- coding: utf-8 -*-
"""
demo/temporal_kg_link_prediction.py
=====================================================================
电力变压器「动态关系时序知识图谱」—— 时序链接预测 (Temporal KG Link Prediction)
=====================================================================

本文件【完全独立、不依赖本项目任何其它文件】，仅依赖 torch / numpy，可直接运行：

    python demo/temporal_kg_link_prediction.py

---------------------------------------------------------------------
一、要解决的问题（与你的场景对应）
---------------------------------------------------------------------
1. 本体 + 动态图谱：用代码动态生成一批电力变压器领域的「关系时序知识图谱」，
   实体会随时间不断加入（新变压器、新传感器），每条关系(边)都带一个连续时间戳 τ
   —— 即你说的「时间作为关系属性」。
2. 归纳式 (inductive)：新加入的实体（之前从没出现过）也要能被预测。
3. 任务：时序链接预测 —— 给定历史四元组 (头实体, 关系, 尾实体, 时间)，
   预测「在某个未来时间，某实体通过某关系会连到哪个实体」。

---------------------------------------------------------------------
二、技术选型（基于此前讨论的综合结论）
---------------------------------------------------------------------
* 连续时间 (continuous-time)：时间戳 τ 是连续值，随边存储，不做离散快照。
* 归纳式编码：实体表示 = 时序邻居 + 关系 + 时间 的「动态聚合」结果，
  而不是 nn.Embedding 静态查表，所以新实体只要有边就能被算出表示。
* 编码器：TGAT 风格的「时序图注意力」。这与 PyG-Temporal 库里的 TGAT / TGN
  是【同一思想】（时间编码 + 时序邻居注意力）。这里为了「不依赖项目文件、
  且能直接读懂/运行」，用纯 torch 自包含实现；如果你已 pip 安装
  torch_geometric_temporal，可直接把本文件的 TemporalKGModel 编码器替换为
  官方 TGAT / TGN 模块，解码器部分仍需自己补（PyG-Temporal 不含 KG 打分解码器）。
* 解码器：时间感知的 DistMult 打分。因为实体表示随查询时间 τ 变化，所以同样的
  (头, 关系, 尾) 在不同时间的得分不同 —— 这就是「时间感知」。
* 推理：对某个 (头, 关系, 时间)，对所有「合法候选尾实体」算分并排序取 Top-K。

---------------------------------------------------------------------
三、文件结构（自上而下阅读顺序）
---------------------------------------------------------------------
  1) 本体定义 (ENTITY_TYPES / RELATIONS)
  2) 数据动态生成 (generate_dataset)  —— 含新实体随时间加入
  3) 时间编码器 (TimeEncoder, Time2Vec 风格)
  4) 模型 (TemporalKGModel: TGAT 编码器 + DistMult 解码器)
  5) 训练 (train) / 负采样 (sample_negative)
  6) 评估与推理 (evaluate / predict_tails)
  7) 主函数 (main) —— 生成数据 -> 训练 -> 评估(含归纳式) -> 示例预测
"""

import math
import argparse
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 1. 本体 (Ontology)
# ====================================================================
# 实体类型：变电站 / 变压器 / 各部件 / 传感器 / 故障类型
ENTITY_TYPES = [
    "变电站",       # 变电站
    "变压器",       # 主变压器
    "绕组",         # 绕组
    "套管",         # 套管
    "铁芯",         # 铁芯
    "绝缘油",       # 绝缘油
    "冷却系统",     # 冷却系统
    "有载分接开关", # 有载分接开关
    "传感器",       # 传感器(测温/油位/溶解气体 DGA)
    "故障类型",     # 故障类型
]

# 关系类型：(关系名, 头实体类型, 尾实体类型, 是否对称)
# —— 这些就是图谱里的「关系」；每条关系实例都带一个时间戳 τ。
RELATIONS = [
    ("位于",     "变压器", "变电站", False),  # 变压器位于某变电站
    ("包含部件", "变压器", "绕组",   False),  # 变压器包含绕组
    ("包含部件", "变压器", "套管",   False),  # 变压器包含套管
    ("包含部件", "变压器", "铁芯",   False),  # 变压器包含铁芯
    ("包含部件", "变压器", "绝缘油", False),  # 变压器包含绝缘油
    ("包含部件", "变压器", "冷却系统", False),  # 变压器包含冷却系统
    ("包含部件", "变压器", "有载分接开关", False),  # 变压器包含分接开关
    ("被监测",   "绕组",   "传感器", False),  # 绕组被传感器监测
    ("被监测",   "套管",   "传感器", False),  # 套管被传感器监测
    ("被监测",   "绝缘油", "传感器", False),  # 绝缘油被传感器监测
    ("被监测",   "铁芯",   "传感器", False),  # 铁芯被传感器监测
    ("导致",     "故障类型", "绕组",   False),  # 故障导致绕组劣化
    ("导致",     "故障类型", "套管",   False),  # 故障导致套管劣化
    ("导致",     "故障类型", "绝缘油", False),  # 故障导致绝缘油劣化
    ("导致",     "故障类型", "铁芯",   False),  # 故障导致铁芯劣化
    ("互联",     "变压器", "变压器", True),  # 同站变压器互联(对称)
]


def build_relation_vocab():
    """把关系名映射成 id，并为非对称关系自动构造反向关系(用于双向聚合)。

    返回:
        rel2id      : 关系名 -> id
        id2rel      : id -> 关系名
        inv         : 关系id -> 其反向关系id (对称关系指向自己)
        rel_head_types / rel_tail_types : 关系id -> 允许的(头/尾)实体类型集合
                     (用于负采样与推理时缩小候选尾实体范围，更贴近真实 KG)
    """
    rel2id, id2rel, inv = {}, [], {}
    rel_head_types, rel_tail_types = defaultdict(set), defaultdict(set)
    for name, h, t, sym in RELATIONS:
        if name not in rel2id:
            rid = len(id2rel)
            rel2id[name] = rid
            id2rel.append(name)
        else:
            rid = rel2id[name]
        rel_head_types[rid].add(h)
        rel_tail_types[rid].add(t)
        if sym:
            inv[rid] = rid
        else:
            inv_name = name + "_反向"
            if inv_name not in rel2id:
                ivid = len(id2rel)
                rel2id[inv_name] = ivid
                id2rel.append(inv_name)
            else:
                ivid = rel2id[inv_name]
            inv[rid] = ivid
            inv[ivid] = rid
            # 反向关系的头/尾类型互换
            rel_head_types[ivid].add(t)
            rel_tail_types[ivid].add(h)
    return rel2id, id2rel, inv, dict(rel_head_types), dict(rel_tail_types)


# ====================================================================
# 2. 数据动态生成
# ====================================================================
@dataclass
class Entity:
    eid: int
    etype: str
    created_at: float   # 实体加入图谱的时间(连续时间戳) —— 新实体=较晚的时间
    name: str


@dataclass
class Quad:
    """一条有向的「时间戳关系」(timed relation)。

    关键设计：时间 τ 是【关系本身的一个属性】，而不是游离在三元组外的第 4 个坐标。
    即 (头, 关系, 尾) 中的「关系」是一个带时间属性的实例：relation 在 time 时刻成立。
    这样模型里「关系」的表示会随 time 变化 —— 见 TemporalKGModel.rel_repr。
    """
    head: int
    relation: int      # 关系 id
    tail: int
    time: float        # 该关系实例发生的连续时间戳 —— 即「关系的时间属性」


def generate_dataset(seed=42, n_substations=3, n_transformers=24, horizon=365.0,
                     train_cutoff=None):
    """动态生成电力变压器时序知识图谱。

    关键设计（对应你的两个约束）：
    * 时间作为关系属性：每条边都是 Quad(头, 关系, 尾, 时间)，其中「时间」是
      关系实例自身的属性(见 Quad.time)，而不是游离在三元组外的独立坐标。
    * 新实体不断加入：变压器/传感器在 [0, horizon] 内随机时间被创建，
      其中创建时间晚于 train_cutoff 的实体，在训练阶段完全不可见
      —— 它们就是「归纳式」要预测的对象。

    返回: 一个字典，含实体、全部有向边、训练/测试切分、邻接表、关系词表等。
    """
    rng = np.random.default_rng(seed)
    rel2id, id2rel, inv, rel_head_types, rel_tail_types = build_relation_vocab()

    entities = []                      # Entity 列表
    ent_by_type = defaultdict(list)    # 类型 -> 实体 id 列表
    quads = []                         # 全部有向四元组 (h, r_id, t, τ)

    def add_entity(etype, created_at, name):
        eid = len(entities)
        entities.append(Entity(eid, etype, float(created_at), name))
        ent_by_type[etype].append(eid)
        return eid

    def add_quad(h, rname, t, tau):
        """添加一条有向边，并自动补一条反向边(便于 TGAT 双向聚合邻居)。

        边以 Quad 存储，tau 作为「关系的时间属性」挂在 relation 上。
        """
        rid = rel2id[rname]
        tau = float(tau)
        quads.append(Quad(h, rid, t, tau))
        quads.append(Quad(t, inv[rid], h, tau))

    # --- (1) 变电站：固定，t=0 就存在 ---
    subs = [add_entity("变电站", 0.0, f"变电站{i}") for i in range(n_substations)]

    # --- (2) 故障类型：固定 4 种，t=0 就存在 ---
    fault_types = ["过热", "局部放电", "绕组变形", "绝缘油劣化"]
    faults = [add_entity("故障类型", 0.0, f"故障_{f}") for f in fault_types]

    # --- (3) 变压器 + 部件：随时间加入(部分晚于 cutoff -> 归纳式新实体) ---
    component_types = ["绕组", "套管", "铁芯", "绝缘油", "冷却系统", "有载分接开关"]
    trf_created = rng.uniform(0, horizon, size=n_transformers)
    for ti, tc in enumerate(trf_created):
        tid = add_entity("变压器", tc, f"变压器{ti}")
        sub = subs[int(rng.integers(0, n_substations))]
        add_quad(tid, "位于", sub, tc)
        # 与同站、且已存在的变压器互联(互联)
        for other in ent_by_type["变压器"]:
            if other != tid and entities[other].created_at <= tc and rng.random() < 0.3:
                add_quad(tid, "互联", other, tc)
        # 部件随变压器一起创建
        for ct in component_types:
            cid = add_entity(ct, tc, f"{ct}{ti}")
            add_quad(tid, "包含部件", cid, tc)

    # --- (4) 传感器：随时间加入(部分晚于 cutoff -> 归纳式新实体) ---
    # 给 绕组/套管/绝缘油/铁芯 各配一个监测传感器
    monitored = (ent_by_type["绕组"] + ent_by_type["套管"]
                 + ent_by_type["绝缘油"] + ent_by_type["铁芯"])
    for ci, comp in enumerate(monitored):
        st = float(rng.uniform(0, horizon))
        sid = add_entity("传感器", st, f"传感器{ci}")
        add_quad(comp, "被监测", sid, st)

    # --- (5) 故障发生(导致 部件)：随时间演化，是图谱里"动态变化"的主要来源 ---
    all_components = (ent_by_type["绕组"] + ent_by_type["套管"]
                      + ent_by_type["绝缘油"] + ent_by_type["铁芯"])
    for _ in range(80):
        f = faults[int(rng.integers(0, len(faults)))]
        comp = all_components[int(rng.integers(0, len(all_components)))]
        tt = float(rng.uniform(horizon * 0.1, horizon))
        add_quad(f, "导致", comp, tt)

    # --- 时间切分：训练用 cutoff 之前的边，测试用之后的边 ---
    if train_cutoff is None:
        train_cutoff = horizon * 0.7
    train_quads = [q for q in quads if q.time <= train_cutoff]
    test_quads = [q for q in quads if q.time > train_cutoff]

    # 邻接表：neigh[实体] = [(邻居, 关系id, 时间), ...]，用于 TGAT 聚合
    # 注意：用「全部」边建表，这样测试时查询未来时间也能用到当时已发生的边。
    neigh = defaultdict(list)
    for q in quads:
        neigh[q.head].append((q.tail, q.relation, q.time))

    # 归纳式测试集：至少涉及一个"训练阶段不存在"的实体
    new_entities = {e.eid for e in entities if e.created_at > train_cutoff}
    inductive_test = [q for q in test_quads if (q.head in new_entities or q.tail in new_entities)]

    return {
        "entities": entities,
        "ent_by_type": ent_by_type,
        "quads": quads,
        "train_quads": train_quads,
        "test_quads": test_quads,
        "inductive_test": inductive_test,
        "neigh": neigh,
        "rel2id": rel2id,
        "id2rel": id2rel,
        "rel_head_types": rel_head_types,
        "rel_tail_types": rel_tail_types,
        "num_ent": len(entities),
        "num_rel": len(id2rel),
        "num_types": len(ENTITY_TYPES),
        "train_cutoff": train_cutoff,
        "new_entities": new_entities,
    }


# ====================================================================
# 3. 时间编码器 (Time2Vec 风格)
# ====================================================================
class TimeEncoder(nn.Module):
    """把连续时间戳映射到向量。

    同时包含：
      * 线性项  w(t)        —— 捕捉时间的单调趋势(越来越老/新)
      * 周期项  sin(f*t+φ)  —— 捕捉周期性(如昼夜/季节)
    这样模型既能感知"绝对时间位置"，也能感知"周期性节奏"。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(1, dim)                       # 线性部分
        # 时间坐标通常是「年份/天数」量级(如 2000+)。若线性权重用默认初始化(std=1)，
        # linear(t) 会达到数千，使时间编码与下游 DistMult 分数发散、loss 爆到数万。
        # 因此把线性权重初始化得很小，让时间编码始终保持在 O(1)，避免数值发散。
        nn.init.normal_(self.linear.weight, 0.0, 0.1)
        nn.init.normal_(self.linear.bias, 0.0, 0.1)
        self.freq = nn.Parameter(torch.randn(dim) * 1.0)      # 正弦频率(适中，避免高频梯度过大)
        self.phase = nn.Parameter(torch.randn(dim))           # 正弦相位

    def forward(self, t):
        # t: 标量或 (N,) 的时间戳张量
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.unsqueeze(-1)                                   # (N, 1)
        linear_part = self.linear(t)                          # (N, dim)
        periodic_part = torch.sin(self.freq * t + self.phase) # (N, dim)
        return linear_part + periodic_part


# ====================================================================
# 4. 模型：TGAT 编码器 + 时间感知 DistMult 解码器
# ====================================================================
class TemporalKGModel(nn.Module):
    def __init__(self, num_ent, num_rel, num_types, ent_etype,
                 dim=32, n_heads=4, dropout=0.1, time_scale=1.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.time_scale = time_scale

        # 实体/关系嵌入。注意：关系数量通常远小于实体数量且固定，
        # 新实体加入时【不需要】扩展关系表，只需扩展实体表(用类型初始化)。
        self.ent_emb = nn.Embedding(num_ent, dim)
        self.rel_emb = nn.Embedding(num_rel, dim)
        self.type_emb = nn.Embedding(num_types, dim)  # 仅用于初始化实体(尤其新实体)
        # 可学习分数缩放：让 DistMult 分数具备合适的幅度，便于 BCE 区分正负样本
        self.gamma = nn.Parameter(torch.ones(1))

        # 嵌入初始化幅度：DistMult 分数 = Σ h·r·t。若嵌入过小(原 std=0.1)，
        # 分数会聚集在 0 附近、sigmoid≈0.5，导致正负样本无法区分、推理排名接近随机
        # （这正是之前所有候选得分都挤在 ±0.6、真实尾实体排不进前列的原因）。
        # 把实体/关系嵌入初始化到 ~0.5 量级，使分数有 ±2 左右的区分度。
        nn.init.normal_(self.rel_emb.weight, 0.0, 0.5)
        nn.init.normal_(self.type_emb.weight, 0.0, 0.5)
        with torch.no_grad():
            # 用「类型嵌入 + 少量随机噪声」初始化实体嵌入：既保留同类实体的合理共同先验，
            # 又打破「同类实体初始完全相同」的对称性，使 TGAT 聚合后同类实体也能分化出各自
            # 独特的表示，从而让 DistMult 打分具备区分度（否则同类型实体得分几乎一致、
            # 推理排名接近随机）。
            for eid, ti in enumerate(ent_etype):
                self.ent_emb.weight[eid] = self.type_emb.weight[ti] + torch.randn(self.dim) * 0.1

        self.time_enc = TimeEncoder(dim)

        # TGAT 的注意力投影矩阵（query / key / value）
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    # -------------------- 关系的时间感知表示 --------------------
    def rel_repr(self, r_ids, tau):
        """时间感知的关系表示：把「时间」作为【关系的一个属性】叠加到基础关系嵌入上。

        关系实例 (relation, τ) 的表示 = 基础关系嵌入(rel_emb) + 时间编码(time_enc(τ))。
        这样同一个关系「位于」在 τ=10 与 τ=300 时表示不同 —— 时间成了关系本身的属性，
        而非一个独立于关系的第 4 维坐标。tau 与 r_ids 形状一致。
        """
        r = self.rel_emb(r_ids)                               # (N, dim) 基础关系嵌入
        # 时间作为关系属性：先把绝对时间 τ 按 time_scale 归一化到 O(1) 再编码，
        # 避免 τ≈2000 时 freq*τ 过大导致正弦支路梯度爆炸、训练失稳（失稳会让模型把实体
        # 嵌入压到极小来「补偿」，最终正负样本分数都挤在 0 附近、无法区分）。
        te = self.time_enc(tau / self.time_scale)             # (N, dim) 时间作为关系属性
        return r + te                                         # (N, dim) 时间感知关系表示

    # -------------------- 编码器：时序图注意力 --------------------
    def entity_repr(self, ent_ids, query_time, neigh):
        """计算在 query_time 时刻，各实体的「时间感知表示」。

        核心(归纳式的关键)：实体表示 = 对其【历史邻居】做注意力聚合，
        而不是查一张静态实体表。因此只要新实体有边，就能算出它的表示。

        邻居消息 = 邻居实体嵌入 + 时间感知关系表示(rel_repr, 含关系的时间属性)
                  + 时间编码(相对时间差, 仅用于调整注意力权重，让更近的边权重更高)
        """
        if not torch.is_tensor(query_time):
            query_time = torch.tensor(query_time, dtype=torch.float)
        B = ent_ids.shape[0]
        base = self.ent_emb(ent_ids)                          # (B, dim) 实体自身基础嵌入
        q = self.Wq(base).view(B, self.n_heads, self.head_dim)  # (B, H, hd) 注意力 query
        ctx = torch.zeros(B, self.n_heads, self.head_dim)

        for i in range(B):
            eid = int(ent_ids[i].item())
            qt = float(query_time[i].item())
            # 只聚合「发生时间 ≤ 查询时间」的历史邻居(未来边不参与)
            hist = [(nb, r, tt) for (nb, r, tt) in neigh.get(eid, []) if tt <= qt]
            if not hist:
                continue  # 无邻居时，保留 base 作为表示(残差)
            nb_ids = torch.tensor([x[0] for x in hist], dtype=torch.long)
            r_ids = torch.tensor([x[1] for x in hist], dtype=torch.long)
            tts = torch.tensor([qt - x[2] for x in hist], dtype=torch.float)  # 相对时间差
            nb_emb = self.ent_emb(nb_ids)                     # (N, dim)
            # 关系携带自己的时间属性：用 rel_repr(关系, 该边的绝对时间)
            r_emb = self.rel_repr(r_ids, torch.tensor([x[2] for x in hist], dtype=torch.float))
            te = self.time_enc(tts / self.time_scale)         # (N, dim) 相对时间差(调权重)
            f = nb_emb + r_emb + te                           # (N, dim) 邻居消息
            k = self.Wk(f).view(-1, self.n_heads, self.head_dim)
            v = self.Wv(f).view(-1, self.n_heads, self.head_dim)
            # 多头注意力：query 与每个邻居消息的相似度
            scores = torch.einsum("hd,nhd->hn", q[i], k) / math.sqrt(self.head_dim)  # (H, N)
            a = torch.softmax(scores, dim=-1)                 # (H, N)
            ctx[i] = torch.einsum("hn,nhd->hd", a, v)         # (H, hd) 加权聚合

        ctx = ctx.reshape(B, self.dim)
        ctx = self.drop(self.out(ctx))
        return base + ctx                                     # 残差连接：自身 + 邻居上下文

    # -------------------- 解码器：时间感知 DistMult --------------------
    def score(self, h_repr, r_ids, t_repr, tau):
        """DistMult 打分：score = <h, r(τ), t> = sum_d h_d * r_d(τ) * t_d。

        关系 r 用 rel_repr(r, τ) 表示 —— 因为 r 随查询时间 τ 变化(时间作为关系属性)，
        所以同一三元组在不同时间的得分不同 —— 即「时间感知」。
        """
        r = self.rel_repr(r_ids, tau)                         # (B, dim) 时间感知关系表示
        return self.gamma * torch.sum(h_repr * r * t_repr, dim=-1)  # (B,)


# ====================================================================
# 5. 负采样 + 训练
# ====================================================================
SAMPLE_RNG = np.random.default_rng(12345)


def sample_negative(ents, rels, type_map, ent_by_type, n_neg):
    """按关系的「合法尾/头类型」做类型感知负采样(比纯随机更贴近真实 KG)。

    ents: (B,) 当前头或尾实体；rels: (B,) 关系 id；
    type_map: rel_tail_types 或 rel_head_types。
    返回: (B, n_neg) 负样本实体 id。
    """
    B = len(ents)
    out = torch.zeros(B, n_neg, dtype=torch.long)
    for i in range(B):
        allowed = type_map.get(int(rels[i]), None)
        cand = []
        if allowed is None:
            cand = [e for et in ent_by_type for e in ent_by_type[et]]  # 兜底：全部实体
        else:
            for et in allowed:
                cand += ent_by_type[et]
        idx = SAMPLE_RNG.integers(0, len(cand), size=n_neg)
        out[i] = torch.tensor([cand[j] for j in idx], dtype=torch.long)
    return out


def fmt_quad(q, entities, id2rel):
    """把一条 Quad 渲染成可读字符串：(头(类型), 关系, 尾(类型), τ=时间)。"""
    e = entities
    return (f"({e[q.head].name}({e[q.head].etype}), "
            f"{id2rel[q.relation]}, "
            f"{e[q.tail].name}({e[q.tail].etype}), "
            f"τ={q.time:.1f})")


def train(model, train_quads, neigh, ent_by_type, rel_tail_types, rel_head_types,
          optimizer, epochs=40, batch_size=32, n_neg=5,
          entities=None, id2rel=None, log_quads_every=5, quad_sample=8):
    """训练模型。

    若传入 entities / id2rel，会在训练过程中打印数据集的四元组信息：
      * 训练开始前：打印训练集前 quad_sample 条四元组(带可读名称)。
      * 每隔 log_quads_every 个 epoch：打印该 epoch 首个批次的四元组。
    """
    model.train()
    quads = list(train_quads)
    n_train = len(quads)
    print(f"[训练] 样本数={n_train}, epochs={epochs}, batch={batch_size}, 负样本/正={n_neg}")

    # —— 训练前：打印训练集四元组示例(带实体名/关系名/时间) ——
    if entities is not None and id2rel is not None:
        print(f"[训练集四元组示例] 共 {n_train} 条，以下展示前 {min(quad_sample, n_train)} 条：")
        for q in quads[:quad_sample]:
            print("    " + fmt_quad(q, entities, id2rel))

    for ep in range(epochs):
        SAMPLE_RNG.shuffle(quads)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, n_train, batch_size):
            batch = quads[i:i + batch_size]
            B = len(batch)                                        # 当前批次正样本数

            # —— 训练过程中：每隔 log_quads_every 个 epoch，打印首个批次的四元组 ——
            if (entities is not None and id2rel is not None
                    and n_batch == 0 and (ep == 0 or (ep + 1) % log_quads_every == 0)):
                print(f"[训练过程 epoch {ep+1}] 当前批次四元组(共 {B} 条，展示前 {min(quad_sample, B)} 条)：")
                for q in batch[:quad_sample]:
                    print("    " + fmt_quad(q, entities, id2rel))

            h = torch.tensor([q.head for q in batch], dtype=torch.long)
            r = torch.tensor([q.relation for q in batch], dtype=torch.long)
            t = torch.tensor([q.tail for q in batch], dtype=torch.long)
            tau = torch.tensor([q.time for q in batch], dtype=torch.float)

            # 正样本：在查询时间 τ 下，头/尾实体的时间感知表示
            h_repr = model.entity_repr(h, tau, neigh)         # (B, dim)
            t_repr = model.entity_repr(t, tau, neigh)         # (B, dim)
            pos_score = model.score(h_repr, r, t_repr, tau)   # (B,)

            # 负采样：分别破坏尾实体 / 头实体
            neg_t = sample_negative(t, r, rel_tail_types, ent_by_type, n_neg)  # (B, n_neg)
            neg_h = sample_negative(h, r, rel_head_types, ent_by_type, n_neg)  # (B, n_neg)

            # 破坏尾：保持头不变，给每个正样本配 n_neg 个假尾
            h_repr_exp = h_repr.repeat_interleave(n_neg, dim=0)               # (B*n_neg, dim)
            r_exp = r.repeat_interleave(n_neg)                                # (B*n_neg,)
            neg_t_flat = neg_t.reshape(-1)                                    # (B*n_neg,)
            neg_t_repr = model.entity_repr(neg_t_flat, tau.repeat_interleave(n_neg), neigh)
            neg_t_score = model.score(h_repr_exp, r_exp, neg_t_repr,
                                      tau.repeat_interleave(n_neg)).view(B, n_neg)

            # 破坏头：保持尾不变，给每个正样本配 n_neg 个假头
            t_repr_exp = t_repr.repeat_interleave(n_neg, dim=0)
            neg_h_flat = neg_h.reshape(-1)
            neg_h_repr = model.entity_repr(neg_h_flat, tau.repeat_interleave(n_neg), neigh)
            neg_h_score = model.score(neg_h_repr, r_exp, t_repr_exp,
                                      tau.repeat_interleave(n_neg)).view(B, n_neg)

            # 损失：正样本标签 1，负样本标签 0（二元交叉熵）
            loss_pos = F.binary_cross_entropy_with_logits(pos_score, torch.ones(B))
            loss_t = F.binary_cross_entropy_with_logits(neg_t_score, torch.zeros(B, n_neg))
            loss_h = F.binary_cross_entropy_with_logits(neg_h_score, torch.zeros(B, n_neg))
            loss = loss_pos + loss_t + loss_h

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  loss={total_loss/max(n_batch,1):.4f}")


# ====================================================================
# 6. 评估与推理
# ====================================================================
def candidate_entities(rel_id, type_map, ent_by_type):
    """某关系允许的候选尾/头实体列表(按本体约束缩小范围)。"""
    cands = []
    for et in type_map.get(int(rel_id), []):
        cands += ent_by_type[et]
    return cands


def evaluate(model, test_quads, neigh, ent_by_type, rel_tail_types, sample_n=60):
    """在测试集上计算过滤式 MRR(平均倒数排名)。

    过滤式(filtered)：排名时把「同一(头,关系,时间)下的其它真实尾实体」剔除，
    避免它们压低真正尾实体的排名，这是 KG 链接预测的标准评测方式。
    """
    model.eval()
    # 统计每个 (头,关系,时间) 对应的全部真实尾实体
    true_tails = defaultdict(set)
    for q in test_quads:
        true_tails[(q.head, q.relation, round(q.time, 3))].add(q.tail)

    ranks = []
    order = list(test_quads)
    SAMPLE_RNG.shuffle(order)
    with torch.no_grad():
        for q in order[:sample_n]:
            h, r, t, tau = q.head, q.relation, q.tail, q.time
            cands = candidate_entities(r, rel_tail_types, ent_by_type)
            if not cands or t not in cands:
                continue
            cands_t = torch.tensor(cands, dtype=torch.long)
            h_repr = model.entity_repr(torch.tensor([h]), torch.tensor([tau]), neigh)
            cands_repr = model.entity_repr(cands_t, torch.tensor([tau] * len(cands)), neigh)
            # 关系携带时间属性：用 rel_repr(关系, 查询时间 τ)；统一走 model.score
            r_ids = torch.tensor([r]).expand(len(cands))
            taus = torch.tensor([tau] * len(cands))
            scores = model.score(h_repr.expand(len(cands), -1), r_ids, cands_repr, taus)

            true_score = float(scores[cands.index(t)])
            # 过滤：去掉其它真实尾实体后再排名
            filtered = [(c, s.item()) for c, s in zip(cands, scores)
                        if c not in true_tails[(h, r, round(tau, 3))] or c == t]
            rank = 1 + sum(1 for _, s in filtered if s > true_score)
            ranks.append(rank)

    if not ranks:
        return 0.0, []
    mrr = sum(1.0 / rk for rk in ranks) / len(ranks)
    return mrr, ranks


def predict_tails(model, h, r, tau, neigh, ent_by_type, rel_tail_types, k=5):
    """推理：给定 (头, 关系, 时间)，返回 Top-K 候选尾实体及其得分。"""
    model.eval()
    with torch.no_grad():
        cands = candidate_entities(r, rel_tail_types, ent_by_type)
        if not cands:
            return []
        k = min(k, len(cands))   # 候选尾实体可能少于 k（如"位于"只有 3 个变电站）
        cands_t = torch.tensor(cands, dtype=torch.long)
        h_repr = model.entity_repr(torch.tensor([h]), torch.tensor([tau]), neigh)
        cands_repr = model.entity_repr(cands_t, torch.tensor([tau] * len(cands)), neigh)
        # 关系携带时间属性：用 rel_repr(关系, 查询时间 τ)；统一走 model.score
        r_ids = torch.tensor([r]).expand(len(cands))
        taus = torch.tensor([tau] * len(cands))
        scores = model.score(h_repr.expand(len(cands), -1), r_ids, cands_repr, taus)
        topk = torch.topk(scores, k)
        return [(int(cands[i]), float(scores[i])) for i in topk.indices.tolist()]


# ====================================================================
# 7. 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="电力变压器时序知识图谱链接预测(独立 demo)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-transformers", type=int, default=24)
    args = parser.parse_args()

    # ---- 1) 生成动态时序知识图谱 ----
    data = generate_dataset(seed=args.seed, n_transformers=args.n_transformers)
    entities = data["entities"]
    ent_by_type = data["ent_by_type"]
    neigh = data["neigh"]
    rel2id = data["rel2id"]
    id2rel = data["id2rel"]
    rel_tail_types = data["rel_tail_types"]
    rel_head_types = data["rel_head_types"]

    print("=" * 70)
    print("【本体与数据概览】")
    print(f"  实体类型数={data['num_types']}, 关系类型数={data['num_rel']}")
    print(f"  实体总数={data['num_ent']}, 有向边总数={len(data['quads'])}")
    print(f"  训练边={len(data['train_quads'])}, 测试边={len(data['test_quads'])}, "
          f"归纳式测试边={len(data['inductive_test'])}")
    print(f"  时间切分点 cutoff={data['train_cutoff']:.1f} (之后创建的实体为归纳式新实体)")
    print("  各类型实体数:", {et: len(ent_by_type[et]) for et in ENTITY_TYPES})
    print("=" * 70)

    # 实体类型索引(用于初始化实体嵌入)
    ent_etype = [ENTITY_TYPES.index(e.etype) for e in entities]

    # 时间归一化尺度：把时间戳(年份/天数量级)缩放到 O(1)，避免 TimeEncoder 中 freq*τ
    # 过大导致梯度爆炸、训练失稳。取全部四元组时间的最大绝对值。
    all_times = [q.time for q in data["quads"]]
    time_scale = float(max((abs(t) for t in all_times), default=1.0)) or 1.0

    # ---- 2) 构建模型 ----
    model = TemporalKGModel(
        num_ent=data["num_ent"],
        num_rel=data["num_rel"],
        num_types=data["num_types"],
        ent_etype=ent_etype,
        dim=args.dim,
        n_heads=4,
        time_scale=time_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 3) 训练 ----
    train(model, data["train_quads"], neigh, ent_by_type, rel_tail_types, rel_head_types,
          optimizer, epochs=args.epochs, batch_size=args.batch, n_neg=args.neg,
          entities=entities, id2rel=id2rel)

    # ---- 4) 评估 ----
    mrr_all, _ = evaluate(model, data["test_quads"], neigh, ent_by_type, rel_tail_types)
    mrr_ind, _ = evaluate(model, data["inductive_test"], neigh, ent_by_type, rel_tail_types)
    print("=" * 70)
    print(f"【评估结果】测试集 MRR={mrr_all:.4f}  |  归纳式(新实体)测试集 MRR={mrr_ind:.4f}")
    print("  (MRR 越接近 1 越好；归纳式 MRR>0 说明模型能预测从未见过的实体)")
    print("=" * 70)

    # ---- 5) 示例推理(可解释输出) ----
    name = lambda eid: f"{entities[eid].name}({entities[eid].etype})"

    def demo_one(quad, tag):
        h, r, t, tau = quad.head, quad.relation, quad.tail, quad.time
        top = predict_tails(model, h, r, tau, neigh, ent_by_type, rel_tail_types, k=5)
        true_in_top5 = any(c == t for c, _ in top)
        print(f"\n[{tag}] 查询: ({name(h)}, {id2rel[r]}, ?)  @时间τ={tau:.1f}")
        print(f"  真实尾实体: {name(t)}")
        print(f"  Top-5 预测尾实体:")
        for c, s in top:
            mark = "  <== 命中真实尾" if c == t else ""
            print(f"    - {name(c)}   得分={s:+.4f}{mark}")
        print(f"  => 真实尾是否在 Top-5: {true_in_top5}")

    print("\n" + "=" * 70)
    print("【示例推理】")
    # 普通测试样本
    if data["test_quads"]:
        demo_one(data["test_quads"][0], "普通测试样本")
    # 归纳式(含新实体)测试样本
    if data["inductive_test"]:
        demo_one(data["inductive_test"][0], "归纳式新实体样本")
    print("=" * 70)


if __name__ == "__main__":
    main()

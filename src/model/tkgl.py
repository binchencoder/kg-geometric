# -*- coding: utf-8 -*-
"""
src/model/tkgl.py
=====================================================================
TKGL-Smallpedia 时序知识图谱链接预测 —— 模型与 checkpoint 序列化

模型：TGAT 风格时序图注意力编码器 + 时间感知 DistMult 解码器
      （「时间」作为关系的一个属性：rel_repr = rel_emb + TimeEncoder(τ)）

本文件自包含，仅依赖 numpy / torch，可被 src/tkgl/train.py 与
src/tkgl/predict.py 复用。原实现位于 demo/tkgl_smallpedia_tkg.py，
此处抽取为独立模块。
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 1. 时间编码器（Time2Vec 风格：线性项 + 周期项）
# ====================================================================
class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(1, dim)
        # 时间已在外面按 time_scale 归一化到 O(1)，线性权重用常规幅度即可
        nn.init.normal_(self.linear.weight, 0.0, 0.1)
        nn.init.normal_(self.linear.bias, 0.0, 0.1)
        self.freq = nn.Parameter(torch.randn(dim) * 1.0)
        self.phase = nn.Parameter(torch.randn(dim))

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.unsqueeze(-1)  # (..., 1)
        # 裁剪到合理范围，避免静态哨兵时间(1899)在居中缩放后变成极端值导致数值爆炸
        t = t.clamp(-12.0, 12.0)
        return self.linear(t) + torch.sin(self.freq * t + self.phase)


# ====================================================================
# 2. 模型：向量化 TGAT 编码器 + 时间感知 DistMult 解码器
# ====================================================================
class TemporalKGModel(nn.Module):
    def __init__(
            self,
            num_ent, num_rel,
            dim=64,
            n_heads=4,
            dropout=0.1,
            time_scale=1.0,
            t_center=0.0,
            neigh=None,
            max_neigh=20
    ):
        super().__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.time_scale = time_scale
        # 时间中心化：rel_repr 用 (tau - t_center)/time_scale，使输入落在 ~[-5,5]，
        # 周期性时间项能随年份变化（原 time_scale=max(|ts|)≈2013 把所有年份压到
        # [0.98,1.0]，时间信号几乎丢失，是推理打不准的重要原因）。
        self.t_center = float(t_center)
        self.neigh = neigh if neigh is not None else {}
        # 每个实体最多缓存 max_neigh 个邻居（度数超过时均匀采样保留），
        # 用于把邻接表预先「打包成定长张量」，使 entity_repr 完全向量化（无 Python 循环）。
        self.max_neigh = max_neigh
        self._nb_cpu = None  # (nb, rel, tt, cnt) numpy 定长数组，一次性构建
        self._nb_dev = None  # 当前已搬到的设备
        self._nb_t = self._rl_t = self._tt_t = self._cnt_t = None

        self.ent_emb = nn.Embedding(num_ent, dim)
        self.rel_emb = nn.Embedding(num_rel, dim)
        self.gamma = nn.Parameter(torch.ones(1))  # 可学习分数缩放

        # DistMult 分数 = Σ h·r·t；把嵌入初始化到 ~0.4 量级，使分数有 ±2 左右的
        # 区分度（过小会让正负样本分数都挤在 0 附近、无法区分）。
        nn.init.normal_(self.ent_emb.weight, 0.0, 0.4)
        nn.init.normal_(self.rel_emb.weight, 0.0, 0.4)

        self.time_enc = TimeEncoder(dim)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    # -------------------- 时间感知关系表示 --------------------
    def rel_repr(self, r_ids, tau):
        r = self.rel_emb(r_ids)
        t = (tau - self.t_center) / self.time_scale  # 居中 + 缩放，使时间信号可分辨
        te = self.time_enc(t)  # 时间作为关系属性
        return r + te

    # -------------------- 邻接表 → 定长张量（一次性预打包） --------------------
    def _ensure_neighbor_cache(self, device):
        """把 dict 形式的邻接表打包成定长 (num_ent, max_neigh) 张量，供向量化聚合。

        - 度数 > max_neigh 的实体：均匀随机采样 max_neigh 个邻居（覆盖各时间段）。
        - 只构建一次（CPU numpy），之后按需搬到目标 device 并缓存。
        这样 entity_repr 里就不再有「逐实体 Python 循环 + cpu/numpy 同步」，
        而是纯张量 gather，速度提升可达 1~2 个数量级。
        """
        if self._nb_cpu is None:
            M, num = self.max_neigh, self.num_ent
            nb = np.zeros((num, M), dtype=np.int64)
            rl = np.zeros((num, M), dtype=np.int64)
            tt = np.full((num, M), 1e18, dtype=np.float32)  # 空位给极大时间，天然被 tt<=qt 过滤掉
            cnt = np.zeros((num,), dtype=np.int64)
            rng = np.random.default_rng(12345)
            for e, (na, ra, ta) in self.neigh.items():
                k = int(ta.size)
                if k == 0:
                    continue
                if k > M:
                    idx = rng.choice(k, M, replace=False)
                    na, ra, ta = na[idx], ra[idx], ta[idx]
                    k = M
                nb[e, :k] = na
                rl[e, :k] = ra
                tt[e, :k] = ta
                cnt[e] = k
            self._nb_cpu = (nb, rl, tt, cnt)

        if self._nb_dev != device:
            nb, rl, tt, cnt = self._nb_cpu
            self._nb_t = torch.from_numpy(nb).to(device)
            self._rl_t = torch.from_numpy(rl).to(device)
            self._tt_t = torch.from_numpy(tt).to(device)
            self._cnt_t = torch.from_numpy(cnt).to(device)
            self._nb_dev = device

    # -------------------- 向量化实体表示（定长邻居 + 多头注意力，无 Python 循环） --------------------
    def entity_repr(self, ent_ids, query_time, num_samples=None):
        """计算在 query_time 时刻各实体的时间感知表示（完全向量化）。

        邻居消息 = 邻居实体嵌入 + 时间感知关系表示 + 相对时间差编码，
        仅聚合 tt <= qt 的历史邻居（未来边不参与，保证无时间泄漏）。
        num_samples 参数已弃用（保留仅为兼容旧调用），邻居数由 self.max_neigh 决定。
        """
        device = ent_ids.device
        self._ensure_neighbor_cache(device)
        B = ent_ids.shape[0]
        M = self.max_neigh
        base = self.ent_emb(ent_ids)  # (B, dim)
        q = self.Wq(base).view(B, self.n_heads, self.head_dim)

        # 定长 gather（纯张量索引，无 Python 循环）
        nb_t = self._nb_t[ent_ids]  # (B, M)
        rel_t = self._rl_t[ent_ids]  # (B, M)
        tt_t = self._tt_t[ent_ids]  # (B, M)
        cnt = self._cnt_t[ent_ids]  # (B,)

        # 掩码：位置在真实邻居数内 且 邻居时间 <= 查询时间（历史）
        ar = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
        qt = query_time.unsqueeze(1)  # (B, 1)
        mask_t = ((ar < cnt.unsqueeze(1)) & (tt_t <= qt)).float()  # (B, M)

        nb_emb = self.ent_emb(nb_t)  # (B, M, dim)
        r_emb = self.rel_repr(rel_t.reshape(-1), tt_t.reshape(-1)).view(B, M, self.dim)
        rel_time = (qt - tt_t) / self.time_scale
        te = self.time_enc(rel_time)  # (B, M, dim)
        f = nb_emb + r_emb + te  # (B, M, dim)

        kk = self.Wk(f).view(B, M, self.n_heads, self.head_dim)
        vv = self.Wv(f).view(B, M, self.n_heads, self.head_dim)
        scores = torch.einsum("bhd,bkhd->bhk", q, kk) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask_t.unsqueeze(1) == 0, -1e9)
        a = torch.softmax(scores, dim=-1)  # (B, H, M)
        ctx = torch.einsum("bhk,bkhd->bhd", a, vv).reshape(B, self.dim)
        ctx = self.drop(self.out(ctx))
        # 无有效邻居的实体：把上下文置零，避免 softmax(全 -1e9) 的无意义均值污染表示
        has_nb = (mask_t.sum(dim=1, keepdim=True) > 0).float()
        ctx = ctx * has_nb
        return base + ctx  # 残差：自身 + 邻居上下文

    # -------------------- 时间感知 DistMult 打分 --------------------
    def score(self, h_repr, r_ids, t_repr, tau):
        r = self.rel_repr(r_ids, tau)
        return self.gamma * torch.sum(h_repr * r * t_repr, dim=-1)


# ====================================================================
# 3. 模型保存 / 加载（自包含：含邻接表与真实尾索引）
# ====================================================================
def save_checkpoint(model, data, filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "num_ent": model.num_ent,
            "num_rel": model.num_rel,
            "dim": model.dim,
            "n_heads": model.n_heads,
            "time_scale": model.time_scale,
            "t_center": model.t_center,
            "max_neigh": model.max_neigh,
        },
        "entity2id": data["entity2id"],
        "id2entity": data["id2entity"],
        "relation2id": data["relation2id"],
        "id2relation": data["id2relation"],
        "neigh": model.neigh,
        "true_tails": data["true_tails"],
        "splits": data["splits"],
        # 小体积的四元组（推理评测用），一并存入以便推理时「仅加载 checkpoint」
        "test_quads": data["test_quads"],
        "val_quads": data["val_quads"],
    }, filepath)


def load_checkpoint(filepath, device="cpu"):
    ckpt = torch.load(filepath, map_location=device)
    cfg = ckpt["config"]
    model = TemporalKGModel(
        num_ent=cfg["num_ent"],
        num_rel=cfg["num_rel"],
        dim=cfg["dim"],
        n_heads=cfg["n_heads"],
        time_scale=cfg["time_scale"],
        t_center=cfg.get("t_center", 0.0),
        neigh=ckpt["neigh"],
        max_neigh=cfg.get("max_neigh", 20),
    )
    model._id2entity = ckpt["id2entity"]
    model._id2relation = ckpt["id2relation"]
    model._true_tails = ckpt["true_tails"]
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    data = {
        "true_tails": ckpt["true_tails"],
        "splits": ckpt["splits"],
        "id2entity": ckpt["id2entity"],
        "id2relation": ckpt["id2relation"],
        "entity2id": ckpt["entity2id"],
        "relation2id": ckpt["relation2id"],
        "test_quads": ckpt["test_quads"],
        "val_quads": ckpt["val_quads"],
    }
    return model, data

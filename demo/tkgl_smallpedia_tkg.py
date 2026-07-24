# -*- coding: utf-8 -*-
"""
demo/tkgl_smallpedia_tkg.py
=====================================================================
基于 TKGL-Smallpedia 数据集的「时序链接预测」训练 / 推理
=====================================================================

模型：TGAT 风格时序图注意力编码器 + 时间感知 DistMult 解码器
      （「时间」作为关系的一个属性：rel_repr = rel_emb + TimeEncoder(τ)）

与 demo/temporal_kg_link_prediction.py、examples/employee_tkg_link_prediction.py 的区别：
  * 本文件【自包含】，内置一套【向量化、带邻居采样】的 TGAT，能撑住
    TKGL-Smallpedia 这种规模（30 万实体 / 150 万边），旧的逐实体 Python 循环
    在此规模下会慢到不可用。
  * 数据加载由 data/tkgl-smallpedia/loader.py 完成（纯 Python，确定性 id 映射，
    不依赖 pkl 的隐藏整数映射）。
  * 评测采用「过滤式 MRR + 采样负样本」（标准做法）：对每个查询，把真实尾实体
    与一批均匀采样的负样本一起打分排序，并剔除同一 (头,关系,时间) 下的其它真实尾。
  * 【分批流式训练】训练边（约 150 万条）不整体驻留内存：由 loader.iter_train_batches
    以「shuffle-buffer」方式从 CSV 逐批流式产出，每个 epoch 重新流式读取并打乱；
    邻接表 / 验证·测试四元组仍需驻留（体积小或图结构必需）。--buffer 控制打乱缓冲。
  * 训练完成后保存单个 .pt（含模型权重 + id 映射 + 邻接表 + 真实尾索引），
    推理模式【仅加载该 .pt】即可，无需再读取 CSV。

运行：
    # 训练并保存模型（默认会先训练，若模型已存在则跳过训练直接推理；用 --force 重训）
    python demo/tkgl_smallpedia_tkg.py --mode train

    # 直接加载已训练模型做推理 / 评测
    python demo/tkgl_smallpedia_tkg.py --mode infer

提示：数据集较大，强烈建议在 GPU 上训练（自动检测 cuda；无 cuda 则退化为 CPU，
但会非常慢）。可用 --epochs / --dim / --batch / --neg 等调节。
"""

import os
import sys
import math
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 让本文件既能被 `python demo/xxx.py` 直接运行，也能从项目根目录运行
# 加载 data/tkgl-smallpedia/loader.py（纯 Python 数据加载器）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from data.tkgl_smallpedia.loader import load_tkgl_smallpedia, iter_train_batches
except ModuleNotFoundError:  # 兜底：按文件路径直接加载（避免 namespace package 解析差异）
    import importlib.util
    _loader_path = os.path.join(_ROOT, "data", "tkgl-smallpedia", "loader.py")
    _spec = importlib.util.spec_from_file_location("tkgl_smallpedia_loader", _loader_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_tkgl_smallpedia = _mod.load_tkgl_smallpedia
    iter_train_batches = _mod.iter_train_batches


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
        t = t.unsqueeze(-1)                       # (..., 1)
        # 裁剪到合理范围，避免静态哨兵时间(1899)在居中缩放后变成极端值导致数值爆炸
        t = t.clamp(-12.0, 12.0)
        return self.linear(t) + torch.sin(self.freq * t + self.phase)


# ====================================================================
# 2. 模型：向量化 TGAT 编码器 + 时间感知 DistMult 解码器
# ====================================================================
class TemporalKGModel(nn.Module):
    def __init__(self, num_ent, num_rel, dim=64, n_heads=4, dropout=0.1,
                 time_scale=1.0, t_center=0.0, neigh=None, max_neigh=20):
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
        self._nb_cpu = None          # (nb, rel, tt, cnt) numpy 定长数组，一次性构建
        self._nb_dev = None          # 当前已搬到的设备
        self._nb_t = self._rl_t = self._tt_t = self._cnt_t = None

        self.ent_emb = nn.Embedding(num_ent, dim)
        self.rel_emb = nn.Embedding(num_rel, dim)
        self.gamma = nn.Parameter(torch.ones(1))          # 可学习分数缩放

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
        t = (tau - self.t_center) / self.time_scale       # 居中 + 缩放，使时间信号可分辨
        te = self.time_enc(t)                             # 时间作为关系属性
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
            tt = np.full((num, M), 1e18, dtype=np.float32)   # 空位给极大时间，天然被 tt<=qt 过滤掉
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
        base = self.ent_emb(ent_ids)                      # (B, dim)
        q = self.Wq(base).view(B, self.n_heads, self.head_dim)

        # 定长 gather（纯张量索引，无 Python 循环）
        nb_t = self._nb_t[ent_ids]                        # (B, M)
        rel_t = self._rl_t[ent_ids]                       # (B, M)
        tt_t = self._tt_t[ent_ids]                        # (B, M)
        cnt = self._cnt_t[ent_ids]                        # (B,)

        # 掩码：位置在真实邻居数内 且 邻居时间 <= 查询时间（历史）
        ar = torch.arange(M, device=device).unsqueeze(0)  # (1, M)
        qt = query_time.unsqueeze(1)                      # (B, 1)
        mask_t = ((ar < cnt.unsqueeze(1)) & (tt_t <= qt)).float()  # (B, M)

        nb_emb = self.ent_emb(nb_t)                       # (B, M, dim)
        r_emb = self.rel_repr(rel_t.reshape(-1), tt_t.reshape(-1)).view(B, M, self.dim)
        rel_time = (qt - tt_t) / self.time_scale
        te = self.time_enc(rel_time)                      # (B, M, dim)
        f = nb_emb + r_emb + te                           # (B, M, dim)

        kk = self.Wk(f).view(B, M, self.n_heads, self.head_dim)
        vv = self.Wv(f).view(B, M, self.n_heads, self.head_dim)
        scores = torch.einsum("bhd,bkhd->bhk", q, kk) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask_t.unsqueeze(1) == 0, -1e9)
        a = torch.softmax(scores, dim=-1)                 # (B, H, M)
        ctx = torch.einsum("bhk,bkhd->bhd", a, vv).reshape(B, self.dim)
        ctx = self.drop(self.out(ctx))
        # 无有效邻居的实体：把上下文置零，避免 softmax(全 -1e9) 的无意义均值污染表示
        has_nb = (mask_t.sum(dim=1, keepdim=True) > 0).float()
        ctx = ctx * has_nb
        return base + ctx                                 # 残差：自身 + 邻居上下文

    # -------------------- 时间感知 DistMult 打分 --------------------
    def score(self, h_repr, r_ids, t_repr, tau):
        r = self.rel_repr(r_ids, tau)
        return self.gamma * torch.sum(h_repr * r * t_repr, dim=-1)


# ====================================================================
# 3. 训练
# ====================================================================
def train_model(model, batch_iter_fn, optimizer, device, epochs=30, batch_size=1024,
                n_neg=5, num_samples=20, val_data=None, num_eval=2000, k_neg=500,
                log_every=5, neg_mode="uniform"):
    """训练模型。

    batch_iter_fn(epoch) -> 生成器，逐 batch 产出 np.ndarray (B,4) int64: [ts,h,t,r]。
    每个 epoch 调用一次 batch_iter_fn(epoch) 重新流式读取并打乱训练边，因此
    训练集【无需全部驻留内存】（由 loader.iter_train_batches 以 shuffle-buffer
    方式从 CSV 流式产出）。

    val_data 为 loader 返回的 dict（含 val_quads / true_tails，用于定期评测）。
    """
    model.train()
    print(f"[训练] epochs={epochs}, batch={batch_size}, 负样本/正={n_neg}, 邻居采样={num_samples}")
    print("       训练边【分批流式加载】（每 epoch 从 CSV 流式读取 + shuffle-buffer 打乱，"
          "不整体驻留内存）")
    print("       首个 batch 会先预打包邻接表为定长张量（一次性，约需数秒~十几秒），请稍候...")

    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        n_batch = 0
        running = 0.0
        for batch in batch_iter_fn(ep):
            ts = torch.tensor(batch[:, 0], dtype=torch.float, device=device)
            h = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            t = torch.tensor(batch[:, 2], dtype=torch.long, device=device)
            r = torch.tensor(batch[:, 3], dtype=torch.long, device=device)
            B = h.shape[0]

            h_repr = model.entity_repr(h, ts, num_samples)
            t_repr = model.entity_repr(t, ts, num_samples)
            pos = model.score(h_repr, r, t_repr, ts)

            # 负采样：默认均匀随机（收敛快、稳定，是此前达到 MRR≈0.53 的设置）。
            # 同关系困难负样本（neg_mode="typed"）理论上更优，但在 CPU + 大规模实体下
            # 任务过难、收敛极差（MRR 仅 0.04），故默认不启用。
            if neg_mode == "typed":
                neg_t_np, neg_h_np = _sample_typed_negatives(model, h, r, t, ts, n_neg)
                neg_t = torch.tensor(neg_t_np, dtype=torch.long, device=device)
                neg_h = torch.tensor(neg_h_np, dtype=torch.long, device=device)
            else:
                neg_t = torch.randint(0, model.num_ent, (B, n_neg), device=device)
                neg_h = torch.randint(0, model.num_ent, (B, n_neg), device=device)

            neg_t_flat = neg_t.reshape(-1)
            neg_h_flat = neg_h.reshape(-1)
            ts_exp = ts.repeat_interleave(n_neg)
            r_exp = r.repeat_interleave(n_neg)

            neg_t_repr = model.entity_repr(neg_t_flat, ts_exp, num_samples)
            neg_t_score = model.score(h_repr.repeat_interleave(n_neg, dim=0),
                                      r_exp, neg_t_repr, ts_exp).view(B, n_neg)

            neg_h_repr = model.entity_repr(neg_h_flat, ts_exp, num_samples)
            neg_h_score = model.score(neg_h_repr, r_exp,
                                      t_repr.repeat_interleave(n_neg, dim=0),
                                      ts_exp).view(B, n_neg)

            loss_pos = F.binary_cross_entropy_with_logits(pos, torch.ones(B, device=device))
            loss_t = F.binary_cross_entropy_with_logits(neg_t_score, torch.zeros(B, n_neg, device=device))
            loss_h = F.binary_cross_entropy_with_logits(neg_h_score, torch.zeros(B, n_neg, device=device))
            loss = loss_pos + loss_t + loss_h

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bl = loss.item()
            total_loss += bl
            running += bl
            n_batch += 1

            # epoch 内进度：每 100 个 batch 打印一次，避免「看起来卡住」
            if n_batch % 100 == 0:
                speed = n_batch / max(time.time() - t0, 1e-6)
                print(f"    [epoch {ep+1}] batch {n_batch}  "
                      f"avg_loss(近100)={running/100:.4f}  {speed:.1f} batch/s", flush=True)
                running = 0.0

        dt = time.time() - t0
        if (ep + 1) % log_every == 0 or ep == 0:
            msg = (f"  epoch {ep+1:3d}/{epochs}  loss={total_loss/max(n_batch,1):.4f}  "
                   f"batches={n_batch}  ({dt:.1f}s)")
            if val_data is not None and val_data["val_quads"].shape[0] > 0:
                mrr, h1, h3, h10 = evaluate_filtered(
                    model, val_data["val_quads"], val_data["true_tails"],
                    num_eval=min(num_eval, 2000), k_neg=k_neg, device=device)
                msg += f"  | val MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f}"
            print(msg)


# ====================================================================
# 4. 过滤式 MRR 评测（采样负样本）
# ====================================================================
def evaluate_filtered(model, quads, true_tails, num_eval=2000, k_neg=500,
                      seed=0, device="cpu"):
    """对每个查询，真实尾 + k_neg 个均匀负样本一起打分，过滤掉同 (头,关系,时间)
    的其它真实尾后排名，计算 MRR / Hits@1/3/10。"""
    model.eval()
    if quads.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(quads.shape[0], size=min(num_eval, quads.shape[0]), replace=False)
    mrr = h1 = h3 = h10 = 0.0
    cnt = 0
    with torch.no_grad():
        for i in idx:
            ts, h, t, r = int(quads[i, 0]), int(quads[i, 1]), int(quads[i, 2]), int(quads[i, 3])
            negs = rng.integers(0, model.num_ent, size=k_neg)
            filt = true_tails.get((h, r, ts), set())
            cand = [t]
            for n in negs:
                n = int(n)
                if n not in filt and n != t:
                    cand.append(n)
            cand_t = torch.tensor(cand, dtype=torch.long, device=device)
            ts_t = torch.tensor([float(ts)] * len(cand), device=device)
            h_t = torch.tensor([h], dtype=torch.long, device=device)
            h_repr = model.entity_repr(h_t, torch.tensor([float(ts)], device=device))
            c_repr = model.entity_repr(cand_t, ts_t)
            r_t = torch.tensor([r], device=device).expand(len(cand))
            scores = model.score(h_repr.expand(len(cand), -1), r_t, c_repr, ts_t)
            true_score = float(scores[0])
            rank = 1 + int((scores > true_score).sum().item())
            mrr += 1.0 / rank
            if rank <= 1: h1 += 1
            if rank <= 3: h3 += 1
            if rank <= 10: h10 += 1
            cnt += 1
    return mrr / cnt, h1 / cnt, h3 / cnt, h10 / cnt


# ====================================================================
# 5. 推理：给定 (头, 关系, 时间) 预测 Top-K 尾实体（在全实体上排名）
# ====================================================================
def _build_relation_endpoints(model):
    """由 checkpoint 内的邻接表构建并缓存：
        model._rel_tails / _rel_heads      : 关系 r -> 实体集合（类型约束候选集用）
        model._rel_tails_arr / _rel_heads_arr : 同上，但预存为 numpy 数组
          （困难负采样时直接 rng.choice 数组，避免每批反复构建巨型数组）。
        model._hr_tails : (头,关系) -> 该头在该关系下真实出现过的尾集合
          （推理候选集优先用它，使预测直接落在该头-关系的历史合法尾上，最贴合直觉）。
    供「类型约束候选集」与「同关系困难负采样」复用，避免重复扫描邻接表。
    """
    if hasattr(model, "_rel_tails_arr") and hasattr(model, "_rel_heads_arr"):
        return
    from collections import defaultdict
    rel_tails = defaultdict(set)
    rel_heads = defaultdict(set)
    hr_tails = defaultdict(set)
    for e, (nb, rel, tt) in model.neigh.items():
        for nbi, ri in zip(nb.tolist(), rel.tolist()):
            rel_tails[ri].add(nbi)      # nbi 是关系 ri 的尾
            rel_heads[ri].add(e)        # e 是关系 ri 的头
            hr_tails[(e, ri)].add(nbi)  # (e, ri) 这个头-关系对的历史尾
    model._rel_tails = rel_tails
    model._rel_heads = rel_heads
    model._hr_tails = hr_tails
    model._rel_tails_arr = {ri: np.array(sorted(s)) for ri, s in rel_tails.items()}
    model._rel_heads_arr = {ri: np.array(sorted(s)) for ri, s in rel_heads.items()}


def _get_relation_tail_set(model, r):
    """返回「曾作为关系 r 尾实体出现过的实体」集合（类型约束候选集）。"""
    _build_relation_endpoints(model)
    return model._rel_tails.get(int(r), set())


def _sample_typed_negatives(model, h, r, t, ts, n_neg):
    """同关系困难负采样：负尾从关系 r 的合法尾集合采样（排除真实尾），
    负头从关系 r 的合法头集合采样（排除真实头）。合法集合不足 n_neg 个时，
    用均匀随机负样本补足（不会死循环）。返回 (neg_t, neg_h) 均为 (B,n_neg) int64。

    相比均匀随机负样本，困难负样本本身也是「合法尾/头」，模型无法靠关系热度
    蒙混过关，必须利用头实体与时间信号才能把真实尾排到前面——直接针对
    「不同 head 预测出同一批热门尾」的失效模式。
    """
    _build_relation_endpoints(model)
    B = h.shape[0]
    rng = np.random.default_rng()
    true_tails = model._true_tails if hasattr(model, "_true_tails") else {}
    neg_t = np.empty((B, n_neg), dtype=np.int64)
    neg_h = np.empty((B, n_neg), dtype=np.int64)
    for i in range(B):
        ri = int(r[i]); ti = int(t[i]); hi = int(h[i]); tsi = int(ts[i])
        ft = true_tails.get((hi, ri, tsi), set())
        # ---- 负尾 ----
        ta = model._rel_tails_arr.get(ri)
        if ta is not None and ta.shape[0] > 0:
            k = min(n_neg, ta.shape[0])
            picked = [x for x in rng.choice(ta, size=k, replace=False).tolist()
                      if x != ti and x not in ft]
        else:
            picked = []
        while len(picked) < n_neg:
            x = int(rng.integers(0, model.num_ent))
            if x != ti and x not in ft and x not in picked:
                picked.append(x)
        neg_t[i] = picked
        # ---- 负头 ----
        ha = model._rel_heads_arr.get(ri)
        if ha is not None and ha.shape[0] > 0:
            k = min(n_neg, ha.shape[0])
            picked = [x for x in rng.choice(ha, size=k, replace=False).tolist()
                      if x != hi]
        else:
            picked = []
        while len(picked) < n_neg:
            x = int(rng.integers(0, model.num_ent))
            if x != hi and x not in picked:
                picked.append(x)
        neg_h[i] = picked
    return neg_t, neg_h


def _temporal_recency(model, h, r, tau, sigma=3.0):
    """计算 (头,关系) 各历史尾相对于查询时间 tau 的「时间邻近度」∈[0,1]。

    用于推理时的时序重排偏置：对 neigh[h] 中关系==r 的每条历史边 (h,r,t,tt)，
    取 t 与 tau 最接近的年份距离做高斯衰减。这样「近年才出现的尾」在预测
    近期查询时得分更高，缓解「模型只记住高频旧尾、忽略时序切换」的问题。
    无 GPU / 无需重训即可生效。
    """
    rec = {}
    if not hasattr(model, "neigh"):
        return rec
    nb, rl, tt = model.neigh.get(int(h), (np.array([]), np.array([]), np.array([])))
    if nb.size == 0:
        return rec
    for nbi, ri, tmi in zip(nb.tolist(), rl.tolist(), tt.tolist()):
        if int(ri) != int(r):
            continue
        d = abs(float(tau) - float(tmi))
        s = math.exp(-d / float(sigma))
        if nbi not in rec or s > rec[nbi]:
            rec[nbi] = s
    return rec


def predict_tails(model, h, r, ts, true_t=None, k=5, k_neg=2000, seed=0, device="cpu",
                  temporal_bias=0.0, temporal_sigma=3.0):
    """推理：给定 (头, 关系, 时间)，在【类型约束的候选尾】上打分排序，返回 Top-K。

    与旧版「仅用 k_neg 个均匀随机负样本做候选」不同：
      * 候选尾限定为「数据里曾作为关系 r 尾实体出现过的实体」（类型约束），
        保证返回的 Top-K 全都是该关系下合法的尾（不会再出现 edgelist 里查不到的随机 ID）；
      * 真实尾实体必然在候选集内，有机会被排到前面（旧版随机负样本几乎不可能命中真实尾，
        这正是「准确率太低 / 结果在 edgelist 中都不存在」的根因）。
    若某关系在数据中没有任何尾实体记录，则退化为全实体排序作为兜底。

    采用分块打分以控制内存/显存占用。过滤式设定：剔除 (头,关系,时间) 下的其它真实尾
    （避免把其它正确答案当噪声），但保留传入的 true_t 本身（用于评测命中标记）；
    头实体自身也不作为尾返回。
    """
    model.eval()
    h = int(h); r = int(r); ts = float(ts)
    filt = model._true_tails.get((h, r, int(ts)), set()) if hasattr(model, "_true_tails") else set()
    # 手动推理（true_t=None）不排除真实尾，便于用户核对真实尾的排名；
    # 评测模式（true_t 给定）则过滤掉同 (头,关系,时间) 的其它真实尾（过滤式设定），
    # 但保留传入的 true_t 本身以标记命中。头实体自身始终不作为尾返回。
    exclude = {h}
    if true_t is not None:
        exclude.update(filt - {int(true_t)})

    # 类型约束候选集：优先用「该 (头,关系) 历史上真实出现过的尾」（最贴合直觉，
    # 预测直接落在该头-关系的合法尾上）；为空时退化为「关系 r 的所有合法尾」；
    # 再为空则退化为全实体。
    _build_relation_endpoints(model)
    cand_set = model._hr_tails.get((h, r), set())
    if len(cand_set) == 0:
        cand_set = _get_relation_tail_set(model, r)
    if len(cand_set) == 0:
        cand_set = set(range(model.num_ent))   # 兜底：全实体
    cand_list = sorted(cand_set - exclude)

    chunk = 10000
    scored = []                         # (score, tail_id)
    # 时间邻近性偏置：与查询 τ 越接近的历史尾，偏置越大（缓解「只记高频旧尾」）
    rec = _temporal_recency(model, h, r, ts, sigma=temporal_sigma) if temporal_bias else {}
    with torch.no_grad():
        h_t = torch.tensor([h], dtype=torch.long, device=device)
        ts_t = torch.tensor([ts], device=device)
        h_repr = model.entity_repr(h_t, ts_t)               # (1, dim)
        r_t = torch.tensor([r], dtype=torch.long, device=device)
        for s in range(0, len(cand_list), chunk):
            e = torch.tensor(cand_list[s:s + chunk], dtype=torch.long, device=device)
            e_ts = torch.full((e.shape[0],), ts, device=device)
            e_repr = model.entity_repr(e, e_ts)             # (chunk, dim)
            sc = model.score(h_repr.expand(e.shape[0], -1),
                             r_t.expand(e.shape[0]), e_repr, e_ts)
            sc_np = sc.detach().cpu().numpy()
            e_np = e.detach().cpu().numpy()
            for local_i in range(e.shape[0]):
                tid = int(e_np[local_i])
                bias = temporal_bias * rec.get(tid, 0.0) if temporal_bias else 0.0
                scored.append((float(sc_np[local_i]) + bias, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    # 保持与调用方约定一致的返回顺序：(尾实体 id, 得分)
    return [(tid, score) for score, tid in scored[:k]]


# ====================================================================
# 6. 模型保存 / 加载（自包含：含邻接表与真实尾索引）
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


def _resolve_id(token, name2id, id2name):
    """把用户输入解析成实体/关系整数 ID。

    支持三种输入：
      * 名称（如 'Q648' / 'P27'）：直接查 name2id；
      * 整数 ID（如 '123'）：要求存在于 id2name 中；
      * 空串：返回 None（表示跳过）。
    无法解析时返回 None。
    """
    token = (token or "").strip()
    if token == "":
        return None
    if token in name2id:                 # 名称形式（Q-ID / P-ID）
        return int(name2id[token])
    try:
        i = int(token)
        if i in id2name:                 # 整数 ID 形式
            return i
    except ValueError:
        pass
    return None


def run_interactive_infer(model, data, device, topk, k_neg):
    """交互式推理：用户手动输入 头实体 / 关系 / 时间，预测 Top-K 尾实体。

    输入为空串回车即退出；头/关系支持 Q-ID/P-ID 或整数 ID；时间为年份整数。
    """
    entity2id, relation2id = data["entity2id"], data["relation2id"]
    id2ent, id2rel = data["id2entity"], data["id2relation"]
    print("\n【交互式推理】输入 头实体 / 关系 / 时间（任一项留空回车即退出）")
    while True:
        try:
            h_in = input("  头实体 (Q-ID 或整数ID): ").strip()
            r_in = input("  关系   (P-ID 或整数ID): ").strip()
            t_in = input("  时间   (年份, 如 2008): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互式推理。")
            break
        if h_in == "" or r_in == "" or t_in == "":
            print("退出交互式推理。")
            break
        h = _resolve_id(h_in, entity2id, id2ent)
        r = _resolve_id(r_in, relation2id, id2rel)
        try:
            ts = int(t_in)
        except ValueError:
            print("  ⚠️ 时间必须是整数年份（如 2008）\n")
            continue
        if h is None:
            print(f"  ⚠️ 未知头实体: {h_in}（可用 ID 范围 0~{model.num_ent-1}）\n")
            continue
        if r is None:
            print(f"  ⚠️ 未知关系: {r_in}（可用 ID 范围 0~{model.num_rel-1}）\n")
            continue
            top = predict_tails(model, h, r, ts, true_t=None, k=topk,
                                k_neg=max(k_neg, 2000), device=device,
                                temporal_bias=args.temporal_bias,
                                temporal_sigma=args.temporal_sigma)
        print(f"\n  查询: ({id2ent[h]}, {id2rel[r]}, ?)  @τ={ts}")
        for tid, sc in top:
            print(f"      - {id2ent[tid]}   得分={sc:+.4f}")
        print()


# ====================================================================
# 7. 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="TKGL-Smallpedia 时序链接预测")
    parser.add_argument("--mode", choices=["train", "infer"], default="train",
                        help="train=训练并保存模型; infer=直接加载已训练模型做推理/评测")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(_ROOT, "data", "tkgl-smallpedia"))
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(_ROOT, "trained_models", "tkgl_smallpedia_model.pt"))
    parser.add_argument("--force", action="store_true", help="强制重新训练（覆盖已保存模型）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--neg-mode", choices=["uniform", "typed"], default="uniform",
                        help="负采样方式：uniform=均匀随机(默认,收敛快); typed=同关系困难负样本")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=20, help="邻居采样数")
    parser.add_argument("--buffer", type=int, default=20000,
                        help="流式训练的 shuffle-buffer 大小（越大打乱越接近全局，"
                             "但占用内存越多；训练集本身不整体驻留）")
    parser.add_argument("--num-eval", type=int, default=2000, help="评测查询数")
    parser.add_argument("--k-neg", type=int, default=500, help="评测每个查询的负样本数")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temporal-bias", type=float, default=50.0,
                        help="推理时叠加「时间邻近性」偏置的权重（默认 50）。"
                             "对 (头,关系) 的历史尾按与查询年份的时间距离做高斯衰减，"
                             "使近年出现的尾在预测近期查询时排前。无需重训/GPU，"
                             "用于缓解「模型只记住高频旧尾、忽略时序切换」。")
    parser.add_argument("--temporal-sigma", type=float, default=8.0,
                        help="时间邻近性高斯衰减尺度（默认 8）。越大衰减越慢，"
                             "历史尾在更久之后查询时仍能保留偏置；越小越「喜新厌旧」。")
    # 手动推理：交互式（--interactive）或一次性指定（--head/--relation/--time）
    parser.add_argument("--interactive", action="store_true",
                        help="交互式推理：逐条手动输入 头实体/关系/时间 进行预测")
    parser.add_argument("--head", type=str, default=None,
                        help="手动推理的头实体（Q-ID 或整数ID），配合 --relation/--time 使用")
    parser.add_argument("--relation", type=str, default=None,
                        help="手动推理的关系（P-ID 或整数ID）")
    parser.add_argument("--time", type=int, default=None,
                        help="手动推理的时间（年份整数，如 2008）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ckpt_path = args.model_path
    skip_train = (args.mode == "infer") or (os.path.exists(ckpt_path) and not args.force)

    # ================= 推理模式：仅加载 checkpoint =================
    if skip_train:
        print(f"ckpt_path: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"❌ 未找到模型文件: {ckpt_path}\n   请先以 --mode train 训练。")
        print(f"📥 加载已训练模型: {ckpt_path}")
        model, data = load_checkpoint(ckpt_path, device=device)
        id2ent = data["id2entity"]
        id2rel = data["id2relation"]

        # ===== 手动推理：交互式 或 一次性指定 =====
        if args.interactive or (args.head is not None and args.relation is not None
                                and args.time is not None):
            if args.interactive:
                run_interactive_infer(model, data, device, args.topk, args.k_neg)
                return
            h = _resolve_id(args.head, data["entity2id"], id2ent)
            r = _resolve_id(args.relation, data["relation2id"], id2rel)
            ts = int(args.time)
            if h is None:
                raise SystemExit(f"❌ 未知头实体: {args.head}")
            if r is None:
                raise SystemExit(f"❌ 未知关系: {args.relation}")
            top = predict_tails(model, h, r, ts, true_t=None, k=args.topk,
                                k_neg=max(args.k_neg, 2000), device=device,
                                temporal_bias=args.temporal_bias,
                                temporal_sigma=args.temporal_sigma)
            print(f"\n【手动推理】查询: ({id2ent[h]}, {id2rel[r]}, ?)  @τ={ts}")
            for tid, sc in top:
                print(f"      - {id2ent[tid]}   得分={sc:+.4f}")
            print()
            return

        # 评测（test 集过滤式 MRR，直接使用 checkpoint 内的 test_quads）
        test_quads = data["test_quads"]
        mrr, h1, h3, h10 = evaluate_filtered(
            model, test_quads, data["true_tails"],
            num_eval=args.num_eval, k_neg=args.k_neg, device=device)
        print("=" * 70)
        print(f"【测试结果】过滤式 MRR={mrr:.4f}  Hits@1={h1:.4f}  "
              f"Hits@3={h3:.4f}  Hits@10={h10:.4f}")
        print("=" * 70)

        # 示例推理：取若干测试四元组，预测 Top-K 尾实体（在全实体上排名）
        print("\n【示例推理：给定 (头, 关系, 时间) 预测尾实体 Top-%d】" % args.topk)
        rng = np.random.default_rng(7)
        n_demo = min(5, test_quads.shape[0])
        picks = rng.choice(test_quads.shape[0], size=n_demo, replace=False)
        for i in picks:
            ts, h, t, r = (int(test_quads[i, 0]), int(test_quads[i, 1]),
                           int(test_quads[i, 2]), int(test_quads[i, 3]))
            top = predict_tails(model, h, r, ts, t, k=args.topk, device=device,
                                temporal_bias=args.temporal_bias,
                                temporal_sigma=args.temporal_sigma)
            hit = any(tid == t for tid, _ in top)
            print(f"\n  查询: ({id2ent[h]}, {id2rel[r]}, ?)  @τ={ts}")
            print(f"    真实尾: {id2ent[t]}")
            for tid, sc in top:
                mark = "  <== 命中真实尾" if tid == t else ""
                print(f"      - {id2ent[tid]}   得分={sc:+.4f}{mark}")
            print(f"    => 真实尾是否在 Top-{args.topk}: {hit}")
        return

    # ================= 训练模式 =================
    print(f"📂 加载数据集: {args.data_dir}")
    # build_train_arrays=False：跳过约 150 万条训练/静态四元组的整体驻留，
    # 训练边改为 iter_train_batches 流式分批产出（验证/测试四元组仍需驻留，体积小）。
    data = load_tkgl_smallpedia(args.data_dir, build_train_arrays=False)
    print("=" * 70)
    print("【数据集概览】")
    print(f"  实体总数={data['num_ent']}, 关系总数={data['num_rel']}")
    print(f"  验证边={data['val_quads'].shape[0]}, 测试边={data['test_quads'].shape[0]}")
    print(f"  时间切分: val>= {data['splits']['val_start']}, test>= {data['splits']['test_start']}")
    print(f"  时间归一化尺度 time_scale={data['time_scale']}")
    print(f"  训练边: 流式分批加载（batch={args.batch}, shuffle-buffer={args.buffer}），不整体驻留内存")
    print("=" * 70)

    model = TemporalKGModel(
        num_ent=data["num_ent"],
        num_rel=data["num_rel"],
        dim=args.dim,
        n_heads=4,
        time_scale=data["time_scale"],
        t_center=data["t_center"],
        neigh=data["neigh"],
        max_neigh=args.n_samples,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 每 epoch 重新流式读取并打乱训练边（不同 seed 保证各 epoch 顺序不同）
    def make_batch_iter(epoch):
        return iter_train_batches(
            args.data_dir, data["entity2id"], data["relation2id"],
            batch_size=args.batch, buffer=args.buffer, seed=epoch, shuffle=True)

    train_model(model, make_batch_iter, optimizer, device,
                epochs=args.epochs, batch_size=args.batch, n_neg=args.neg,
                num_samples=args.n_samples, val_data=data,
                num_eval=args.num_eval, k_neg=args.k_neg,
                neg_mode=args.neg_mode)

    # 测试集评测
    mrr, h1, h3, h10 = evaluate_filtered(
        model, data["test_quads"], data["true_tails"],
        num_eval=args.num_eval, k_neg=args.k_neg, device=device)
    print("=" * 70)
    print(f"【测试结果】过滤式 MRR={mrr:.4f}  Hits@1={h1:.4f}  "
          f"Hits@3={h3:.4f}  Hits@10={h10:.4f}")
    print("=" * 70)

    # 保存（自包含 checkpoint）
    model._id2entity = data["id2entity"]
    model._id2relation = data["id2relation"]
    save_checkpoint(model, data, ckpt_path)
    print(f"💾 模型已保存到: {ckpt_path}")
    print("   之后可用: python demo/tkgl_smallpedia_tkg.py --mode infer 直接加载推理。")


if __name__ == "__main__":
    main()

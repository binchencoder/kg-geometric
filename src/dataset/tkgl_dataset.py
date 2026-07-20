# -*- coding: utf-8 -*-
"""
src/dataset/tkgl_dataset.py
=====================================================================
TKGL-Smallpedia 数据集加载器（纯 Python，不依赖 torch）

把原始的
    tkgl-smallpedia_edgelist.csv          (时序边: ts,head,tail,relation_type)
    tkgl-smallpedia_static_edgelist.csv   (静态边: head,tail,relation_type)
转换为「整数四元组」形式，并构建 TGAT 训练所需的邻接表 / 时间切分 /
过滤式评测所需的真实尾实体索引。

设计要点：
  * 实体 / 关系 的 id 映射由本加载器确定性地生成（首现顺序：先扫描时序边，
    再扫描静态边），不依赖外部 pkl 的隐藏映射 —— 因此本管线完全自包含、可复现。
  * 时间切分沿用数据集本身的时间分布（见下方 val_start / test_start 默认值）：
        训练  ts <= 1997
        验证  1998 <= ts <= 2007
        测试  ts >= 2008
    （这与 tkgl-smallpedia_{val,test}_ns.pkl 中查询的时间戳范围一致：
     val 1998~2007，test 2008~2024。）
  * 静态边没有时间戳，给一个「哨兵时间」(static_sentinel，默认 1899)，
    保证它在任意时序查询时都可用作邻居上下文（tt <= qt 恒成立）。
  * 邻接表 neigh[eid] = (nb[], rel[], tt[]) 用 numpy 数组存储，便于在
    entity_repr 中做向量化的「仅聚合历史邻居(tt <= qt)」过滤与采样。
  * 为支持归纳式聚合，邻接表同时加入「反向边」(同关系 id，无向聚合)，
    使信息在图上双向流动。
  * true_tails[(h, r, ts)] 收集全量四元组中同一 (头,关系,时间) 下的所有真实尾，
    供过滤式 MRR 评测剔除「其它真实尾」使用。

原实现位于 data/tkgl-smallpedia/loader.py，此处抽取为独立模块以便复用
（data/tkgl-smallpedia/loader.py 现已改为从本模块 re-export，保持向后兼容）。
"""

import csv
import os
from collections import defaultdict

import numpy as np

# 默认时间切分点（与数据集 pkl 负样本的时间戳范围一致）
DEFAULT_VAL_START = 1998
DEFAULT_TEST_START = 2008
DEFAULT_STATIC_SENTINEL = 1899  # 小于任何时序时间戳，使静态边始终可作为历史邻居


def iter_train_batches(data_dir,
                        entity2id,
                        relation2id,
                        batch_size=1024,
                        val_start=DEFAULT_VAL_START,
                        test_start=DEFAULT_TEST_START,
                        static_sentinel=DEFAULT_STATIC_SENTINEL,
                        buffer=20000,
                        seed=0,
                        shuffle=True):
    """【分批流式】生成训练批次，避免一次性把约 150 万条训练边全部驻留内存。

    每个批次是 np.ndarray (B, 4) int64: [ts, h, t, r]，
    包含：时序训练边(ts < val_start) 与 静态边(哨兵时间)。

    采用「shuffle buffer（洗牌缓冲）」近似全局打乱：
        维护一个固定大小的缓冲，随机弹出一个样本、再从数据流随机补入一个，
        从而在「逐行流式读取」条件下获得接近全局打乱的效果，且无需把全量
        数据放进内存。每个 epoch 重新调用本函数即可重新流式读取并打乱。

    注意：本函数需要预先构建好的 entity2id / relation2id（由 load_tkgl_smallpedia
    一次性流式构建，开销很小），因为字符串 Q-ID/P-ID 需先映射为整数才能组批。
    """
    import csv as _csv
    edgelist = os.path.join(data_dir, "tkgl-smallpedia_edgelist.csv")
    staticlist = os.path.join(data_dir, "tkgl-smallpedia_static_edgelist.csv")

    def row_stream():
        # 先产出静态边（哨兵时间），再产出时序训练边；两者统一进入 shuffle buffer
        with open(staticlist, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                h = entity2id[row["head"]]
                t = entity2id[row["tail"]]
                r = relation2id[row["relation_type"]]
                yield (static_sentinel, h, t, r)
        with open(edgelist, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                ts = int(row["ts"])
                if ts >= val_start:
                    continue  # 只取训练边（ts < val_start）
                h = entity2id[row["head"]]
                t = entity2id[row["tail"]]
                r = relation2id[row["relation_type"]]
                yield (ts, h, t, r)

    rng = np.random.default_rng(seed)
    stream = row_stream()
    buf = []
    # 预热：填满 buffer
    for _ in range(buffer):
        try:
            buf.append(next(stream))
        except StopIteration:
            break

    batch = []
    while buf:
        if shuffle:
            i = int(rng.integers(0, len(buf)))
            item = buf[i]
            try:
                buf[i] = next(stream)
            except StopIteration:
                buf.pop(i)
        else:
            item = buf.pop(0)
        batch.append(item)
        if len(batch) >= batch_size:
            yield np.array(batch, dtype=np.int64)
            batch = []
    if batch:  # 收尾不足一个 batch 的尾巴
        yield np.array(batch, dtype=np.int64)


def load_tkgl_smallpedia(data_dir,
                         val_start=DEFAULT_VAL_START,
                         test_start=DEFAULT_TEST_START,
                         static_sentinel=DEFAULT_STATIC_SENTINEL,
                         build_train_arrays=True):
    """加载 TKGL-Smallpedia 数据集。

    返回 dict，含：
        entity2id / id2entity / relation2id / id2relation
        num_ent / num_rel
        train_quads / val_quads / test_quads : np.ndarray (N,4) int  [h, r, t, ts]
        static_quads : np.ndarray (M,4) int  [h, r, t, static_sentinel]
        neigh : dict eid -> (nb np.int64, rel np.int64, tt np.float32)
        true_tails : dict (h, r, ts) -> set of tail ids
        time_scale : float
        splits : dict
    """
    edgelist = os.path.join(data_dir, "tkgl-smallpedia_edgelist.csv")
    staticlist = os.path.join(data_dir, "tkgl-smallpedia_static_edgelist.csv")

    # ---------- 1) 构建 id 映射（首现顺序：时序 -> 静态） ----------
    entity2id, relation2id = {}, {}
    id2entity, id2relation = [], []

    def eid(q):
        if q not in entity2id:
            entity2id[q] = len(id2entity)
            id2entity.append(q)
        return entity2id[q]

    def rid(p):
        if p not in relation2id:
            relation2id[p] = len(id2relation)
            id2relation.append(p)
        return relation2id[p]

    # 时序边（带时间戳）
    temporal = []  # (ts, h, t, r)  —— 先不转 int，先扫一遍建映射
    with open(edgelist, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = int(row["ts"])
            h = eid(row["head"]); t = eid(row["tail"]); r = rid(row["relation_type"])
            temporal.append((ts, h, t, r))

    # 静态边（哨兵时间）
    static = []
    with open(staticlist, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = eid(row["head"]); t = eid(row["tail"]); r = rid(row["relation_type"])
            static.append((static_sentinel, h, t, r))

    num_ent = len(id2entity)
    num_rel = len(id2relation)

    # ---------- 2) 切分时序边 ----------
    train_q, val_q, test_q = [], [], []
    for ts, h, t, r in temporal:
        if ts < val_start:
            train_q.append((ts, h, t, r))
        elif ts < test_start:
            val_q.append((ts, h, t, r))
        else:
            test_q.append((ts, h, t, r))

    def to_arr(qs):
        if not qs:
            return np.zeros((0, 4), dtype=np.int64)
        return np.array(qs, dtype=np.int64)  # (N,4): ts, h, t, r

    # 验证/测试四元组始终构建（评测与 checkpoint 需要，体积小：各约 8 万条）。
    # 训练边 + 静态边体积大（约 150 万条），仅当 build_train_arrays=True 时驻留内存；
    # 使用流式分批训练（iter_train_batches）时可设为 False 以显著降低峰值内存。
    train_quads = to_arr(train_q) if build_train_arrays else np.zeros((0, 4), dtype=np.int64)
    val_quads = to_arr(val_q)
    test_quads = to_arr(test_q)
    static_quads = to_arr(static) if build_train_arrays else np.zeros((0, 4), dtype=np.int64)

    # ---------- 3) 构建邻接表（含反向边，供 TGAT 无向聚合） ----------
    # 用列表收集，再转 numpy 数组
    nb_dict = defaultdict(list)
    rel_dict = defaultdict(list)
    tt_dict = defaultdict(list)

    def add_edge(h, r, t, tt):
        nb_dict[h].append(t)
        rel_dict[h].append(r)
        tt_dict[h].append(tt)
        # 反向边：同关系 id（无向聚合）
        nb_dict[t].append(h)
        rel_dict[t].append(r)
        tt_dict[t].append(tt)

    # 训练 / 验证 / 测试 时序边都加入邻接表（聚合时按 tt<=qt 时间过滤，
    # 因此测试边的「未来」部分不会泄漏进更早的查询 —— 时间过滤保证无泄漏）
    for ts, h, t, r in temporal:
        add_edge(h, r, t, float(ts))
    for ts, h, t, r in static:
        add_edge(h, r, t, float(ts))

    neigh = {}
    for e in nb_dict:
        neigh[e] = (
            np.array(nb_dict[e], dtype=np.int64),
            np.array(rel_dict[e], dtype=np.int64),
            np.array(tt_dict[e], dtype=np.float32),
        )

    # ---------- 4) 真实尾实体索引（过滤式评测用） ----------
    true_tails = defaultdict(set)
    for ts, h, t, r in temporal:
        true_tails[(h, r, ts)].add(t)

    # ---------- 5) 时间归一化尺度 ----------
    # 关键修正：原 time_scale = max(|ts|) ≈ 2013，会把所有年份归一化到 [0.98,1.0]，
    # 使 TimeEncoder 的周期性项几乎不随年份变化、时间信号丢失。
    # 改为：把时间居中并缩放到约 [-5,5]（每一年对应约 0.33 的输入变化），
    # 周期性时间项即可区分不同年份；t_center 为时间中点，供模型居中使用。
    all_ts = [ts for ts, _, _, _ in temporal]
    if all_ts:
        tmin, tmax = float(min(all_ts)), float(max(all_ts))
        span = (tmax - tmin) or 1.0
        time_scale = span / 10.0
        t_center = (tmin + tmax) / 2.0
    else:
        time_scale, t_center = 1.0, 0.0

    return {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "relation2id": relation2id,
        "id2relation": id2relation,
        "num_ent": num_ent,
        "num_rel": num_rel,
        "train_quads": train_quads,
        "val_quads": val_quads,
        "test_quads": test_quads,
        "static_quads": static_quads,
        "neigh": neigh,
        "true_tails": true_tails,
        "time_scale": time_scale,
        "t_center": t_center,
        "splits": {
            "val_start": val_start,
            "test_start": test_start,
            "static_sentinel": static_sentinel,
        },
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # 本模块位于 src/dataset/，数据集在 <仓库根>/data/tkgl-smallpedia/ 下；
    # 直接运行时用相对路径回退到仓库根目录的 data 目录
    # （src/dataset 距仓库根两层，故取两次 dirname）。
    _ROOT = os.path.dirname(os.path.dirname(here))
    d = load_tkgl_smallpedia(os.path.join(_ROOT, "data", "tkgl-smallpedia"))
    print("num_ent =", d["num_ent"])
    print("num_rel =", d["num_rel"])
    print("train_quads =", d["train_quads"].shape)
    print("val_quads   =", d["val_quads"].shape)
    print("test_quads  =", d["test_quads"].shape)
    print("static_quads=", d["static_quads"].shape)
    print("time_scale  =", d["time_scale"])
    print("neigh size  =", len(d["neigh"]))
    # 度数统计
    degs = [len(v[0]) for v in d["neigh"].values()]
    print("avg degree  =", sum(degs) / len(degs))
    print("max degree  =", max(degs))

# -*- coding: utf-8 -*-
"""
examples/employee_tkg_link_prediction.py
=====================================================================
基于「中文时序知识图谱 (TKG)」的员工 - 公司雇佣关系 链接预测 (demo)
=====================================================================

本文件复用 demo/temporal_kg_link_prediction.py 中已实现的通用时序链接预测模型
(TGAT 时序图注意力编码器 + 时间感知 DistMult 解码器，时间作为「关系」的属性)，
并在此之上构建「员工 - 公司雇佣关系」这一具体领域的本体 (Ontology) 与足量的
模拟数据，完成 训练 + 推理 全流程。

---------------------------------------------------------------------
一、本体 (Ontology) —— 实体类型 / 关系类型
---------------------------------------------------------------------
实体类型 (ENTITY_TYPES):
    人员    (Person，原英文名为 Person)
    组织    (Organization，公司 / 高校)
    地点    (Location，城市)
    职位    (Position)

关系类型 (RELATIONS):  (关系名, 头类型, 尾类型, 是否对称)
    任职于   (人员, 组织, interval)            某人在某公司任职
    担任职位 (人员, 职位, interval)            某人担任某职位
    收购     (组织, 组织, instant)            某公司收购另一公司
    投资     (组织, 组织, instant_quarter)     某公司投资另一公司(按季度)
    位于     (组织, 地点, interval)            某公司总部位于某城市
    接任     (人员, 人员, instant)            某人接任另一人的职位
    合作     (组织, 组织, interval，对称)       两公司合作
    毕业于   (人员, 组织, instant)            某人毕业于某组织(高校)
    成立于   (组织, 地点, instant)            某公司成立于某城市

---------------------------------------------------------------------
二、时间如何作为「关系」的属性
---------------------------------------------------------------------
每条关系实例都带一个连续时间戳 τ（这里用「年份」作连续坐标，如 2018.5）：
    * interval 类关系（任职于 / 担任职位 / 位于 / 合作）
      用其「生效起始时间」作为 τ；
    * instant / instant_quarter 类关系（收购 / 投资 / 接任 /
      毕业于 / 成立于）用事件发生时刻作为 τ。
因此同样的 (头, 关系, 尾) 在不同时间的表示/得分不同 —— 即「时间感知」。

---------------------------------------------------------------------
三、归纳式 (inductive)
---------------------------------------------------------------------
部分人员「入图时间」(毕业/入职年份) 被故意设在训练切分点 cutoff 之后，
这些实体在训练阶段完全不可见，用于验证模型对新实体的预测能力。

---------------------------------------------------------------------
运行:
    python examples/employee_tkg_link_prediction.py
（也可加 --epochs / --dim / --batch / --neg / --lr 等参数）
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

# 复用通用模型：Entity / Quad 数据结构，以及 TGAT + 时间感知 DistMult 全套逻辑。
# 兼容两种运行方式：直接 `python demo/xxx.py`（同目录导入）或从项目根目录运行。
try:
    from temporal_kg_link_prediction import (
        Entity, Quad, TemporalKGModel, sample_negative, train,
        evaluate, predict_tails, candidate_entities, fmt_quad,
    )
except ModuleNotFoundError:  # 从项目根目录 `python examples/employee_tkg_link_prediction.py`
    from demo.temporal_kg_link_prediction import (
        Entity, Quad, TemporalKGModel, sample_negative, train,
        evaluate, predict_tails, candidate_entities, fmt_quad,
    )


# ====================================================================
# 1. 本体定义 (Ontology)
# ====================================================================
ENTITY_TYPES = ["人员", "组织", "地点", "职位"]

ENTITIES = {
    "人员": ["张伟", "李明", "王芳", "陈磊", "刘洋", "赵静", "孙强", "周杰", "吴敏", "郑浩",
            "黄丽", "林涛", "何雪", "马超", "罗琳", "梁军", "宋佳", "唐杰", "许峰", "邓超",
            "韩梅", "冯刚", "曹颖", "彭勇", "董洁"],
    "组织": ["腾讯科技", "阿里巴巴", "字节跳动", "百度", "京东", "美团", "拼多多", "华为",
            "小米集团", "网易", "滴滴出行", "快手", "哔哩哔哩", "携程旅行", "蔚来汽车",
            "理想汽车", "小鹏汽车", "大疆创新", "商汤科技", "清华大学"],
    "地点": ["北京市", "上海市", "深圳市", "杭州市", "广州市", "成都市", "武汉市", "南京市", "西安市", "苏州市"],
    "职位": ["首席执行官", "首席技术官", "技术总监", "产品副总裁", "首席科学家"],
}

# 关系类型：(关系名, 头实体类型, 尾实体类型, 是否对称)
RELATIONS = [
    ("任职于",     "人员", "组织", False),
    ("担任职位",   "人员", "职位", False),
    ("收购",       "组织", "组织", False),
    ("投资",       "组织", "组织", False),
    ("位于",       "组织", "地点", False),
    ("接任",       "人员", "人员", False),
    ("合作",       "组织", "组织", True),
    ("毕业于",     "人员", "组织", False),
    ("成立于",     "组织", "地点", False),
]

# 各公司的「成立年份」与「总部城市」（用于构造更真实的数据）
ORG_FOUNDED = {
    "腾讯科技": 1998, "阿里巴巴": 1999, "字节跳动": 2012, "百度": 2000, "京东": 1998,
    "美团": 2010, "拼多多": 2015, "华为": 1987, "小米集团": 2010, "网易": 1997,
    "滴滴出行": 2012, "快手": 2011, "哔哩哔哩": 2009, "携程旅行": 1999, "蔚来汽车": 2014,
    "理想汽车": 2015, "小鹏汽车": 2014, "大疆创新": 2006, "商汤科技": 2014, "清华大学": 1911,
}
ORG_HQ = {
    "腾讯科技": "深圳市", "阿里巴巴": "杭州市", "字节跳动": "北京市", "百度": "北京市",
    "京东": "北京市", "美团": "北京市", "拼多多": "上海市", "华为": "深圳市",
    "小米集团": "北京市", "网易": "杭州市", "滴滴出行": "北京市", "快手": "北京市",
    "哔哩哔哩": "上海市", "携程旅行": "上海市", "蔚来汽车": "上海市", "理想汽车": "北京市",
    "小鹏汽车": "广州市", "大疆创新": "深圳市", "商汤科技": "上海市", "清华大学": "北京市",
}


def build_relation_vocab(relations):
    """把关系名映射成 id，并为非对称关系自动构造反向关系(用于双向聚合)。

    返回 rel2id / id2rel / inv / rel_head_types / rel_tail_types，
    语义同 demo/temporal_kg_link_prediction.build_relation_vocab，但此处参数化，
    以便套用本文件的员工 - 公司本体。
    """
    rel2id, id2rel, inv = {}, [], {}
    rel_head_types, rel_tail_types = defaultdict(set), defaultdict(set)
    for name, h, t, sym in relations:
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
            rel_head_types[ivid].add(t)
            rel_tail_types[ivid].add(h)
    return rel2id, id2rel, inv, dict(rel_head_types), dict(rel_tail_types)


# ====================================================================
# 2. 模拟数据生成（足量、含时间、含归纳式新实体）
# ====================================================================
def generate_employee_dataset(seed=42, cutoff=2020.0, horizon_start=2000.0,
                              horizon_end=2025.0):
    """生成「员工 - 公司雇佣关系」时序知识图谱。

    设计要点：
    * 时间作为关系属性：每条边都是 Quad(头, 关系, 尾, 时间)，时间即关系的生效/发生时刻。
    * 足量：25 人 × 多段职业经历 + 20 家组织间的大量收购/投资/合作，共生成约 1300 条有向边（含正反方向）。
    * 归纳式：最后 6 位人员入图年份设在 cutoff 之后，训练阶段不可见。

    返回: 含实体、全部有向边、训练/测试切分、邻接表、关系词表等的字典。
    """
    rng = np.random.default_rng(seed)
    rel2id, id2rel, inv, rel_head_types, rel_tail_types = build_relation_vocab(RELATIONS)

    entities = []
    ent_by_type = defaultdict(list)
    quads = []

    def add_entity(etype, created_at, name):
        eid = len(entities)
        entities.append(Entity(eid, etype, float(created_at), name))
        ent_by_type[etype].append(eid)
        return eid

    def add_quad(h, rname, t, tau):
        """添加一条有向边，并自动补一条反向边(便于 TGAT 双向聚合邻居)。"""
        rid = rel2id[rname]
        tau = float(tau)
        quads.append(Quad(h, rid, t, tau))
        quads.append(Quad(t, inv[rid], h, tau))

    # --- (1) 地点 / 职位：始终存在(created_at=0) ---
    loc_ids = {loc: add_entity("地点", 0.0, loc) for loc in ENTITIES["地点"]}
    pos_ids = {pos: add_entity("职位", 0.0, pos) for pos in ENTITIES["职位"]}

    # --- (2) 组织：成立时间 = 成立年份；并补 位于 / 成立于 ---
    org_ids = {}
    for org in ENTITIES["组织"]:
        fy = float(ORG_FOUNDED.get(org, int(rng.integers(1990, 2016))))
        org_ids[org] = add_entity("组织", fy, org)
        hq = ORG_HQ.get(org, ENTITIES["地点"][0])
        add_quad(org_ids[org], "位于", loc_ids[hq], fy + 0.5)   # 总部所在城市
        add_quad(org_ids[org], "成立于", loc_ids[hq], fy)         # 成立城市(近似用总部)

    # --- (3) 人员：毕业/入图年份；职业经历(任职于 + 担任职位)；毕业院校 ---
    person_ids = {}
    n_person = len(ENTITIES["人员"])
    for i, pname in enumerate(ENTITIES["人员"]):
        # 让最后 6 位在 2021-2023 年入图 -> 成为归纳式新实体
        if i >= n_person - 6:
            gy = int(rng.integers(2021, 2024))
        else:
            gy = int(rng.integers(2002, 2021))
        person_ids[pname] = add_entity("人员", float(gy), pname)

        # graduated_from：以清华大学为主，少量其它组织作为母校（仅用于 demo 多样性）
        # 时间设为入图年份 gy，与 created_at 一致 —— 这样「新入图人员」的毕业关系
        # 也会落在 cutoff 之后，不会把归纳式新实体泄漏进训练集。
        alma = ("清华大学" if rng.random() < 0.6
                else ENTITIES["组织"][int(rng.integers(0, len(ENTITIES["组织"])))])
        add_quad(person_ids[pname], "毕业于", org_ids[alma], gy)

        # 职业经历：1~4 段雇佣，每段对应一个 任职于 与一个 担任职位
        n_spells = int(rng.integers(1, 5))
        prev_org = None
        st = gy
        for _ in range(n_spells):
            if prev_org is None:
                st = gy
            else:
                lo, hi = int(st) + 1, int(horizon_end) - 1
                if lo > hi:
                    break  # 时间轴已无空间再安排下一段职业
                st = int(rng.integers(lo, hi + 1))
            cand_orgs = [o for o in ENTITIES["组织"] if o != prev_org]
            org = cand_orgs[int(rng.integers(0, len(cand_orgs)))]
            prev_org = org
            # 起始时间不得早于该组织成立时间，且不超出时间轴
            st = max(int(st), int(ORG_FOUNDED.get(org, 2000)))
            st = min(st, int(horizon_end) - 1)
            pos = ENTITIES["职位"][int(rng.integers(0, len(ENTITIES["职位"])))]
            add_quad(person_ids[pname], "任职于", org_ids[org], st)
            add_quad(person_ids[pname], "担任职位", pos_ids[pos], st)

    # --- (4) 组织间关系：收购 / 投资 / 合作 ---
    org_list = ENTITIES["组织"]
    n_org = len(org_list)
    for _ in range(35):                       # 收购（instant）
        a, b = rng.integers(0, n_org, size=2)
        if a == b:
            continue
        add_quad(org_ids[org_list[a]], "收购", org_ids[org_list[b]],
                 float(rng.uniform(2005, horizon_end)))
    for _ in range(45):                       # 投资（instant_quarter）
        a, b = rng.integers(0, n_org, size=2)
        if a == b:
            continue
        year = int(rng.integers(2005, horizon_end))
        q = int(rng.integers(1, 5))
        add_quad(org_ids[org_list[a]], "投资", org_ids[org_list[b]],
                 year + (q - 1) / 4.0)
    for _ in range(45):                       # 合作（对称 interval）
        a, b = rng.integers(0, n_org, size=2)
        if a == b:
            continue
        add_quad(org_ids[org_list[a]], "合作", org_ids[org_list[b]],
                 float(rng.uniform(2005, horizon_end)))

    # --- (5) 人员接任：接任（instant） ---
    person_list = ENTITIES["人员"]
    n_person_list = len(person_list)
    for _ in range(18):
        a, b = rng.integers(0, n_person_list, size=2)
        if a == b:
            continue
        add_quad(person_ids[person_list[a]], "接任", person_ids[person_list[b]],
                 float(rng.uniform(2005, horizon_end)))

    # --- 时间切分：cutoff 之前为训练，之后为测试 ---
    train_quads = [q for q in quads if q.time <= cutoff]
    test_quads = [q for q in quads if q.time > cutoff]

    # 邻接表：neigh[实体] = [(邻居, 关系id, 时间)]，用于 TGAT 聚合
    neigh = defaultdict(list)
    for q in quads:
        neigh[q.head].append((q.tail, q.relation, q.time))

    # 归纳式：入图时间晚于 cutoff 的实体，在训练阶段完全不可见
    new_entities = {e.eid for e in entities if e.created_at > cutoff}
    inductive_test = [q for q in test_quads
                      if (q.head in new_entities or q.tail in new_entities)]

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
        "train_cutoff": cutoff,
        "new_entities": new_entities,
    }


# ====================================================================
# 2.5 将完整四元组数据集输出到 csv 文件
# ====================================================================
def dump_quads_to_csv(data, filepath, entities, id2rel, cutoff=None):
    """把完整的四元组（Quad）数据集写入 csv 文件。

    采用标准 csv 格式（utf-8 + BOM，Excel 可直接打开），每行一条四元组：
        头实体, 头类型, 关系, 尾实体, 尾类型, 时间(τ)[, 数据集]

    其中「时间(τ)」保留一位小数；若提供 cutoff，则追加一列「数据集」
    标注该边属于 训练/测试（按时间切分点判断）。全部有向边（含自动生成的
    反向边）按时间 τ 升序排列。

    Args:
        data:     generate_employee_dataset 返回的字典。
        filepath: 输出 csv 路径。
        entities: Entity 列表（用于解析名称/类型）。
        id2rel:   关系 id -> 关系名 映射。
        cutoff:   训练切分点（可选）；若提供则额外追加「数据集」列。
    """
    import csv

    quads = data["quads"]
    # 按时间升序排列，时间相同则按 (头, 关系, 尾) 稳定排序
    ordered = sorted(quads, key=lambda q: (q.time, q.head, q.relation, q.tail))

    header = ["头实体", "头类型", "关系", "尾实体", "尾类型", "时间(τ)"]
    if cutoff is not None:
        header.append("数据集")

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # utf-8-sig 写入 BOM，保证 Excel 正确识别中文
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q in ordered:
            h, r, t, tau = q.head, q.relation, q.tail, q.time
            row = [entities[h].name, entities[h].etype,
                   id2rel[r],
                   entities[t].name, entities[t].etype,
                   f"{tau:.1f}"]
            if cutoff is not None:
                row.append("训练" if tau <= cutoff else "测试")
            writer.writerow(row)

    return filepath


# ====================================================================
# 2.6 模型保存 / 加载（训练后保存，推理时直接加载）
# ====================================================================
def save_checkpoint(model, data, filepath):
    """把训练好的模型权重 + 推理所需的全部图谱结构保存到单个 .pt 文件。

    推理时需要的不仅是模型权重，还包括实体表、邻接表、关系词表、候选类型约束、
    测试集四元组等（这些由数据集决定，必须随模型一起持久化）。
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "num_ent": data["num_ent"],
            "num_rel": data["num_rel"],
            "num_types": data["num_types"],
            "dim": model.dim,
            "n_heads": model.n_heads,
            "time_scale": model.time_scale,
        },
        # —— 推理所需的图谱结构（与具体数据集绑定） ——
        "ent_etype": [ENTITY_TYPES.index(e.etype) for e in data["entities"]],
        "entities": data["entities"],
        "neigh": {k: list(v) for k, v in data["neigh"].items()},
        "id2rel": data["id2rel"],
        "rel_tail_types": data["rel_tail_types"],
        "rel_head_types": data["rel_head_types"],
        "ent_by_type": {k: list(v) for k, v in data["ent_by_type"].items()},
        "rel2id": data["rel2id"],
        "new_entities": data["new_entities"],
        "train_cutoff": data["train_cutoff"],
        "test_quads": data["test_quads"],
        "inductive_test": data["inductive_test"],
    }, filepath)


def load_checkpoint(filepath):
    """从 .pt 文件加载模型与推理所需的图谱结构。返回 (model, data)。"""
    ckpt = torch.load(filepath, map_location="cpu")
    cfg = ckpt["config"]
    model = TemporalKGModel(
        num_ent=cfg["num_ent"],
        num_rel=cfg["num_rel"],
        num_types=cfg["num_types"],
        ent_etype=ckpt["ent_etype"],
        dim=cfg["dim"],
        n_heads=cfg["n_heads"],
        time_scale=cfg.get("time_scale", 1.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    data = {
        "entities": ckpt["entities"],
        "neigh": ckpt["neigh"],
        "id2rel": ckpt["id2rel"],
        "rel_tail_types": ckpt["rel_tail_types"],
        "rel_head_types": ckpt["rel_head_types"],
        "ent_by_type": ckpt["ent_by_type"],
        "rel2id": ckpt["rel2id"],
        "new_entities": ckpt["new_entities"],
        "train_cutoff": ckpt["train_cutoff"],
        "test_quads": ckpt["test_quads"],
        "inductive_test": ckpt["inductive_test"],
        "num_ent": cfg["num_ent"],
        "num_rel": cfg["num_rel"],
        "num_types": cfg["num_types"],
    }
    return model, data


def demo_inference(model, data):
    """用训练好的模型做一组可解释的链接预测示例（训练后 / 推理时共用）。"""
    entities = data["entities"]
    neigh = data["neigh"]
    id2rel = data["id2rel"]
    rel_tail_types = data["rel_tail_types"]
    ent_by_type = data["ent_by_type"]

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
    # 专门演示 任职于 与 担任职位 的推理
    works_at_rid = data["rel2id"]["任职于"]
    holds_rid = data["rel2id"]["担任职位"]
    located_rid = data["rel2id"]["位于"]
    wa_demo = next((q for q in data["test_quads"] if q.relation == works_at_rid), None)
    if wa_demo:
        demo_one(wa_demo, "任职于：预测某人任职的公司")
    hp_demo = next((q for q in data["test_quads"] if q.relation == holds_rid), None)
    if hp_demo:
        demo_one(hp_demo, "担任职位：预测某人担任的职位")
    lo_demo = next((q for q in data["test_quads"] if q.relation == located_rid), None)
    if lo_demo:
        demo_one(lo_demo, "位于：预测某公司总部所在城市")
    print("=" * 70)


def _entity_id_by_token(token, entities):
    """把用户输入解析成实体整数 ID：支持中文名称 或 整数 ID。无法解析返回 None。"""
    token = (token or "").strip()
    if token == "":
        return None
    try:
        i = int(token)
        if 0 <= i < len(entities):
            return i
    except ValueError:
        pass
    for e in entities:
        if e.name == token:
            return e.eid
    return None


def _relation_id_by_token(token, rel2id, num_rel):
    """把用户输入解析成关系整数 ID：支持关系名 或 整数 ID。无法解析返回 None。"""
    token = (token or "").strip()
    if token == "":
        return None
    try:
        i = int(token)
        if 0 <= i < num_rel:
            return i
    except ValueError:
        pass
    if token in rel2id:
        return int(rel2id[token])
    return None


def run_manual_infer(model, data, topk, head=None, relation=None, time=None,
                     interactive=False):
    """手动推理：给定 头实体 / 关系 / 时间，预测 Top-K 尾实体。

    - interactive=True：进入交互式循环，逐条输入；任一项留空回车即退出。
    - 否则用 head/relation/time 做一次推理（三者需同时提供）。
    时间 τ 由用户手动指定，可体现「同一 (头,关系) 在不同时间的预测差异」。
    """
    entities = data["entities"]
    neigh = data["neigh"]
    ent_by_type = data["ent_by_type"]
    rel_tail_types = data["rel_tail_types"]
    rel2id = data["rel2id"]
    id2rel = data["id2rel"]
    num_rel = data["num_rel"]
    name = lambda eid: f"{entities[eid].name}({entities[eid].etype})"

    def do_one(h, r, tau):
        top = predict_tails(model, h, r, tau, neigh, ent_by_type, rel_tail_types, k=topk)
        print(f"\n  查询: ({name(h)}, {id2rel[r]}, ?)  @时间τ={tau:.1f}")
        if not top:
            print("    (该关系无候选尾实体类型，无法预测)")
        for c, s in top:
            print(f"      - {name(c)}   得分={s:+.4f}")
        print()

    if interactive:
        print("\n【交互式推理】输入 头实体 / 关系 / 时间（任一项留空回车即退出）")
        while True:
            try:
                h_in = input("  头实体 (名称或整数ID): ").strip()
                r_in = input("  关系   (名称或整数ID): ").strip()
                t_in = input("  时间   (年份, 如 2021): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出交互式推理。")
                break
            if h_in == "" or r_in == "" or t_in == "":
                print("退出交互式推理。")
                break
            h = _entity_id_by_token(h_in, entities)
            r = _relation_id_by_token(r_in, rel2id, num_rel)
            try:
                tau = float(t_in)
            except ValueError:
                print("  ⚠️ 时间必须是数字（年份，如 2021）\n")
                continue
            if h is None:
                print(f"  ⚠️ 未知头实体: {h_in}\n")
                continue
            if r is None:
                print(f"  ⚠️ 未知关系: {r_in}\n")
                continue
            do_one(h, r, tau)
        return

    h = _entity_id_by_token(head, entities)
    r = _relation_id_by_token(relation, rel2id, num_rel)
    tau = float(time)
    if h is None:
        raise SystemExit(f"❌ 未知头实体: {head}")
    if r is None:
        raise SystemExit(f"❌ 未知关系: {relation}")
    print(f"\n【手动推理】查询: ({name(h)}, {id2rel[r]}, ?)  @时间τ={tau:.1f}")
    do_one(h, r, tau)


# ====================================================================
# 3. 主函数：生成数据 -> 训练 -> 评估 -> 保存 -> 推理示例
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="员工-公司雇佣关系 时序知识图谱链接预测 demo")
    parser.add_argument("--mode", choices=["train", "infer"], default="train",
                        help="train=训练并保存模型; infer=直接加载已训练模型做推理")
    parser.add_argument("--model_dir", type=str, default="trained_models",
                        help="模型保存/加载目录")
    parser.add_argument("--force", action="store_true",
                        help="即使模型已存在也强制重新训练（覆盖已保存模型）")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=float, default=2020.0)
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "employee_tkg_quads.csv"),
                        help="完整四元组数据集输出路径（csv，仅 train 模式生成）")
    # 手动推理：交互式（--interactive）或一次性指定（--head/--relation/--time）
    parser.add_argument("--topk", type=int, default=5,
                        help="推理返回的 Top-K 候选尾实体数")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式推理：逐条手动输入 头实体/关系/时间 进行预测")
    parser.add_argument("--head", type=str, default=None,
                        help="手动推理的头实体（名称或整数ID），配合 --relation/--time 使用")
    parser.add_argument("--relation", type=str, default=None,
                        help="手动推理的关系（名称或整数ID）")
    parser.add_argument("--time", type=float, default=None,
                        help="手动推理的时间（年份，如 2021；可带小数）")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.model_dir, "employee_tkg_model.pt")

    # ================= 推理模式：直接加载已训练模型 =================
    # 判断逻辑：模型文件已存在 且 未显式 --force 重训时，跳过训练直接推理
    #   * --mode infer            -> 强制走推理（模型不存在则报错）
    #   * --mode train（默认）    -> 模型存在则自动跳过训练；不存在则训练并保存
    #   * --force                 -> 无论模型是否存在都重新训练
    skip_train = (args.mode == "infer") or (os.path.exists(ckpt_path) and not args.force)
    if skip_train:
        if not os.path.exists(ckpt_path):
            raise SystemExit(
                f"❌ 未找到模型文件: {ckpt_path}\n"
                f"   请先以 --mode train 训练并保存模型。")
        if args.mode == "train" and not args.force:
            print(f"✅ 检测到已训练模型: {ckpt_path}，跳过训练直接推理。")
        print(f"📥 加载已训练模型: {ckpt_path}")
        model, data = load_checkpoint(ckpt_path)
        print("=" * 70)
        print("【本体与数据概览】（从已保存模型加载）")
        print(f"  实体总数={data['num_ent']}, 关系类型数={data['num_rel']} (含反向)")
        print(f"  时间切分点 cutoff={data['train_cutoff']:.1f}")
        print("=" * 70)

        # ===== 手动推理：交互式 或 一次性指定（--head/--relation/--time） =====
        if args.interactive or (args.head is not None and args.relation is not None
                                and args.time is not None):
            run_manual_infer(model, data, args.topk,
                             head=args.head, relation=args.relation, time=args.time,
                             interactive=args.interactive)
            return

        demo_inference(model, data)
        return

    # ================= 训练模式：生成数据 -> 训练 -> 保存 =================
    # ---- 1) 生成动态时序知识图谱 ----
    data = generate_employee_dataset(seed=args.seed, cutoff=args.cutoff)
    entities = data["entities"]
    ent_by_type = data["ent_by_type"]
    neigh = data["neigh"]
    id2rel = data["id2rel"]
    rel_tail_types = data["rel_tail_types"]
    rel_head_types = data["rel_head_types"]

    print("=" * 70)
    print("【本体与数据概览】（员工 - 公司雇佣关系 TKG）")
    print(f"  实体类型数={data['num_types']} (人员/组织/地点/职位)")
    print(f"  关系类型数={data['num_rel']} (含反向关系)")
    print(f"  实体总数={data['num_ent']}, 有向边总数={len(data['quads'])}")
    print(f"  训练边={len(data['train_quads'])}, 测试边={len(data['test_quads'])}, "
          f"归纳式测试边={len(data['inductive_test'])}")
    print(f"  时间切分点 cutoff={data['train_cutoff']:.1f} "
          f"(之后入图的实体为归纳式新实体，共 {len(data['new_entities'])} 个)")
    print("  各类型实体数:", {et: len(ent_by_type[et]) for et in ENTITY_TYPES})
    print("=" * 70)

    # ---- 1.5) 将完整四元组数据集输出到 csv 文件 ----
    out_path = dump_quads_to_csv(data, args.out, entities, id2rel,
                                 cutoff=data["train_cutoff"])
    print(f"📄 完整四元组数据集已写出: {out_path}  (共 {len(data['quads'])} 条有向边)")

    # 实体类型索引(用于初始化实体嵌入)
    ent_etype = [ENTITY_TYPES.index(e.etype) for e in entities]

    # 时间归一化尺度：把年份量级的时间戳缩放到 O(1)，避免 TimeEncoder 中 freq*τ 过大
    # 导致梯度爆炸、训练失稳（失稳会让实体嵌入被压到极小，正负样本分数都挤在 0 附近、
    # 推理排名接近随机）。取全部四元组时间的最大绝对值。
    all_times = [q.time for q in data["quads"]]
    time_scale = float(max((abs(t) for t in all_times), default=1.0)) or 1.0

    # ---- 2) 构建模型（时间作为关系属性：TGAT 编码 + 时间感知 DistMult 解码） ----
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

    # ---- 4) 评估（过滤式 MRR） ----
    mrr_all, _ = evaluate(model, data["test_quads"], neigh, ent_by_type, rel_tail_types)
    mrr_ind, _ = evaluate(model, data["inductive_test"], neigh, ent_by_type, rel_tail_types)
    print("=" * 70)
    print(f"【评估结果】测试集 MRR={mrr_all:.4f}  |  归纳式(新实体)测试集 MRR={mrr_ind:.4f}")
    print("  (MRR 越接近 1 越好；归纳式 MRR>0 说明模型能预测从未见过的实体)")
    print("=" * 70)

    # ---- 5) 保存模型（含推理所需的全部图谱结构） ----
    save_checkpoint(model, data, ckpt_path)
    print(f"💾 模型已保存到: {ckpt_path}")

    # ---- 6) 示例推理（可解释输出） ----
    demo_inference(model, data)


if __name__ == "__main__":
    main()

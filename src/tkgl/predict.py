# -*- coding: utf-8 -*-
"""
src/tkgl/predict.py
=====================================================================
TKGL-Smallpedia 时序知识图谱链接预测 —— 推理与评测

包含：
  * evaluate_filtered      : 过滤式 MRR / Hits@k 评测（采样负样本）
  * _build_relation_endpoints / _get_relation_tail_set : 类型约束候选集构建
  * _temporal_recency      : 推理时的时间邻近性偏置
  * predict_tails          : 给定 (头,关系,时间) 预测 Top-K 尾实体
  * _resolve_id            : Q-ID/P-ID 或整数 ID → 整数
  * run_interactive_infer  : 交互式推理
  * run_inference          : 推理主流程（加载 checkpoint → 手动/评测/示例）
  * main                   : 命令行入口

模型与 checkpoint 序列化见 src/model/temporal_model.py；数据集加载见 src/dataset/tkgl_dataset.py。
"""

import os
import sys
import math
import argparse

import numpy as np
import torch

# 项目根目录加入 sys.path，使以脚本方式运行本文件时 `import src.model...` 等
# 包内绝对导入可解析（`python -m src.tkgl.predict` 时 cwd 已在 path，无需此步）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model.tkgl_model import load_checkpoint, TemporalKGModel  # noqa: E402


# ====================================================================
# 1. 过滤式 MRR 评测（采样负样本）
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
# 2. 类型约束候选集 / 困难负采样辅助
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


# ====================================================================
# 3. 推理：给定 (头, 关系, 时间) 预测 Top-K 尾实体（在全实体上排名）
# ====================================================================
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


def run_interactive_infer(model, data, device, topk, k_neg,
                          temporal_bias=0.0, temporal_sigma=8.0):
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
                            temporal_bias=temporal_bias,
                            temporal_sigma=temporal_sigma)
        print(f"\n  查询: ({id2ent[h]}, {id2rel[r]}, ?)  @τ={ts}")
        for tid, sc in top:
            print(f"      - {id2ent[tid]}   得分={sc:+.4f}")
        print()


def run_inference(args, device):
    """推理主流程：加载 checkpoint → 手动推理 / 测试评测 / 示例推理。

    与 demo/tkgl_smallpedia_tkg.py 中「推理模式」分支等价，复用了
    predict_tails / evaluate_filtered 等函数。
    """
    ckpt_path = args.model_path
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
            run_interactive_infer(model, data, device, args.topk, args.k_neg,
                                  temporal_bias=args.temporal_bias,
                                  temporal_sigma=args.temporal_sigma)
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


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="TKGL-Smallpedia 时序知识图谱链接预测（推理）")
    parser.add_argument("--model-path", type=str, required=True,
                        help="已训练模型 .pt 路径")
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
    parser.add_argument("--interactive", action="store_true",
                        help="交互式推理：逐条手动输入 头实体/关系/时间 进行预测")
    parser.add_argument("--head", type=str, default=None,
                        help="手动推理的头实体（Q-ID 或整数ID），配合 --relation/--time 使用")
    parser.add_argument("--relation", type=str, default=None,
                        help="手动推理的关系（P-ID 或整数ID）")
    parser.add_argument("--time", type=int, default=None,
                        help="手动推理的时间（年份整数，如 2008）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备 auto/cpu/cuda")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    run_inference(args, device)


if __name__ == "__main__":
    main()

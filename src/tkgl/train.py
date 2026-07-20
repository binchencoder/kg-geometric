# -*- coding: utf-8 -*-
"""
src/tkgl/train.py
=====================================================================
TKGL-Smallpedia 时序知识图谱链接预测 —— 训练

包含：
  * _sample_typed_negatives : 同关系困难负采样（训练用）
  * train_model             : 训练循环（分批流式训练边，定期 val 评测）
  * run_training            : 训练主流程（加载数据 → 建模型 → 训练 → 评测 → 保存）
  * main                    : 命令行入口

模型见 src/model/tkgl.py；数据集加载见 src/dataset/tkgl_dataset.py；
过滤式评测 evaluate_filtered 见 src/tkgl/predict.py。
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# 项目根目录加入 sys.path，使以脚本方式运行本文件时 `import src.model...` /
# `import src.dataset...` / `import src.tkgl...` 等包内绝对导入可解析
# （`python -m src.tkgl.train` 时 cwd 已在 path 中，无需此步；但 `python
#  src/tkgl/train.py` 直接运行时脚本目录才在 path，必须补上根目录）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model.tkgl import TemporalKGModel, save_checkpoint  # noqa: E402
from src.dataset.tkgl_dataset import load_tkgl_smallpedia, iter_train_batches  # noqa: E402

# 过滤式评测（训练期 val 监控用）；predict 模块本身只依赖 temporal_model，
# 不会产生循环依赖。
from src.tkgl.predict import evaluate_filtered, _build_relation_endpoints, _get_relation_tail_set  # noqa: E402


# ====================================================================
# 1. 同关系困难负采样
# ====================================================================
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
        ri = int(r[i]);
        ti = int(t[i]);
        hi = int(h[i]);
        tsi = int(ts[i])
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


# ====================================================================
# 2. 训练循环
# ====================================================================
def train_model(
        model,
        batch_iter_fn, optimizer,
        device, epochs=30, batch_size=1024,
        n_neg=5, num_samples=20, val_data=None, num_eval=2000, k_neg=500,
        log_every=5, neg_mode="uniform"
):
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
                print(f"    [epoch {ep + 1}] batch {n_batch}  "
                      f"avg_loss(近100)={running / 100:.4f}  {speed:.1f} batch/s", flush=True)
                running = 0.0

        dt = time.time() - t0
        if (ep + 1) % log_every == 0 or ep == 0:
            msg = (f"  epoch {ep + 1:3d}/{epochs}  loss={total_loss / max(n_batch, 1):.4f}  "
                   f"batches={n_batch}  ({dt:.1f}s)")
            if val_data is not None and val_data["val_quads"].shape[0] > 0:
                mrr, h1, h3, h10 = evaluate_filtered(
                    model, val_data["val_quads"], val_data["true_tails"],
                    num_eval=min(num_eval, 2000), k_neg=k_neg, device=device)
                msg += f"  | val MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f}"
            print(msg)


# ====================================================================
# 3. 训练主流程
# ====================================================================
def run_training(args, device):
    """训练主流程：加载数据集 → 构建模型 → 训练 → 测试评测 → 保存 checkpoint。

    与 demo/tkgl_smallpedia_tkg.py 中「训练模式」分支等价。
    """
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
    save_checkpoint(model, data, args.model_path)
    print(f"💾 模型已保存到: {args.model_path}")
    print("   之后可用: python demo/tkgl_smallpedia_tkg.py --mode infer 直接加载推理。")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="TKGL-Smallpedia 时序知识图谱链接预测（训练）")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(_ROOT, "data", "tkgl-smallpedia"))
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(_ROOT, "trained_models", "tkgl_smallpedia_model.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--neg-mode", choices=["uniform", "typed"], default="uniform",
                        help="负采样方式：uniform=均匀随机(默认,收敛快); typed=同关系困难负样本")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=20, help="邻居采样数（=max_neigh）")
    parser.add_argument("--buffer", type=int, default=20000,
                        help="流式训练的 shuffle-buffer 大小（越大打乱越接近全局，"
                             "但占用内存越多；训练集本身不整体驻留）")
    parser.add_argument("--num-eval", type=int, default=2000, help="评测查询数")
    parser.add_argument("--k-neg", type=int, default=500, help="评测每个查询的负样本数")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备 auto/cpu/cuda")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    run_training(args, device)


if __name__ == "__main__":
    main()

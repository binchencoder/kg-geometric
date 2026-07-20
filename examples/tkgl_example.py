# -*- coding: utf-8 -*-
"""
demo/tkgl_smallpedia_tkg.py
=====================================================================
基于 TKGL-Smallpedia 数据集的「时序知识图谱链接预测」训练 / 推理 入口

模型：TGAT 风格时序图注意力编码器 + 时间感知 DistMult 解码器
      （「时间」作为关系的一个属性：rel_repr = rel_emb + TimeEncoder(τ)）

本文件现在是**薄入口**：模型 / 数据集 / 训练 / 推理逻辑已抽取到独立模块：
  * src/model/temporal_model.py : TemporalKGModel + checkpoint 序列化
  * src/dataset/tkgl_dataset.py : 数据集加载（load_tkgl_smallpedia / iter_train_batches）
  * src/tkgl/train.py      : 训练主流程（run_training）
  * src/tkgl/predict.py    : 推理与评测主流程（run_inference）

本文件只负责解析命令行参数，并委派给上述模块，保持
`python demo/tkgl_smallpedia_tkg.py --mode train|infer ...` 的原有用法不变。

运行：
    # 训练并保存模型
    python demo/tkgl_smallpedia_tkg.py --mode train

    # 直接加载已训练模型做推理 / 评测
    python demo/tkgl_smallpedia_tkg.py --mode infer

提示：数据集较大，强烈建议在 GPU 上训练（自动检测 cuda；无 cuda 则退化为 CPU，
但会非常慢）。可用 --epochs / --dim / --batch / --neg 等调节。
"""

import os
import sys
import argparse

# 让本文件既能被 `python demo/xxx.py` 直接运行，也能从项目根目录运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tkgl.train import run_training       # noqa: E402
from src.tkgl.predict import run_inference    # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="TKGL-Smallpedia 时序知识图谱链接预测")
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

    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ckpt_path = args.model_path
    skip_train = (args.mode == "infer") or (os.path.exists(ckpt_path) and not args.force)

    if skip_train:
        run_inference(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()

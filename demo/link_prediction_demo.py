"""知识图谱链接预测 Demo —— 端到端演示。

支持三种数据源模式：
1. demo 模式（默认）：使用内置 KGFaultDataset 工业故障知识图谱（小数据，全图训练）
2. es 模式：从 Elasticsearch 读取全部三元组到内存后训练（中等数据）
3. streaming 模式（推荐大数据）：异步子图采样 + "边查边训"（海量数据）

完整流程：
1. 构建链接预测数据集
2. 训练 GCN + DistMult 链接预测模型
3. 评估指标
4. Top-K 链接推理

用法：
    # 内置 Demo 模式
    python demo/link_prediction_demo.py --mode demo

    # ES 全量模式
    python demo/link_prediction_demo.py --mode es --index knowledge_entity_relation_index

    # 流式子图采样模式（推荐大数据场景）
    python demo/link_prediction_demo.py --mode streaming --index knowledge_entity_relation_index
    python demo/link_prediction_demo.py --mode streaming --index my_kg_index --graph-id 980044155496734720 \\
        --max-edges 500000 --epochs 100 --num-hops 2 --max-neighbors 50
"""

from __future__ import annotations

import argparse
import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.core.config import ESConfig, logger
from src.core.types import Triple
from src.dataset.link_prediction import LinkPredictionData, LinkPredictionStreamingData
from src.model import (
    LinkPredictionGCN,
    train_link_prediction,
    evaluate_link_prediction,
    predict_top_k,
    print_link_prediction_results,
    train_link_prediction_streaming,
    evaluate_link_prediction_streaming,
    predict_top_k_streaming,
)


# -------------------- 命令行参数解析 --------------------


def _parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="知识图谱链接预测 —— 训练与推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 内置 Demo 模式（小型工业故障知识图谱）
  python demo/link_prediction_demo.py --mode demo

  # ES 全量模式（中等规模，加载全部到内存）
  python demo/link_prediction_demo.py --mode es --index knowledge_entity_relation_index

  # 流式子图采样模式（推荐大数据场景，不加载全量到内存）
  python demo/link_prediction_demo.py --mode streaming --index knowledge_entity_relation_index
  python demo/link_prediction_demo.py --mode streaming --index my_kg_index --max-edges 1000000
        """,
    )

    # ── 数据源 ──
    parser.add_argument(
        "--mode", choices=["es", "streaming"], default="streaming",
        help="数据源模式: es=ES全量加载 | streaming=ES流式子图采样 (default: streaming)",
    )
    parser.add_argument(
        "--index", default=["knowledge_entity_relation_index"], nargs="+",
        help="ES 关系索引名，支持多个 (es/streaming 模式)",
    )

    # ── ES 连接与过滤 ──
    parser.add_argument("--graph-id", default="980044155496734720",
                        help="按 graphId 过滤三元组 (es/streaming 模式)")
    parser.add_argument("--ontology-id", default="979748419706068992",
                        help="关系类型 ontologyId (仅 es 模式)")
    parser.add_argument("--es-batch-size", type=int, default=5000,
                        help="ES 每批读取文档数")

    # ── 字段映射 ──
    parser.add_argument("--head-field", default="srcEntityId",
                        help="头实体字段名")
    parser.add_argument("--relation-field", default="relationTypeId",
                        help="关系类型字段名")
    parser.add_argument("--tail-field", default="dstEntityId",
                        help="尾实体字段名")

    # ── 实体/关系类型索引（ID→名称解析） ──
    parser.add_argument("--entity-index", default="knowledge_entity_index",
                        help="实体索引名称")
    parser.add_argument("--entity-id-field", default="entityId",
                        help="实体索引中的 ID 字段名")
    parser.add_argument("--entity-name-field", default="name",
                        help="实体索引中的名称字段名")
    parser.add_argument("--relation-type-index", default="knowledge_entity_type_relation_index",
                        help="关系类型索引名称")
    parser.add_argument("--relation-type-id-field", default="relationTypeId",
                        help="关系类型索引中的 ID 字段名")
    parser.add_argument("--relation-type-name-field", default="name",
                        help="关系类型索引中的名称字段名")
    parser.add_argument("--no-resolve", action="store_true",
                        help="禁用 ID→名称解析")
    parser.add_argument("--resolve-debug", action="store_true",
                        help="启用解析器调试模式")

    # ── 模型与训练参数 ──
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GCN 隐藏层维度")
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="L2 正则化系数")
    parser.add_argument("--batch-size-train", type=int, default=64,
                        help="训练时每批正样本数")
    parser.add_argument("--num-negatives", type=int, default=1,
                        help="每个正样本的负样本数")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="训练日志输出间隔")
    parser.add_argument("--device", default=None,
                        help="训练设备 (cuda / cpu)，默认自动检测")

    # ── 流式模式特有参数 ──
    parser.add_argument("--num-hops", type=int, default=2,
                        help="子图采样跳数 (streaming 模式)")
    parser.add_argument("--max-neighbors", type=int, default=100,
                        help="每跳最大邻居数 (streaming 模式)")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="最大三元组数限制（调试用，None=不限制）")
    parser.add_argument("--val-batches", type=int, default=5,
                        help="验证时采样批次数 (streaming 模式)")
    parser.add_argument("--eval-batches", type=int, default=20,
                        help="测试评估时采样批次数 (streaming 模式)")

    # ── 推理参数 ──
    parser.add_argument("--top-k", type=int, default=3,
                        help="链接预测推理时返回的 Top-K 候选数")
    parser.add_argument("--query-head", nargs="+", default=["李一诺"],
                        help="推理时指定的查询头实体名称，多个用空格分隔")
    parser.add_argument("--query-rel", nargs="+", default=["具有"],
                        help="推理时指定的查询关系名称，多个用空格分隔")

    return parser.parse_args()


# -------------------- ES ID→名称解析器 --------------------


def _build_resolver(args, config: ESConfig):
    """构建 ID→名称解析器。"""
    from src.es import create_es_client, IDNameResolver

    if args.no_resolve:
        logger.info("已禁用 ID→名称解析，将直接使用原始 ID")
        return None

    logger.info("Phase 0: 构建 ID→名称解析器...")
    if args.resolve_debug:
        logger.info("[DEBUG] 解析器调试模式已启用")

    resolver = IDNameResolver(create_es_client(config))
    resolver.build_entity_map(
        entity_index=args.entity_index,
        graph_id=args.graph_id,
        id_field=args.entity_id_field,
        name_field=args.entity_name_field,
        batch_size=args.es_batch_size,
        extra_query={"match": {"graphId": args.graph_id}} if args.graph_id else None,
        debug=args.resolve_debug,
    )
    resolver.build_relation_type_map(
        relation_type_index=args.relation_type_index,
        ontology_id=args.ontology_id,
        id_field=args.relation_type_id_field,
        name_field=args.relation_type_name_field,
        batch_size=args.es_batch_size,
        debug=args.resolve_debug,
    )
    logger.info(
        "ID→名称解析器就绪: 实体=%d 种, 关系类型=%d 种",
        len(resolver.entity_map), len(resolver.relation_type_map),
    )
    return resolver


# -------------------- ES Streamer 工厂 --------------------


def _create_streamer(args, resolver) -> "ESTripletStreamer":
    """创建 ESTripletStreamer 实例。"""
    from src.es import ESTripletStreamer

    config = ESConfig()
    index_names = args.index if len(args.index) > 1 else args.index[0]
    logger.info("创建 ES Streamer，目标索引: %s", index_names)

    return ESTripletStreamer(
        es_hosts=[f"{config.scheme}://{config.host}:{config.port}"],
        index_name=index_names,
        batch_size=args.es_batch_size,
        prefetch=True,
        checkpoint_dir="./es_checkpoints",
        resolver=resolver,
    )


def _get_extra_filters(args) -> dict | None:
    """构建 ES 额外过滤条件。"""
    if args.graph_id:
        logger.info("启用 graphId 过滤: %s", args.graph_id)
        return {"match": {"graphId": args.graph_id}}
    return None


# -------------------- Demo 模式数据 --------------------


def _load_triples_demo() -> list:
    """返回内置工业故障知识图谱三元组列表。"""
    return [
        Triple("泵_01", "存在症状", "振动过高"),
        Triple("泵_01", "存在症状", "温度过高"),
        Triple("泵_01", "原因在于", "轴承磨损"),
        Triple("电机_02", "存在症状", "电流过高"),
        Triple("电机_02", "原因在于", "定子故障"),
        Triple("齿轮箱_03", "存在症状", "噪音异常"),
        Triple("齿轮箱_03", "原因在于", "齿轮磨损"),
        Triple("压缩机_04", "存在症状", "压力过低"),
        Triple("压缩机_04", "原因在于", "阀门泄漏"),
        Triple("轴承磨损", "类型为", "故障"),
        Triple("定子故障", "类型为", "故障"),
        Triple("齿轮磨损", "类型为", "故障"),
        Triple("阀门泄漏", "类型为", "故障"),
    ]


# -------------------- ES 全量加载模式 --------------------


def _load_triples_from_es(args) -> list:
    """从 ES 流式读取全部三元组到内存。"""
    config = ESConfig()
    resolver = _build_resolver(args, config)
    streamer = _create_streamer(args, resolver)
    extra_filters = _get_extra_filters(args)

    triples = []
    total = 0
    logger.info("开始从 ES 全量加载三元组...")
    for batch in streamer.stream_triplets(
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        resume=False,
        extra_filters=extra_filters,
    ):
        for t in batch:
            triples.append(Triple(
                head=t["head"],
                relation=t["relation"],
                tail=t["tail"],
            ))
            total += 1
            if args.max_edges and total >= args.max_edges:
                break
        if total % 50000 == 0:
            logger.info("已加载 %d 条三元组...", total)
        if args.max_edges and total >= args.max_edges:
            break

    logger.info("三元组加载完成: 共 %d 条", total)

    if not triples:
        raise ValueError("未能从 ES 读取到任何三元组！请检查索引名、graphId 及字段映射参数。")

    logger.info("预览前 5 条三元组:")
    for t in triples[:5]:
        logger.info("  (%s) --[%s]--> (%s)", t.head, t.relation, t.tail)

    return triples


# -------------------- 流式模式：构建数据 + 采样器 --------------------


def _build_streaming_data_and_sampler(args):
    """Phase 1: 构建流式数据集（vocab + 边划分）+ AsyncSubgraphSampler。

    Returns
    -------
    tuple
        (streaming_data, sampler, streamer)
    """
    from src.graph.sampler import AsyncSubgraphSampler

    config = ESConfig()
    resolver = _build_resolver(args, config)
    streamer = _create_streamer(args, resolver)
    extra_filters = _get_extra_filters(args)

    # Phase 1: 流式构建词汇表 + 边划分
    streaming_data = LinkPredictionStreamingData()
    total = streaming_data.build_from_streamer(
        streamer=streamer,
        split_ratios=(0.8, 0.1, 0.1),
        seed=42,
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        extra_filters=extra_filters,
        max_edges=args.max_edges,
    )
    logger.info("Phase 1 完成: 词汇表已构建，共 %d 条三元组", total)

    # Phase 2: 创建异步子图采样器
    sampler = AsyncSubgraphSampler(
        streamer=streamer,
        vocab=streaming_data.vocab,
        num_hops=args.num_hops,
        max_neighbors_per_hop=args.max_neighbors,
        prefetch_size=2,  # 预取队列大小
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        embedding_dim=args.hidden_dim,
        resolver=resolver,
    )
    logger.info(
        "Phase 2 完成: 异步子图采样器就绪 (num_hops=%d, max_neighbors=%d)",
        args.num_hops, args.max_neighbors,
    )

    return streaming_data, sampler, streamer


# -------------------- 推理执行 --------------------


def _run_inference_queries_full(
    model: LinkPredictionGCN,
    lp_data: LinkPredictionData,
    device: str,
    args,
) -> None:
    """全图模式下的链接预测推理。"""
    print("\n" + "-" * 40)
    print("链接预测推理 (Top-K 尾实体预测)")
    print("-" * 40)

    if args.query_head and args.query_rel:
        heads = args.query_head
        rels = args.query_rel
        if len(heads) != len(rels):
            logger.warning(
                "--query-head 和 --query-rel 数量不匹配 (%d vs %d)",
                len(heads), len(rels),
            )
        for h, r in zip(heads, rels):
            try:
                results = predict_top_k(
                    model=model,
                    data=lp_data.data,
                    head_name=h,
                    relation_name=r,
                    node_to_idx=lp_data.node_to_idx,
                    idx_to_node=lp_data.idx_to_node,
                    rel_to_idx=lp_data.rel_to_idx,
                    all_triples_set=lp_data._all_triples_set,
                    num_nodes=lp_data.num_nodes,
                    top_k=args.top_k,
                    device=device,
                )
                print_link_prediction_results(results, h, r)
            except ValueError as e:
                logger.warning("查询 (%s, %s, ?) 失败: %s", h, r, e)
    else:
        if lp_data.test_edges:
            h_idx, r_idx, t_idx = lp_data.test_edges[0]
            h_name = lp_data.idx_to_node.get(h_idx, str(h_idx))
            r_name = lp_data.idx_to_rel.get(r_idx, str(r_idx))
            logger.info("自动选取测试边作为查询示例: (%s, %s, %s)", h_name, r_name,
                         lp_data.idx_to_node.get(t_idx, str(t_idx)))
            try:
                results = predict_top_k(
                    model=model,
                    data=lp_data.data,
                    head_name=h_name,
                    relation_name=r_name,
                    node_to_idx=lp_data.node_to_idx,
                    idx_to_node=lp_data.idx_to_node,
                    rel_to_idx=lp_data.rel_to_idx,
                    all_triples_set=lp_data._all_triples_set,
                    num_nodes=lp_data.num_nodes,
                    top_k=args.top_k,
                    device=device,
                )
                print_link_prediction_results(results, h_name, r_name)
            except ValueError as e:
                logger.warning("自动查询失败: %s", e)
        else:
            logger.warning("测试边为空，请通过 --query-head/--query-rel 手动指定。")


def _run_inference_queries_streaming(
    model: LinkPredictionGCN,
    sampler,
    streaming_data: LinkPredictionStreamingData,
    device: str,
    args,
) -> None:
    """流式模式下的链接预测推理。"""
    print("\n" + "-" * 40)
    print("链接预测推理 (Top-K 尾实体预测) [流式子图模式]")
    print("-" * 40)

    if args.query_head and args.query_rel:
        heads = args.query_head
        rels = args.query_rel
        for h, r in zip(heads, rels):
            try:
                results = predict_top_k_streaming(
                    model=model,
                    sampler=sampler,
                    head_name=h,
                    relation_name=r,
                    streaming_data=streaming_data,
                    top_k=args.top_k,
                    device=device,
                )
                print_link_prediction_results(results, h, r)
            except (ValueError, RuntimeError) as e:
                logger.warning("查询 (%s, %s, ?) 失败: %s", h, r, e)
    else:
        if streaming_data.test_edges:
            h_name, r_name, t_name = streaming_data.test_edges[0]
            logger.info(
                "自动选取测试边作为查询示例: (%s, %s, %s)",
                h_name, r_name, t_name,
            )
            try:
                results = predict_top_k_streaming(
                    model=model,
                    sampler=sampler,
                    head_name=h_name,
                    relation_name=r_name,
                    streaming_data=streaming_data,
                    top_k=args.top_k,
                    device=device,
                )
                print_link_prediction_results(results, h_name, r_name)
            except (ValueError, RuntimeError) as e:
                logger.warning("自动查询失败: %s", e)
        else:
            logger.warning("测试边为空，请通过 --query-head/--query-rel 手动指定。")


# -------------------- 主流程 --------------------

def _run_es_mode(args):
    """ES 全量模式 —— 加载全部三元组后全图训练。"""
    print("\n从 Elasticsearch 知识图谱全量加载三元组...")
    triples = _load_triples_from_es(args)

    print("\n构建链接预测数据集 (边划分 80%/10%/10%)...")
    lp_data = LinkPredictionData(triples, split_ratios=(0.8, 0.1, 0.1), seed=42)

    print("\n数据集统计:")
    for k, v in lp_data.statistics().items():
        print(f"  {k}: {v}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkPredictionGCN(
        in_dim=lp_data.num_nodes,
        hidden_dim=args.hidden_dim,
        num_relations=lp_data.num_relations,
        dropout=0.2,
    )
    print(f"\n模型架构:\n{model}")
    print(f"设备: {device}")
    print(f"训练边: {len(lp_data.train_edges)}, 验证边: {len(lp_data.val_edges)}, 测试边: {len(lp_data.test_edges)}")

    print("\n" + "-" * 40)
    print("开始训练链接预测模型 (全图模式)...")
    print("-" * 40)

    model = train_link_prediction(
        model=model,
        data=lp_data.data,
        train_edges=lp_data.train_edges,
        val_edges=lp_data.val_edges,
        all_triples_set=lp_data._all_triples_set,
        num_nodes=lp_data.num_nodes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=min(args.batch_size_train, len(lp_data.train_edges)),
        num_negatives=args.num_negatives,
        log_interval=args.log_interval,
        device=device,
    )

    print("\n" + "-" * 40)
    print("测试集评估...")
    print("-" * 40)

    metrics = evaluate_link_prediction(
        model=model,
        data=lp_data.data,
        test_edges=lp_data.test_edges,
        all_triples_set=lp_data._all_triples_set,
        num_nodes=lp_data.num_nodes,
        device=device,
    )

    print("\n链接预测评估结果:")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"  Hits@1:    {metrics['hits@1']:.3f}")
    print(f"  Hits@3:    {metrics['hits@3']:.3f}")
    print(f"  Hits@10:   {metrics['hits@10']:.3f}")

    _run_inference_queries_full(model, lp_data, device, args)


def _run_streaming_mode(args):
    """流式子图采样模式 —— 不加载全量三元组，边查边训。"""
    print("\n" + "=" * 60)
    print("  流式子图采样链接预测 (边查边训)")
    print("=" * 60)

    # ── Phase 1+2: 构建流式数据集 + 采样器 ──
    streaming_data, sampler, streamer = _build_streaming_data_and_sampler(args)

    print("\n数据集统计:")
    for k, v in streaming_data.statistics().items():
        print(f"  {k}: {v}")

    # ── Phase 3: 构建模型 ──
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 流式模式：使用 fixed_dim=hidden_dim 随机特征，模型输入维度 = 隐藏维度
    model = LinkPredictionGCN(
        in_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_relations=streaming_data.num_relations,
        dropout=0.2,
    )
    print(f"\n模型架构:\n{model}")
    print(f"设备: {device}")
    print(f"训练边: {len(streaming_data.train_edges)}")
    print(f"验证边: {len(streaming_data.val_edges)}")
    print(f"测试边: {len(streaming_data.test_edges)}")
    print(f"子图采样: num_hops={args.num_hops}, max_neighbors={args.max_neighbors}")

    # ── Phase 4: 训练 ──
    print("\n" + "-" * 40)
    print("开始流式子图采样训练...")
    print("-" * 40)

    model = train_link_prediction_streaming(
        model=model,
        sampler=sampler,
        streaming_data=streaming_data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size_train,
        num_negatives=args.num_negatives,
        log_interval=args.log_interval,
        val_batches=args.val_batches,
        device=device,
    )

    # ── Phase 5: 测试评估 ──
    print("\n" + "-" * 40)
    print("测试集近似评估 (子图内 Hits@K)...")
    print("-" * 40)

    metrics = evaluate_link_prediction_streaming(
        model=model,
        sampler=sampler,
        streaming_data=streaming_data,
        batch_size=args.batch_size_train,
        num_negatives=args.num_negatives,
        num_batches=args.eval_batches,
        device=device,
    )

    print("\n链接预测近似评估结果 (子图内):")
    print(f"  val_loss:  {metrics['val_loss']:.4f}")
    print(f"  Hits@1:    {metrics['hits@1']:.3f}")
    print(f"  Hits@3:    {metrics['hits@3']:.3f}")
    print(f"  Hits@10:   {metrics['hits@10']:.3f}")
    print("  (注: 流式模式下 Hits@K 在采样子图内计算，非全局指标)")

    # ── Phase 6: 推理 ──
    _run_inference_queries_streaming(model, sampler, streaming_data, device, args)

    # 清理
    sampler.shutdown()
    print("\n采样器已关闭。")


def main():
    args = _parse_args()

    print("=" * 60)
    print(f"  知识图谱链接预测 Demo (mode={args.mode})")
    print("=" * 60)

    if args.mode == "demo":
        _run_demo_mode(args)
    elif args.mode == "es":
        _run_es_mode(args)
    elif args.mode == "streaming":
        _run_streaming_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\n" + "=" * 60)
    print("  Demo 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

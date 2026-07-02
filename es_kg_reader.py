"""
Elasticsearch 知识图谱数据读取与训练脚本。

这是 es_kg_reader 的命令行入口与示例主程序，所有模块已重构至 src/ 子包：
- src/core/  : ESConfig, Triple, BatchProgress, logger
- src/es/    : ESKnowledgeGraphReader, IDNameResolver, ESTripletStreamer, KGVocabulary
- src/graph/ : TripleToDatasetConverter, KGNeighborLoaderAdapter, AsyncSubgraphSampler
- src/model/ : FaultGCN, FaultLabelBuilder, split_masks, train, evaluate
- src/pipeline/: StreamingTrainingPipeline, KGTrainInferPipeline, topk_fault_diagnosis

用法:
    python es_kg_reader.py --mode streaming --index knowledge_entity_relation_index
    python es_kg_reader.py --mode legacy
    python es_kg_reader.py --mode full
"""

from __future__ import annotations

import logging

# 从重构后的 src 子包导入所有公共 API
from src import (
    # core
    ESConfig, Triple, BatchProgress, logger,
    # es
    ESKnowledgeGraphReader, IDNameResolver, ESTripletStreamer, KGVocabulary, create_es_client,
    # graph
    TripleToDatasetConverter, KGNeighborLoaderAdapter, AsyncSubgraphSampler,
    # model
    FaultGCN, FaultLabelBuilder, split_masks, train, evaluate,
    # pipeline
    StreamingTrainingPipeline, KGTrainInferPipeline, topk_fault_diagnosis,
)


# -------------------- 命令行参数解析 --------------------


def _parse_args():
    """解析命令行参数。"""
    import argparse

    parser = argparse.ArgumentParser(description="ES 知识图谱读取与训练")
    parser.add_argument(
        "--mode", choices=["legacy", "full", "streaming"], default="streaming",
        help="legacy: 传统scroll模式 | full: NeighborLoader全图模式 | streaming: 异步流式边查边训",
    )
    parser.add_argument(
        "--index", default=["knowledge_entity_relation_index"], nargs="+",
        help="ES 索引名（支持多个）",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prefetch", type=int, default=2, help="异步预取队列大小")
    parser.add_argument("--head-field", default="srcEntityId")
    parser.add_argument("--relation-field", default="relationTypeId")
    parser.add_argument("--tail-field", default="dstEntityId")

    parser.add_argument("--graph-id", default="992504969637961728", help="按 graphId 过滤（如 980044155496734720）")
    parser.add_argument("--ontology-id", default="992355151124930560", help="关系类型 ontologyId")

    parser.add_argument("--entity-index", default="knowledge_entity_index", help="实体索引名称")
    parser.add_argument("--entity-id-field", default="entityId", help="实体索引中的 ID 字段名")
    parser.add_argument("--entity-name-field", default="name",
                        help="实体索引中的名称字段名（如 name / entityName / displayName）")
    parser.add_argument("--relation-type-index", default="knowledge_entity_type_relation_index",
                        help="关系类型索引名称")
    parser.add_argument("--relation-type-id-field", default="relationTypeId", help="关系类型索引中的 ID 字段名")
    parser.add_argument("--relation-type-name-field", default="name",
                        help="关系类型索引中的名称字段名（如 name / typeName / label）")
    parser.add_argument("--no-resolve", action="store_true", help="禁用 ID 到名称的解析（直接使用原始 ID）")
    parser.add_argument("--resolve-debug", action="store_true", help="启用解析器调试模式：采样索引文档并输出字段名")

    parser.add_argument("--fault-relations", nargs="+",
                        default=["类型为", "类别为", "is_fault", "故障类型", "由...引起"],
                        help="标识故障分类的关系名称列表")

    parser.add_argument("--gcn-epochs", type=int, default=200, help="GCN 训练轮数")
    parser.add_argument("--no-infer", action="store_true", help="跳过推理阶段")
    parser.add_argument("--symptoms", nargs="+", default=["急加速时进气管有嘶嘶声"],
                        help="推理时输入的症状节点名称（多个用空格分隔）")
    parser.add_argument("--model-save", default="./models/kg_fault_model.pt", help="模型保存路径")
    return parser.parse_args()


# -------------------- 模式 A: Legacy 模式 --------------------


def _run_legacy_mode(args) -> None:
    """模式 A: 传统 scroll/scan 模式（兼容旧代码 + 推理）。"""
    config = ESConfig()
    reader = ESKnowledgeGraphReader(config)
    try:
        logger.info("运行模式: legacy (scroll/scan)")
        logger.info("发现索引: %s", reader.list_indices())

        triples = reader.fetch_triples(
            graph_id=args.graph_id,
            ontology_id=args.ontology_id,
            entity_index="knowledge_entity_index",
            relation_index="knowledge_entity_relation_index",
            relation_type_index="knowledge_entity_type_relation_index",
            batch_size=args.batch_size,
        )

        if not triples:
            logger.error("未提取到三元组")
            return

        logger.info("共提取 %d 个三元组，预览前 5 条:", len(triples))
        for t in triples[:5]:
            logger.info("  (%s) --[%s]--> (%s)", t.head, t.relation, t.tail)

        converter = TripleToDatasetConverter(triples)
        data = converter.to_data()

        data.train_mask, data.val_mask, data.test_mask = split_masks(data.num_nodes)
        model = FaultGCN(in_dim=data.num_features, hidden_dim=32)
        train(model, data)
        evaluate(model, data)

        # 推理：Top-K 故障诊断
        if not args.no_infer and converter.fault_nodes:
            symptoms = args.symptoms
            if symptoms is None:
                normal_nodes = [
                    n for n, lbl in converter.labels.items() if lbl == 0
                ][:3]
                symptoms = normal_nodes if normal_nodes else ["泵_01"]
                logger.info("未指定 --symptoms，自动选取: %s", symptoms)

            logger.info("\n执行 Top-K 故障诊断推理 ...")
            try:
                results = topk_fault_diagnosis(
                    model, data,
                    converter.node_to_idx,
                    converter.fault_nodes,
                    symptoms,
                    top_k=min(5, len(converter.fault_nodes)),
                )
                logger.info("=" * 50)
                logger.info("  Top-K 故障诊断结果")
                logger.info("  输入症状: %s", ", ".join(symptoms))
                for rank, (fault, score) in enumerate(results, start=1):
                    logger.info("  %d. %-30s similarity=%.4f", rank, fault, score)
                logger.info("=" * 50)
            except ValueError as e:
                logger.warning("推理失败: %s", e)
    finally:
        reader.close()


# -------------------- 模式 B & C: 公共组件 --------------------


def _build_resolver(args, config: ESConfig) -> IDNameResolver | None:
    """构建 ID→名称解析器。

    如果 args.no_resolve 为 True，返回 None。
    """
    if args.no_resolve:
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
        batch_size=args.batch_size,
        extra_query={"match": {"graphId": args.graph_id}} if args.graph_id else None,
        debug=args.resolve_debug,
    )
    resolver.build_relation_type_map(
        relation_type_index=args.relation_type_index,
        ontology_id=args.ontology_id,
        id_field=args.relation_type_id_field,
        name_field=args.relation_type_name_field,
        batch_size=args.batch_size,
        debug=args.resolve_debug,
    )
    logger.info(
        "ID→名称解析器就绪: 实体=%d 种, 关系类型=%d 种",
        len(resolver.entity_map), len(resolver.relation_type_map),
    )
    if not resolver.is_ready:
        logger.error(
            "❌ 解析器未就绪 (entity_map=%d, relation_type_map=%d)！"
            "词汇表中的实体/关系名称将保持为原始 ID。"
            "请运行 --resolve-debug 查看索引详情，"
            "或检查 --graph-id / --ontology-id / --entity-name-field 等参数。",
            len(resolver.entity_map), len(resolver.relation_type_map),
        )
    return resolver


def _build_vocabulary_and_streamer(
        args,
        config: ESConfig,
        resolver: IDNameResolver | None
):
    """构建全局词汇表和 ESTripletStreamer。

    Returns:
        (streamer, vocab, extra_filters) 三元组
    """
    index_names = args.index if len(args.index) > 1 else args.index[0]
    logger.info("目标索引: %s", index_names)

    streamer = ESTripletStreamer(
        es_hosts=[f"{config.scheme}://{config.host}:{config.port}"],
        index_name=index_names,
        batch_size=args.batch_size,
        prefetch=True,
        checkpoint_dir="./es_checkpoints",
        resolver=resolver,
    )

    logger.info("Phase 1: 构建全局实体/关系词汇表 (已解析为名称)...")
    if resolver is None:
        logger.warning(
            "⚠️  未启用 ID→名称解析！词汇表中将存储原始 ID。"
            "请确认是否需要名称解析（检查 --no-resolve 参数）"
        )
    else:
        logger.info(
            "解析器配置: 实体索引=%s (id=%s, name=%s), 关系类型索引=%s (id=%s, name=%s)",
            args.entity_index, args.entity_id_field, args.entity_name_field,
            args.relation_type_index, args.relation_type_id_field, args.relation_type_name_field,
        )

    vocab = KGVocabulary(checkpoint_dir="./vocab_checkpoints")
    vocab._entity_index_hint = args.entity_index
    vocab._relation_type_index_hint = args.relation_type_index

    extra_filters = None
    if args.graph_id:
        extra_filters = {"match": {"graphId": args.graph_id}}
        logger.info("启用 graphId 过滤: %s", args.graph_id)

    vocab.build_from_streamer(
        streamer,
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        extra_filters=extra_filters,
    )
    vocab.save("./vocab_checkpoints/vocab.json")

    sample_entities = list(vocab.entity2idx.keys())[:5]
    sample_relations = list(vocab.relation2idx.keys())[:3]
    logger.info("解析后实体示例: %s", sample_entities)
    logger.info("解析后关系类型示例: %s", sample_relations)

    return streamer, vocab, extra_filters


def _auto_select_symptoms(vocab: KGVocabulary, fault_nodes: list[str]) -> list[str]:
    """自动选取非故障节点作为推理症状。"""
    all_entity_names = list(vocab.entity2idx.keys())
    non_fault_set = set(all_entity_names) - set(fault_nodes)
    symptoms = list(non_fault_set)[:3] if non_fault_set else all_entity_names[:2]
    logger.info("未指定 --symptoms，自动选取示例: %s", symptoms)
    return symptoms


def _run_inference(
        pipeline, data, vocab: KGVocabulary,
        fault_nodes: list[str], symptoms: list[str],
) -> None:
    """执行 Top-K 故障诊断推理并打印结果。"""
    logger.info("执行 Top-K 故障诊断推理 (symptoms=%s) ...", symptoms)
    try:
        results = pipeline.infer_topk(
            model=pipeline.model,
            data=data,
            symptoms=symptoms,
            node_to_idx=vocab.entity2idx,
            fault_nodes=fault_nodes,
            top_k=min(5, len(fault_nodes)),
        )
        logger.info("=" * 50)
        logger.info("  Top-K 故障诊断结果")
        logger.info("  输入症状: %s", ", ".join(symptoms))
        for rank, (fault, score) in enumerate(results, start=1):
            logger.info("  %d. %-30s similarity=%.4f", rank, fault, score)
        logger.info("=" * 50)
    except ValueError as e:
        logger.warning("推理失败: %s", e)


def _run_stream_based_mode(
        args,
        streamer: ESTripletStreamer,
        vocab: KGVocabulary,
        extra_filters,
        mode_label: str,
) -> None:
    """模式 B/C 公共逻辑：构建全图 → 训练 → 评估 → 推理。

    Args:
        mode_label: 用于日志的友好名称（如 "全图模式" / "流式模式"）
    """
    logger.info("模式: %s → 构建 edge_index + 全图 GCN 训练 + 推理", mode_label)

    # Phase 2: 构建全图 edge_index
    logger.info("Phase 2: 构建全图 edge_index (search_after 逐批加载)...")
    adapter = KGNeighborLoaderAdapter.from_streamer(
        streamer=streamer,
        vocab=vocab,
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
    )
    data = adapter.data

    # Phase 3: 构建故障标签
    logger.info("Phase 3: 识别故障节点并构建标签 ...")
    label_builder = FaultLabelBuilder(
        vocab=vocab,
        fault_relations=args.fault_relations,
    )
    y, fault_nodes = label_builder.build_from_streamer(
        streamer=streamer,
        head_field=args.head_field,
        relation_field=args.relation_field,
        tail_field=args.tail_field,
        extra_filters=extra_filters,
    )
    data.y = y

    if not fault_nodes:
        logger.error(
            "未识别到任何故障节点，无法训练。"
            "请通过 --fault-relations 指定正确的故障分类关系。"
        )
        return

    # Phase 4: 划分 train/val/test
    data.train_mask, data.val_mask, data.test_mask = split_masks(data["entity"].num_nodes)
    logger.info(
        "数据集划分: train=%d, val=%d, test=%d",
        data.train_mask.sum().item(),
        data.val_mask.sum().item(),
        data.test_mask.sum().item(),
    )

    # Phase 5: 训练
    logger.info("Phase 4: 全图 GCN 训练 (epochs=%d) ...", args.gcn_epochs)
    pipeline = KGTrainInferPipeline(in_dim=adapter.embedding_dim, hidden_dim=32)
    pipeline.train_full_graph(
        data=data,
        y=y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        epochs=args.gcn_epochs,
        verbose=True,
    )

    # Phase 6: 评估
    logger.info("Phase 5: 测试集评估 ...")
    metrics = pipeline.evaluate(
        data=data,
        y=y,
        test_mask=data.test_mask,
        model=pipeline.model,
    )
    logger.info(
        "测试集结果: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
    )

    # Phase 7: 推理 (Top-K 故障诊断)
    if not args.no_infer and fault_nodes:
        symptoms = args.symptoms
        if symptoms is None:
            symptoms = _auto_select_symptoms(vocab, fault_nodes)
        _run_inference(pipeline, data, vocab, fault_nodes, symptoms)

    # 保存模型
    if args.model_save:
        pipeline.save_model(args.model_save)


# -------------------- 入口 --------------------


def main() -> None:
    """示例主程序：支持 3 种模式的知识图谱读取与训练。

    模式 A: 传统 scroll/scan 模式（兼容旧代码）
    模式 B: search_after 全图模式 → NeighborLoader 训练（图较小，内存可容纳）
    模式 C: search_after 流式模式 → 异步子图采样训练（图巨大，"边查边训"）
    """
    args = _parse_args()

    if args.mode == "legacy":
        _run_legacy_mode(args)
        return

    # 模式 B & C: search_after 流式架构
    config = ESConfig()
    resolver = _build_resolver(args, config)
    streamer, vocab, extra_filters = _build_vocabulary_and_streamer(args, config, resolver)

    mode_label = "全图模式" if args.mode == "full" else "流式模式"
    _run_stream_based_mode(args, streamer, vocab, extra_filters, mode_label)


if __name__ == "__main__":
    main()

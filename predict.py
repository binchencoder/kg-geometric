#!/usr/bin/env python3
"""
知识图谱模型统一推理引擎

支持自动发现 models/ 目录下所有模型，并为每种模型动态构建推理流程。

支持的模型类型：
- kg_fault_model: 基于 FaultRGCN 的知识图谱故障诊断模型

使用方式：
    python predict.py
    python predict.py --instance "振动过高\n温度过高"
    python predict.py --models-dir ./models --top-k 5 --instance "电流过高"
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.config import InferenceConfig, RelationMappingConfig
from src.model.rgcn import FaultRGCN
from src.pipeline.inference import DiagnosisResult, infer_from_text

if TYPE_CHECKING:
    from src.dataset.triple_dataset import KGTripleDataset

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("KGInference")


# ============================================================
# 1. 工具函数
# ============================================================


def load_model(model_dir: str, params: Optional[dict] = None) -> Dict[str, Any]:
    """    加载模型目录下所有可用模型，返回模型信息汇总。

    Parameters
    ----------
    model_dir : str
        模型文件目录，默认为 "models"。
    params : Optional[dict]
        可选参数（保留兼容，当前未使用）。

    Returns
    -------
    Dict[str, Any]
        包含模型列表与元信息的字典：
        - "models" : List[dict]   每个模型的 {"name", "path", "type", "handler"}
        - "total"  : int          发现模型总数
        - "dir"    : str          模型目录绝对路径
    """
    _ = params  # 保留兼容
    models = _discover_models(model_dir)
    if not models:
        logger.warning("目录 %s 中未发现任何模型", model_dir)
    else:
        logger.info("共发现 %d 个模型: %s", len(models), [m["name"] for m in models])
    return {
        "models": models,
        "total": len(models),
        "dir": str(Path(model_dir).resolve()),
    }


def generate_unique_id() -> str:
    """生成唯一 ID，用于临时目录等场景。
    """
    timestamp = int(time.time() * 1000)
    rand_num = random.randint(0, 1000)
    unique_id = f"{timestamp}{rand_num}"
    return unique_id


def convert_percent(num: float) -> str:
    """将小数格式化为百分比字符串。

    Parameters
    ----------
    num : float
        浮点数分数（0-1 之间）。

    Returns
    -------
    str
        百分比字符串，如 "85.5%"。
    """
    percent = np.round(num * 100, 2)
    return f"{percent}%"


# ============================================================
# 2. 模型处理器注册机制（可插拔，便于扩展新模型类型）
# ============================================================


class ModelHandler:
    """模型处理器基类。

    每种模型类型对应一个处理器子类，负责完整的推理生命周期：
    1. load()        — 从文件加载模型及附属资源
    2. preprocess()  — 原始文本 → 模型可接受的输入
    3. infer()       — 执行模型正向传播
    4. postprocess() — 原始结果 → JSON 字符串

    子类只需重写这四个方法即可接入推理引擎。
    """

    # 模型文件名匹配模式（用于自动发现时的匹配）
    model_pattern: str = ""

    def load(self, model_path: str) -> Any:
        """从文件加载模型。

        Parameters
        ----------
        model_path : str
            模型文件路径。

        Returns
        -------
        Any
            加载后的模型对象（具体类型由子类定义）。
        """
        raise NotImplementedError

    def preprocess(self, instance: str) -> Any:
        """预处理原始输入文本。

        Parameters
        ----------
        instance : str
            原始输入文本。

        Returns
        -------
        Any
            模型可接受的输入（具体类型由子类定义）。
        """
        raise NotImplementedError

    def infer(self, model: Any, processed_input: Any) -> Any:
        """执行模型推理。

        Parameters
        ----------
        model : Any
            已加载的模型对象。
        processed_input : Any
            预处理后的输入。

        Returns
        -------
        Any
            原始推理结果。
        """
        raise NotImplementedError

    def postprocess(self, raw_result: Any) -> str:
        """将原始推理结果后处理为 JSON 字符串。

        Parameters
        ----------
        raw_result : Any
            原始推理结果。

        Returns
        -------
        str
            JSON 格式字符串。
        """
        raise NotImplementedError


# 全局处理器注册表
_HANDLER_REGISTRY: Dict[str, ModelHandler] = {}


def register_handler(name: str, handler: ModelHandler) -> None:
    """注册一个模型处理器。

    Parameters
    ----------
    name : str
        处理器注册名，与模型文件名前缀匹配。
    handler : ModelHandler
        处理器实例。
    """
    if name in _HANDLER_REGISTRY:
        logger.warning("处理器 '%s' 已存在，将被覆盖", name)
    _HANDLER_REGISTRY[name] = handler
    logger.debug("已注册模型处理器: %s -> %s", name, handler.__class__.__name__)


def _get_handler(model_name: str) -> Optional[ModelHandler]:
    """根据模型文件名查找匹配的处理器。

    按注册名前缀匹配，每个处理器返回独立实例，避免并发状态污染。
    """
    for handler_name, handler in _HANDLER_REGISTRY.items():
        if model_name.startswith(handler_name) or handler_name in model_name:
            return handler
    return None


# ============================================================
# 3. 知识图谱故障诊断模型处理器
# ============================================================


class DiagnosisModelHandler(ModelHandler):
    """知识图谱故障诊断模型处理器（仅支持 FaultRGCN checkpoint）。

    checkpoint 由 ``train.py`` 训练生成，已内嵌完整图结构（dataset /
    graph_data / node_to_idx / fault_nodes），推理时无需再连接 ES。

    采用与 ``demo/fault_diagnosis.py`` 一致的四阶段推理：语义匹配 → 故障定位
    → 答案生成 → 结果组装，除 Top-K 故障定位外，还输出每个故障的
    可能原因（causes）、维修措施（actions）、所需工具（tools）。

    输入：换行分隔的症状文本（如 "振动过高\\n温度过高"）
    输出：含故障定位与 causes/actions/tools 分析的 JSON
    """

    model_pattern = "diagnosis_model"

    def __init__(
            self, device: str = "cpu",
            top_k: int = 3,
            symptom_relation: str | None = None,
    ):
        """
        Parameters
        ----------
        device : str
            推理设备，支持 "cpu" / "cuda" / "cuda:0" 等。
        top_k : int
            返回 Top-K 故障诊断结果。
        symptom_relation : str | None
            症状→故障的语义匹配关系名。为 None 时从 config.yaml 的
            relation_mapping.symptoms 读取（默认 "表现为"）。
        """
        self.device = device
        self.top_k = top_k
        self.symptom_relation = symptom_relation
        self.dataset: Optional["KGTripleDataset"] = None
        self.data: Optional[Any] = None

    def _detect_model_type(self, checkpoint: dict) -> str:
        """根据 checkpoint 字段判断模型类型（仅支持 FaultRGCN）。"""
        if "num_nodes" in checkpoint and "num_relations" in checkpoint:
            return "rgcn"
        raise KeyError(
            "无法识别的 checkpoint 格式，缺少必要字段 (num_nodes/num_relations): "
            f"got keys={sorted(checkpoint.keys())}"
        )

    def load(self, model_path: str) -> Tuple[Any, dict]:
        """加载 FaultRGCN 模型及内嵌的图结构（无需连接 ES）。

        Parameters
        ----------
        model_path : str
            .pt 模型文件路径。

        Returns
        -------
        Tuple[nn.Module, dict]
            (模型实例, 模型元信息)。
        """
        logger.info("正在加载模型: %s", model_path)

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"模型文件格式异常，期望 dict，实际为 {type(checkpoint).__name__}"
            )

        model_type = self._detect_model_type(checkpoint)
        hidden_dim = checkpoint["hidden_dim"]

        if model_type == "rgcn":
            # ---- 加载 FaultRGCN（由 train.py 训练）----
            num_nodes = int(checkpoint["num_nodes"])
            num_relations = int(checkpoint["num_relations"])
            num_layers = int(checkpoint.get("num_layers", 2))
            dropout = float(checkpoint.get("dropout", 0.3))

            model = FaultRGCN(
                num_nodes=num_nodes,
                num_relations=num_relations,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(self.device)
            model.eval()

            # 直接从 checkpoint 恢复图结构，无需再连 ES
            if "graph_data" not in checkpoint:
                raise KeyError(
                    "checkpoint 缺少 graph_data 字段，请使用最新 train.py 重新训练"
                    "以将图结构写入模型文件"
                )
            self.data = checkpoint["graph_data"]

            # 完整数据集对象（含图遍历映射），用于提取原因/措施/工具
            if "dataset" not in checkpoint:
                raise KeyError(
                    "checkpoint 缺少 dataset 字段，请使用最新 train.py 重新训练"
                    "以将图遍历映射写入模型文件（支持 causes/actions/tools 分析）"
                )
            self.dataset = checkpoint["dataset"]
            assert self.dataset is not None
            # 将图遍历所需的边张量移动到推理设备
            self.dataset.edge_index = self.dataset.edge_index.to(self.device)
            self.dataset.edge_type = self.dataset.edge_type.to(self.device)

            # FaultRGCN 使用内部的 node_emb 作为输入，无需外部 x
            # 若数据集节点数与模型不一致，给出警告但继续运行
            data_num_nodes = int(self.data.num_nodes)
            if data_num_nodes != num_nodes:
                logger.warning(
                    "图结构节点数与模型不匹配: graph=%d, model=%d。"
                    "推理时将使用现有图结构（嵌入由模型内部 node_emb 提供）。",
                    data_num_nodes, num_nodes,
                )

            meta = {
                "model_type": "rgcn",
                "num_nodes": num_nodes,
                "num_relations": num_relations,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "device": self.device,
                "fault_nodes": checkpoint["fault_nodes"],
            }
            logger.info(
                "FaultRGCN 加载成功: nodes=%d, relations=%d, hidden=%d, layers=%d"
                "（图结构来自本地 checkpoint，未连接 ES）",
                num_nodes, num_relations, hidden_dim, num_layers,
            )

        else:
            raise ValueError(
                f"不支持的模型类型 '{model_type}'，当前仅支持 FaultRGCN checkpoint"
            )

        if self.device.startswith("cuda") and self.data is not None:
            self.data = self.data.to(self.device)

        return model, meta

    def preprocess(self, instance: str) -> str:
        """将输入文本解析为查询字符串。

        支持换行分隔的多个症状（如 "振动过高\\n温度过高"），
        交由 ``infer_from_text`` 做字符级语义匹配。

        Parameters
        ----------
        instance : str
            原始输入文本。

        Returns
        -------
        str
            去空白后的查询文本。

        Raises
        ------
        ValueError
            输入为空时抛出。
        """
        text = str(instance).strip()
        if not text:
            raise ValueError("输入为空，请提供至少一个症状")

        logger.info("预处理完成: 输入症状文本 (长度=%d)", len(text))
        return text

    def infer(
            self,
            model: Tuple[FaultRGCN, dict],
            processed_input: str,
    ) -> Optional[DiagnosisResult]:
        """基于症状文本执行四阶段故障诊断推理（含原因/措施/工具分析）。

        流程与 ``demo/fault_diagnosis.py`` 一致：语义匹配 → 故障定位 →
        答案生成（causes/actions/tools）→ 结果组装。

        Parameters
        ----------
        model : Tuple[FaultRGCN, dict]
            (模型实例, 元信息)。
        processed_input : str
            症状查询文本。

        Returns
        -------
        Optional[DiagnosisResult]
            完整诊断结果；未匹配到任何症状时返回 None。
        """
        rgcn_model, _ = model

        if self.dataset is None or self.data is None:
            raise RuntimeError("模型未正确加载，缺少图结构数据")

        try:
            result = infer_from_text(
                model=rgcn_model,
                dataset=self.dataset,
                query_text=processed_input,
                symptom_relation=self.symptom_relation,
                top_k_symptoms=5,
                top_k_faults=self.top_k,
                device=self.device,
            )
        except ValueError as e:
            logger.warning("推理时图谱中未找到匹配节点: %s", e)
            return None

        logger.info(
            "推理完成: 最佳故障=%s, %d 原因, %d 措施, %d 工具",
            result.best_fault,
            len(result.causes), len(result.actions), len(result.tools),
        )
        return result

    def postprocess(self, raw_result: Optional[DiagnosisResult]) -> str:
        """将诊断结果格式化为 JSON 字符串（含 causes/actions/tools）。

        Parameters
        ----------
        raw_result : Optional[DiagnosisResult]
            四阶段推理结果。

        Returns
        -------
        str
            JSON 字符串，包含故障定位、可能原因、维修措施、所需工具等。
        """
        if raw_result is None or not raw_result.fault_candidates:
            return json.dumps(
                {"diagnosis": [], "message": "未找到匹配的故障诊断结果"},
                ensure_ascii=False,
            )

        best_fault = raw_result.best_fault
        best_score = raw_result.fault_candidates[0][1]

        diagnosis_list = []
        for rank, (fault, score) in enumerate(raw_result.fault_candidates, 1):
            diagnosis_list.append({
                "rank": rank,
                "fault": fault,
                "confidence": convert_percent(score),
                "score": round(float(score), 4),
            })

        result = {
            "diagnosis": diagnosis_list,
            "top_fault": best_fault,
            "top_confidence": convert_percent(float(best_score)),
            "causes": raw_result.causes,
            "actions": raw_result.actions,
            "tools": raw_result.tools,
            "system": raw_result.system,
            "category": raw_result.category,
            "alternative_faults": [
                {
                    "fault": alt["fault"],
                    "confidence": convert_percent(alt["confidence"]),
                    "score": round(float(alt["confidence"]), 4),
                    "causes": alt.get("causes", []),
                    "actions": alt.get("actions", []),
                    "tools": alt.get("tools", []),
                }
                for alt in raw_result.alternative_faults
            ],
        }
        return json.dumps(result, ensure_ascii=False)


# ============================================================
# 4. 推理入口
# ============================================================


def predict(model_info: dict, instance: str) -> str:
    """对单个模型执行推理，返回 JSON 结果字符串。

    参数: (model, instance) — model 为模型信息字典, instance 为输入文本。
    返回: JSON 字符串。

    Parameters
    ----------
    model_info : dict
        单个模型信息字典，必须包含 {"name", "path", "handler"} 字段。
    instance : str
        输入文本，格式由模型类型决定。

    Returns
    -------
    str
        JSON 格式的推理结果字符串。异常时返回包含 error 字段的 JSON。

    Raises
    ------
    ValueError
        model_info 缺少 handler 或 handler 无效时抛出。
    """
    model_name = model_info.get("name", "unknown")
    handler = model_info.get("handler")

    if handler is None:
        raise ValueError(
            f"模型 '{model_name}' 缺少 handler，请确认处理器已正确注册"
        )
    if not isinstance(handler, ModelHandler):
        raise ValueError(
            f"模型 '{model_name}' 的 handler 类型不正确: {type(handler).__name__}"
        )

    start_time = time.time()

    try:
        # -------- 1. 预处理 --------
        processed = handler.preprocess(instance)

        # -------- 2. 加载模型 --------
        model = handler.load(model_info["path"])

        # -------- 3. 执行推理 --------
        raw_result = handler.infer(model, processed)

        # -------- 4. 后处理 --------
        result_json = handler.postprocess(raw_result)

        elapsed = time.time() - start_time
        logger.info("模型 '%s' 推理成功, 耗时 %.3fs", model_name, elapsed)
        return result_json

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("模型 '%s' 推理失败 (%.3fs): %s", model_name, elapsed, e)
        return json.dumps(
            {"error": str(e), "model": model_name, "diagnosis": []},
            ensure_ascii=False,
        )


def _discover_models(models_dir: str) -> List[dict]:
    """自动遍历 models/ 目录，发现所有可推理的模型文件。

    对每个模型文件自动匹配注册的处理器，未匹配的文件记录警告并跳过。

    Parameters
    ----------
    models_dir : str
        模型目录路径。

    Returns
    -------
    List[dict]
        模型信息列表，每项为 {"name", "path", "type", "handler"}。
    """
    dir_path = Path(models_dir)
    if not dir_path.exists():
        logger.warning("模型目录不存在: %s", dir_path)
        return []
    if not dir_path.is_dir():
        logger.error("路径存在但不是目录: %s", dir_path)
        return []

    discovered: List[dict] = []
    supported_extensions = {".pt", ".pth", ".bin", ".ckpt"}

    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix not in supported_extensions:
            continue

        model_name = file_path.stem
        handler = _get_handler(model_name)

        if handler is not None:
            model_info = {
                "name": model_name,
                "path": str(file_path.resolve()),
                "type": handler.__class__.__name__,
                "handler": handler,
            }
            discovered.append(model_info)
            logger.info(
                "✓ 发现模型: %-30s 处理器: %s",
                model_name, handler.__class__.__name__,
            )
        else:
            logger.warning(
                "✗ 模型 '%s' 未匹配处理器，已跳过 (已注册: %s)",
                model_name,
                list(_HANDLER_REGISTRY.keys()) or "(无)",
            )

    if not discovered and list(dir_path.iterdir()):
        logger.warning(
            "目录 %s 中存在文件但均未匹配处理器，请确认处理器是否正确注册",
            dir_path,
        )

    return discovered


def predict_all(models_dir: str, instance: str) -> Dict[str, str]:
    """对所有已发现模型执行批量推理。

    Parameters
    ----------
    models_dir : str
        模型目录路径。
    instance : str
        输入文本（所有模型共用）。

    Returns
    -------
    Dict[str, str]
        {模型名称: JSON 结果字符串} 的映射。
    """
    model_info = load_model(models_dir)
    models = model_info.get("models", [])

    if not models:
        logger.error("没有可用的模型，推理中止")
        return {}

    all_results: Dict[str, str] = {}
    for mi in models:
        try:
            result = predict(mi, instance)
            all_results[mi["name"]] = result
        except Exception as e:
            logger.error("模型 '%s' 批量推理异常: %s", mi["name"], e)
            all_results[mi["name"]] = json.dumps(
                {"error": str(e), "model": mi["name"], "diagnosis": []},
                ensure_ascii=False,
            )

    logger.info(
        "批量推理完成: %d/%d 个模型成功",
        sum(1 for v in all_results.values() if "error" not in json.loads(v)),
        len(models),
    )
    return all_results


# ============================================================
# 5. 命令行入口
# ============================================================


def _build_arg_parser(cfg: InferenceConfig) -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="知识图谱模型统一推理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py
  python predict.py --instance "振动过高\\n温度过高"
  python predict.py --models-dir ./models --top-k 5 --instance "电流过高"
  python predict.py --device cuda --instance "噪音异常"
        """,
    )
    parser.add_argument(
        "--models-dir",
        default=cfg.models_dir,
        help=f"模型目录路径 (默认来自 config.yaml: {cfg.models_dir})",
    )
    parser.add_argument(
        "--instance",
        default=cfg.instance,
        help="输入实例文本，多个症状以换行符分隔 (默认来自 config.yaml)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg.top_k,
        help=f"返回 Top-K 故障诊断结果 (默认来自 config.yaml: {cfg.top_k})",
    )
    parser.add_argument(
        "--device",
        default=cfg.device,
        help=f"推理设备 (默认来自 config.yaml: {cfg.device})",
    )
    return parser


def main() -> None:
    """主入口。"""
    cfg = InferenceConfig.default()
    parser = _build_arg_parser(cfg)
    args = parser.parse_args()

    # ---- 注册模型处理器 ----
    # symptom_relation 不再通过参数传递，由处理器从 config.yaml 的
    # relation_mapping.symptoms 读取。
    register_handler(
        "diagnosis_model",
        DiagnosisModelHandler(
            device=args.device,
            top_k=args.top_k,
            symptom_relation=RelationMappingConfig.default().mapping.get("symptoms"),
        ),
    )

    # ---- Phase 1: 发现并加载模型 ----
    model_info = load_model(args.models_dir)
    models = model_info.get("models", [])

    if not models:
        logger.error(
            "没有可用的模型。请确认:\n"
            "  1. 模型目录 '%s' 中存在 .pt/.pth 文件\n"
            "  2. 已为模型类型注册对应的处理器",
            args.models_dir,
        )
        return

    # ---- 打印概览 ----
    print("\n" + "=" * 64)
    print("  知识图谱模型统一推理引擎")
    print("=" * 64)
    print(f"  模型目录 : {model_info['dir']}")
    print(f"  模型数量 : {model_info['total']}")
    print(f"  模型列表 : {', '.join(m['name'] for m in models)}")
    print(f"  输入实例 : {repr(args.instance)}")
    print(f"  Top-K    : {args.top_k}")
    print(f"  设备     : {args.device}")
    print("=" * 64)

    # ---- Phase 2: 单模型逐一推理 ----
    for idx, mi in enumerate(models, 1):
        print(f"\n{'─' * 50}")
        print(f"  [{idx}/{len(models)}] 模型: {mi['name']}")
        print(f"  处理器: {mi['type']} | 路径: {mi['path']}")
        print(f"{'─' * 50}")

        start = time.time()
        result_json = predict(mi, args.instance)
        elapsed = time.time() - start

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            print(f"  原始输出: {result_json}")
            continue

        if "error" in result:
            print(f"  ✗ 推理失败: {result['error']}")
        else:
            print(f"  最佳匹配故障 : {result.get('top_fault', 'N/A')}")
            print(f"  置信度       : {result.get('top_confidence', 'N/A')}")
            causes = result.get("causes", [])
            actions = result.get("actions", [])
            tools = result.get("tools", [])
            if causes:
                print(f"  可能原因     : {'、'.join(causes)}")
            if actions:
                print(f"  维修措施     : {'、'.join(actions)}")
            if tools:
                print(f"  所需工具     : {'、'.join(tools)}")
            print(f"  诊断详情     :")
            for d in result.get("diagnosis", []):
                bar = "█" * max(1, int(float(d.get("confidence", "0%").rstrip("%")) / 5))
                print(f"    Top-{d['rank']}: {d['fault']:<12} {d['confidence']:>8} {bar}")
            alts = result.get("alternative_faults", [])
            if alts:
                print(f"  备选故障     :")
                for alt in alts:
                    extra = []
                    if alt.get("causes"):
                        extra.append(f"原因: {'、'.join(alt['causes'][:3])}")
                    if alt.get("actions"):
                        extra.append(f"措施: {'、'.join(alt['actions'][:3])}")
                    suffix = f" ({'；'.join(extra)})" if extra else ""
                    print(f"    - {alt['fault']} ({alt['confidence']}){suffix}")
        print(f"  耗时         : {elapsed:.3f}s")

    # ---- Phase 3: 全模型批量推理 ----
    print(f"\n{'=' * 64}")
    print("  全模型批量推理汇总")
    print("=" * 64)

    all_results = predict_all(args.models_dir, args.instance)
    for model_name, result_str in all_results.items():
        print(f"\n  [{model_name}]")
        try:
            parsed = json.loads(result_str)
            print(f"  {json.dumps(parsed, ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            print(f"  {result_str}")

    print(f"\n{'=' * 64}")
    print("  推理任务全部完成")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()

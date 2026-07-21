# -*- coding: utf-8 -*-
"""
src/api/inference_server.py
=====================================================================
统一推理 —— HTTP 服务（故障诊断 + 时序趋势预测 + TKGL 链接预测）

基于 Python 标准库 http.server 实现，零额外依赖（无需 FastAPI/Flask）。
启动时按需加载三类模型，随后通过 REST 接口对外提供：

  A. 故障诊断（src/fault/diagnosis/predict.py）
     输入症状文本（换行分隔），复用 DiagnosisModelHandler 的四阶段推理
     （语义匹配 → 故障定位 → 答案生成 → 结果组装），输出含 causes /
     actions / tools 的结构化诊断 JSON。

  B. 时序趋势预测（src/fault/prediction/predict.py）
     加载训练时持久化的异构图与 R-GCN / TGN 权重，对指定时序切片输出
     健康状态、油温趋势与故障风险等级。

  C. TKGL 时序知识图谱链接预测（src/tkgl/predict.py）
     给定 (头, 关系, 时间) 预测 Top-K 尾实体，并支持测试集过滤式 MRR 评测。

-------------------------------------------------------------------
启动方式
-------------------------------------------------------------------
    python -m src.api.inference_server \
        --diagnosis-models-dir ./models \
        --prediction-model-dir ./trained_models/trend \
        --model-path trained_models/tkgl_smallpedia_model.pt \
        --host 0.0.0.0 --port 8000

  （三类能力各自独立，缺省对应参数即不加载该项能力，其余照常工作）

-------------------------------------------------------------------
接口
-------------------------------------------------------------------
1) 健康检查
    GET  /health
    -> {"status": "ok", "device": ..., "diagnosis_models": [...],
        "prediction_loaded": true, "tkgl_loaded": true}

2) 故障诊断（JSON 请求体）
    POST /diagnosis
    {
      "symptoms": "振动过高\n温度过高",  # 症状文本，换行分隔（也可用 "instance"）
      "top_k":    3,                      # 可选，默认服务端配置
      "model":    "diagnosis_model_xxx"   # 可选，默认首个发现的模型
    }
    -> 结构化诊断 JSON（diagnosis / top_fault / causes / actions / tools ...）

3) 时序趋势预测（JSON 请求体）
    POST /trend
    {
      "slice_idx": 100        # 可选，默认最新一个时序切片
    }
    -> {
         "slice_idx": 100,
         "date": "...",
         "features": {"HUFL": ..., ...},
         "health_state": "正常运行", "health_probabilities": {...},
         "rules": [...],
         "future_oil_temperature": 42.13,
         "fault_risk": 0.05, "risk_level": "低风险",
         "suggestion": "..."
       }

4) TKGL 链接预测（JSON 请求体）
    POST /predict
    # 单次推理
    {
      "head":     "Q648" | 123,   # 头实体：Q-ID/P-ID 字符串或整数 ID
      "relation": "P27"  | 7,     # 关系：P-ID 字符串或整数 ID
      "time":     2008,           # 查询年份（整数）
      "topk":     5,              # 可选，默认 5
      "temporal_bias": 50.0,      # 可选，时间邻近性偏置权重
      "temporal_sigma": 8.0       # 可选，偏置高斯衰减尺度
    }
    -> {"query": {"head": "Q648", "relation": "P27", "time": 2008},
        "predictions": [{"tail": "Q42", "tail_id": 12, "score": 1.234}, ...]}

    # 批量推理：queries 为查询列表，每个元素支持独立的 topk / temporal_bias / temporal_sigma
    {
      "queries": [
        {"head": "Q648", "relation": "P27", "time": 2008},
        {"head": "Q5",   "relation": "P19", "time": 2010, "topk": 10},
        ...
      ]
    }
    -> {"results": [
          {"query": {...}, "predictions": [...]},
          {"query": {...}, "predictions": [...]},
          ...
        ]}
    # 单条失败时返回 {"error": "...", "index": <下标>}，不影响其余查询

5) TKGL 评测（JSON 请求体）
    POST /evaluate
    {"num_eval": 2000, "k_neg": 500}   # 均可选
    -> {"mrr": ..., "hits@1": ..., "hits@3": ..., "hits@10": ...}

-------------------------------------------------------------------
说明
-------------------------------------------------------------------
* 诊断模型自动发现 models 目录下 .pt/.pth 文件并逐一加载缓存；
  趋势预测需要 prediction_model_dir 下存在 temporal_kg_graph.pt、
  temporal_diag_rgcn.pth、temporal_tgn.pth；
  TKGL 需要 --model-path 指向已训练的 .pt checkpoint。
* 推理为只读、线程安全（ThreadingHTTPServer + 锁保护）。
* 缺字段 / 未知实体或关系 / 未加载对应能力时返回 400/503，内部异常 500。
"""

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# 项目根目录加入 sys.path，使 `import src...` 等绝对导入可解析
# （以脚本方式运行时 cwd 未必在 path，故显式注入）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.config import (
    TKGLPredictionConfig,
    InferenceConfig,
    TrendPredictionConfig,
    LinkPredictionConfig,
)

from src.fault.diagnosis.predict import (  # noqa: E402
    DiagnosisModelHandler,
    _discover_models,
    register_handler,
)

from src.fault.prediction.predict import (  # noqa: E402
    RGCNFaultDiagnosis,
    build_tgn_oil_temperature_predict,
    HEALTH_MAPPING,
    FEATURE_LIST,
    HIDDEN_DIM,
    HEALTH_NUM,
)
from src.model.tkgl import load_checkpoint  # noqa: E402
from src.tkgl.predict import (  # noqa: E402
    predict_tails,
    _resolve_id,
    evaluate_filtered,
    _build_relation_endpoints,
)

logger = logging.getLogger("InferenceServer")


# 默认症状→故障语义关系名（用于诊断模型处理器）
def _default_relation_mapping() -> dict[str, str]:
    try:
        from src.core.config import RelationMappingConfig
        return RelationMappingConfig.default().mapping
    except Exception:  # noqa: BLE001
        return {}


# 默认推理配置（来自 config.yaml 的 inference）
def _default_inference_config() -> InferenceConfig:
    try:
        from src.core.config import InferenceConfig
        return InferenceConfig.default()
    except Exception:  # noqa: BLE001
        return {}


# 默认链接预测配置（来自 config.yaml 的 inference.link_prediction）
def _link_prediction_config() -> LinkPredictionConfig:
    try:
        from src.core.config import InferenceConfig
        return InferenceConfig.default().link_prediction
    except Exception:  # noqa: BLE001
        return {}


# 默认趋势预测配置（来自 config.yaml 的 inference.trend_prediction）
def _trend_prediction_config() -> TrendPredictionConfig:
    try:
        from src.core.config import InferenceConfig
        return InferenceConfig.default().trend_prediction
    except Exception:  # noqa: BLE001
        return {}


def _tkgl_prediction_config() -> TKGLPredictionConfig:
    try:
        from src.core.config import InferenceConfig
        return InferenceConfig.default().tkgl_prediction
    except Exception:  # noqa: BLE001
        return {}


class InferenceService:
    """封装三类模型的加载与单次推理，供 HTTP handler 调用。

    线程安全：模型参数只读；诊断与趋势推理均在 ``torch.no_grad()`` 下执行，
    并加锁保护共享模型的并发前向传播。
    """

    def __init__(
            self,
            diagnosis_models_dir: str,
            prediction_model_dir: str,
            tkgl_model_path: "str | None" = None,
            device: str = "cpu",
            top_k: int = 3,
            symptom_relation: "str | None" = None,
    ):
        self.device = device
        self.default_top_k = top_k
        self.symptom_relation = symptom_relation
        self._lock = threading.Lock()

        # 诊断模型缓存: name -> (model, meta, data, dataset)
        self._diag_cache: dict = {}
        # 趋势预测资源
        self._pred: "dict | None" = None
        # TKGL 链接预测资源
        self._tkgl: "dict | None" = None

        # 注册诊断模型处理器，使 _discover_models 能按文件名前缀自动匹配
        register_handler(
            "diagnosis_model",
            DiagnosisModelHandler(
                device=self.device,
                top_k=self.default_top_k,
                symptom_relation=self.symptom_relation,
            ),
        )

        self._load_diagnosis(diagnosis_models_dir)
        self._load_prediction(prediction_model_dir)
        self._load_tkgl(tkgl_model_path)

    # ---------------- 加载 ----------------
    def _load_diagnosis(self, models_dir: str) -> None:
        try:
            discovered = _discover_models(models_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("诊断模型目录扫描失败: %s", e)
            discovered = []

        for mi in discovered:
            try:
                handler = DiagnosisModelHandler(
                    device=self.device,
                    top_k=self.default_top_k,
                    symptom_relation=self.symptom_relation,
                )
                model, meta = handler.load(mi["path"])
                # 缓存已加载的模型与图结构，避免每次请求重复加载
                self._diag_cache[mi["name"]] = (
                    model, meta, handler.data, handler.dataset,
                )
                logger.info("✓ 诊断模型已加载: %s", mi["name"])
            except Exception as e:  # noqa: BLE001
                logger.error("诊断模型 '%s' 加载失败: %s", mi["name"], e)

    def _load_prediction(self, model_dir: str) -> None:
        graph_path = os.path.join(model_dir, "temporal_kg_graph.pt")
        diag_path = os.path.join(model_dir, "temporal_diag_rgcn.pth")
        tgn_path = os.path.join(model_dir, "temporal_tgn.pth")

        if not (os.path.exists(graph_path)
                and os.path.exists(diag_path)
                and os.path.exists(tgn_path)):
            logger.warning(
                "趋势预测模型不完整，跳过加载（需: temporal_kg_graph.pt / "
                "temporal_diag_rgcn.pth / temporal_tgn.pth）。目录: %s",
                model_dir,
            )
            return

        try:
            kg_graph = torch.load(graph_path, map_location=self.device,
                                  weights_only=False)
            kg_graph = kg_graph.to(self.device)

            total = int(kg_graph["time_slice"].x.shape[0])
            diag_model = RGCNFaultDiagnosis(HIDDEN_DIM, HEALTH_NUM).to(self.device)
            tgn_model = build_tgn_oil_temperature_predict(
                HIDDEN_DIM, total).to(self.device)
            diag_model.load_state_dict(
                torch.load(diag_path, map_location=self.device, weights_only=True))
            tgn_model.load_state_dict(
                torch.load(tgn_path, map_location=self.device, weights_only=True))
            diag_model.eval()
            tgn_model.eval()

            self._pred = {
                "kg_graph": kg_graph,
                "diag_model": diag_model,
                "tgn_model": tgn_model,
                "total": total,
                "model_dir": model_dir,
            }
            logger.info("✓ 趋势预测模型已加载: 切片数=%d", total)
        except Exception as e:  # noqa: BLE001
            logger.error("趋势预测模型加载失败: %s", e)
            self._pred = None

    def _load_tkgl(self, model_path: "str | None") -> None:
        if not model_path:
            logger.info("未指定 --model-path，跳过 TKGL 链接预测模型加载")
            return
        if not os.path.exists(model_path):
            logger.warning(
                "TKGL 模型文件不存在，跳过加载: %s", model_path)
            return

        try:
            model, data = load_checkpoint(model_path, device=self.device)
            # 预热：构建关系端点候选集缓存，避免首个请求时额外开销
            try:
                _build_relation_endpoints(model)
            except Exception as e:  # noqa: BLE001
                logger.warning("TKGL 关系端点预构建失败（将按需构建）: %s", e)
            self._tkgl = {"model": model, "data": data}
            logger.info("✓ TKGL 链接预测模型已加载: %s", model_path)
        except Exception as e:  # noqa: BLE001
            logger.error("TKGL 模型加载失败: %s", e)
            self._tkgl = None

    # ---------------- 推理 ----------------
    def diagnose(
            self, symptoms: str,
            model_name: "str | None" = None,
            top_k: "int | None" = None
    ) -> dict:
        """执行故障诊断，返回结构化 JSON dict。"""
        if not self._diag_cache:
            raise RuntimeError("未加载任何诊断模型，请检查 --diagnosis-models-dir")

        name = model_name or next(iter(self._diag_cache))
        if name not in self._diag_cache:
            raise ValueError(
                f"未知诊断模型: {model_name}，可用: {list(self._diag_cache)}")

        model, meta, data, dataset = self._diag_cache[name]
        tk = int(top_k) if top_k else self.default_top_k

        # 按请求 top_k 构造轻量处理器，复用已缓存的模型与图结构
        handler = DiagnosisModelHandler(
            device=self.device, top_k=tk,
            symptom_relation=self.symptom_relation)
        handler.data = data
        handler.dataset = dataset

        processed = handler.preprocess(symptoms)
        with self._lock:
            raw_result = handler.infer((model, meta), processed)
        result_str = handler.postprocess(raw_result)
        return json.loads(result_str)

    def predict_trend(self, slice_idx: "int | None" = None) -> dict:
        """对指定时序切片执行健康诊断 + 油温趋势预测，返回结构化 dict。"""
        if self._pred is None:
            raise RuntimeError(
                "未加载趋势预测模型/图谱，请检查 --prediction-model-dir")

        kg = self._pred["kg_graph"]
        total = self._pred["total"]
        if slice_idx is None:
            slice_idx = total - 1
        slice_idx = int(slice_idx)
        if slice_idx < 0 or slice_idx >= total:
            raise ValueError(
                f"slice_idx 超出范围 [0, {total - 1}]: {slice_idx}")

        diag_model = self._pred["diag_model"]
        tgn_model = self._pred["tgn_model"]

        with torch.no_grad():
            # 1) R-GCN 故障诊断
            health_logits, _ = diag_model(kg.x_dict, kg.edge_index_dict)
            logits = health_logits[slice_idx]
            pred_health = int(torch.argmax(logits).item())
            probs = F.softmax(logits, dim=0).cpu().numpy()

            # 2) TGN 油温趋势预测（输出为 z-score，需反标准化成摄氏度）
            future_ot_pred, fault_risk_pred, _ = tgn_model(kg)
            ot_mean = kg["time_slice"].ot_mean.item()
            ot_std = kg["time_slice"].ot_std.item()
            pred_ot = float(future_ot_pred[slice_idx].item() * ot_std + ot_mean)
            risk = float(fault_risk_pred[slice_idx].item())

        # 切片原始特征
        slice_feat = kg["time_slice"].x_raw[slice_idx].cpu().numpy()
        features = {
            name: float(slice_feat[i]) for i, name in enumerate(FEATURE_LIST)
        }

        date_str = None
        if hasattr(kg["time_slice"], "date_str"):
            try:
                date_str = kg["time_slice"].date_str[slice_idx]
            except Exception:  # noqa: BLE001
                date_str = None

        health_probs = {
            HEALTH_MAPPING[i]: round(float(probs[i]), 4)
            for i in HEALTH_MAPPING
        }

        # 知识图谱行业规则匹配解释
        rules = []
        ot_val = slice_feat[6]
        hufl_val = slice_feat[0]
        if pred_health == 3 and hufl_val > 20:
            rules.append("规则1：主负载HUFL超标 → 图谱关联过载故障典型特征")
        if pred_health == 2 and ot_val >= 50:
            rules.append("规则2：油温OT≥50℃ → 图谱关联严重过热典型特征")
        if pred_health == 1 and 40 <= ot_val < 50:
            rules.append("规则3：油温40℃≤OT<50℃ → 图谱关联轻微过热典型特征")
        if not rules:
            rules.append("无异常特征，符合正常运行状态")

        risk_level = ("低风险" if risk < 0.3
                      else ("中风险" if risk < 0.7 else "高风险"))

        if pred_health == 0 and risk < 0.3:
            suggestion = "设备运行正常，维持常规巡检周期"
        elif pred_health == 1 or risk >= 0.3:
            suggestion = "设备轻微异常，缩短巡检周期，密切关注油温与负载变化"
        elif pred_health == 2 or risk >= 0.7:
            suggestion = "设备严重过热，立即安排停电检修，检查绝缘与散热系统"
        elif pred_health == 3:
            suggestion = "设备过载故障，立即降低负载，紧急停机检查"
        else:
            suggestion = "请结合规则与风险等级进一步研判"

        return {
            "slice_idx": slice_idx,
            "date": date_str,
            "features": features,
            "health_state": HEALTH_MAPPING[pred_health],
            "health_state_id": pred_health,
            "health_probabilities": health_probs,
            "rules": rules,
            "future_oil_temperature": round(pred_ot, 4),
            "fault_risk": round(risk, 4),
            "risk_level": risk_level,
            "suggestion": suggestion,
        }

    def predict_link(
            self,
            head=None,
            relation=None,
            time=None,
            queries=None,
            topk: int = 5,
            temporal_bias: float = 50.0,
            temporal_sigma: float = 8.0,
    ) -> dict:
        """TKGL 链接预测：给定 (头, 关系, 时间) 预测 Top-K 尾实体。

        支持两种调用方式：
          * 单次：传入 head / relation / time（标量）
          * 批量：传入 queries（元素为 {head, relation, time, ...} 的列表）
        批量模式返回 {"results": [...]}；单次模式返回单条结果 dict。
        """
        # 批量模式：queries 为查询列表
        if queries is not None:
            if not isinstance(queries, list) or not queries:
                raise ValueError("queries 必须是非空列表")

            results = []
            for i, q in enumerate(queries):
                try:
                    results.append(self._predict_link_one(
                        q.get("head"), q.get("relation"), q.get("time"),
                        topk=int(q.get("topk", topk)),
                        temporal_bias=float(q.get("temporal_bias", temporal_bias)),
                        temporal_sigma=float(q.get("temporal_sigma", temporal_sigma)),
                    ))
                except (ValueError, RuntimeError) as e:  # noqa: BLE001
                    results.append({"error": str(e), "index": i})
            return {"results": results}

        # 单次模式
        return self._predict_link_one(
            head, relation, time,
            topk=topk,
            temporal_bias=temporal_bias, temporal_sigma=temporal_sigma
        )

    def _predict_link_one(
            self,
            head,
            relation,
            time,
            topk: int = 5,
            temporal_bias: float = 50.0,
            temporal_sigma: float = 8.0,
    ) -> dict:
        """TKGL 链接预测单条推理。"""
        if self._tkgl is None:
            raise RuntimeError("未加载 TKGL 模型，请检查 --model-path")

        model = self._tkgl["model"]
        data = self._tkgl["data"]
        h = _resolve_id(str(head), data["entity2id"], data["id2entity"])
        r = _resolve_id(str(relation), data["relation2id"], data["id2relation"])
        if h is None:
            raise ValueError(f"未知头实体: {head}")
        if r is None:
            raise ValueError(f"未知关系: {relation}")

        try:
            ts = int(time)
        except (TypeError, ValueError):
            raise ValueError(f"时间必须是整数年份，收到: {time!r}")

        with self._lock:
            top = predict_tails(
                model, h, r, ts, true_t=None, k=topk,
                k_neg=max(2000, 2000), device=self.device,
                temporal_bias=temporal_bias, temporal_sigma=temporal_sigma)

        id2ent = data["id2entity"]
        return {
            "query": {
                "head": id2ent[h],
                "relation": data["id2relation"][r],
                "time": ts,
            },
            "predictions": [
                {"tail": id2ent[tid], "tail_id": int(tid), "score": float(sc)}
                for tid, sc in top
            ],
        }

    def evaluate_link(self, num_eval: int = 2000, k_neg: int = 500) -> dict:
        """TKGL 测试集过滤式 MRR / Hits@k 评测。"""
        if self._tkgl is None:
            raise RuntimeError("未加载 TKGL 模型，请检查 --model-path")

        model = self._tkgl["model"]
        data = self._tkgl["data"]
        mrr, h1, h3, h10 = evaluate_filtered(
            model, data["test_quads"], data["true_tails"],
            num_eval=num_eval, k_neg=k_neg, device=self.device)
        return {"mrr": mrr, "hits@1": h1, "hits@3": h3, "hits@10": h10}

    # ---------------- 健康检查 ----------------
    def health(self) -> dict:
        return {
            "status": "ok",
            "device": str(self.device),
            "diagnosis_models": list(self._diag_cache.keys()),
            "prediction_loaded": self._pred is not None,
            "tkgl_loaded": self._tkgl is not None,
        }


class _Handler(BaseHTTPRequestHandler):
    # 由 ThreadingHTTPServer 为每个连接新建实例；service 通过类属性共享
    service = None

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}

        raw = self.rfile.read(length)
        if not raw:
            return {}

        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):  # 静默默认访问日志，避免刷屏
        pass

    # ---------------- 各路由处理 ----------------
    def _do_diagnosis(self, payload: dict) -> None:
        """POST /diagnosis：故障诊断推理。"""
        symptoms = payload.get("symptoms", payload.get("instance"))
        if not symptoms or not str(symptoms).strip():
            self._send_json(
                {"error": "缺少必填字段: symptoms（或 instance）"},
                status=400)
            return
        top_k = payload.get("top_k")
        model = payload.get("model")
        result = _Handler.service.diagnose(
            str(symptoms), model_name=model, top_k=top_k)
        self._send_json(result)

    def _do_trend(self, payload: dict) -> None:
        """POST /trend：时序趋势预测推理。"""
        slice_idx = payload.get("slice_idx")
        result = _Handler.service.predict_trend(
            slice_idx=int(slice_idx) if slice_idx is not None else None)
        self._send_json(result)

    def _do_predict(self, payload: dict) -> None:
        """POST /predict：TKGL 链接预测（单次或批量）。"""
        queries = payload.get("queries")
        if queries is not None:
            # 批量推理：queries 为查询列表
            try:
                result = _Handler.service.predict_link(queries=queries)
            except (ValueError, RuntimeError) as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=400)
                return
            self._send_json(result)
            return

        # 单次推理
        head = payload.get("head")
        relation = payload.get("relation")
        time = payload.get("time")
        if head is None or relation is None or time is None:
            self._send_json(
                {"error": "缺少必填字段: head / relation / time（或 queries）"},
                status=400)
            return
        topk = int(payload.get("topk", 5))
        temporal_bias = float(payload.get("temporal_bias", 50.0))
        temporal_sigma = float(payload.get("temporal_sigma", 8.0))
        result = _Handler.service.predict_link(
            head, relation, time, topk=topk,
            temporal_bias=temporal_bias, temporal_sigma=temporal_sigma)
        self._send_json(result)

    def _do_evaluate(self, payload: dict) -> None:
        """POST /evaluate：TKGL 测试集过滤式 MRR / Hits@k 评测。"""
        num_eval = int(payload.get("num_eval", 2000))
        k_neg = int(payload.get("k_neg", 500))
        result = _Handler.service.evaluate_link(num_eval, k_neg)
        self._send_json(result)

    # ---------------- 路由 ----------------
    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/health/"):
            if _Handler.service is None:
                self._send_json({"error": "服务未初始化"}, status=503)
                return
            self._send_json(_Handler.service.health())
            return
        self._send_json({"error": f"未知路径: {self.path}"}, status=404)

    def do_POST(self):
        if _Handler.service is None:
            self._send_json({"error": "服务未初始化"}, status=503)
            return
        path = self.path.split("?")[0]
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError as e:
            self._send_json({"error": f"JSON 解析失败: {e}"}, status=400)
            return

        try:
            if path in ("/diagnosis", "/diagnosis/"):
                self._do_diagnosis(payload)
                return

            if path in ("/trend", "/trend/"):
                self._do_trend(payload)
                return

            if path in ("/predict", "/predict/"):
                self._do_predict(payload)
                return

            if path in ("/evaluate", "/evaluate/"):
                self._do_evaluate(payload)
                return

            self._send_json({"error": f"未知路径: {path}"}, status=404)
        except ValueError as e:  # 业务校验错误（缺模型/越界/未知模型）
            self._send_json({"error": str(e)}, status=400)
        except RuntimeError as e:  # 能力未加载
            self._send_json({"error": str(e)}, status=503)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"推理失败: {e}"}, status=500)


def run_server(
        diagnosis_models_dir: str,
        prediction_model_dir: str,
        tkgl_model_path: "str | None" = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "auto",
        top_k: int = 3,
        symptom_relation: "str | None" = None,
) -> None:
    """加载模型并启动多线程 HTTP 推理服务。"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    svc = InferenceService(
        diagnosis_models_dir=diagnosis_models_dir,
        prediction_model_dir=prediction_model_dir,
        tkgl_model_path=tkgl_model_path,
        device=device,
        top_k=top_k,
        symptom_relation=symptom_relation,
    )
    _Handler.service = svc

    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"🚀 统一推理服务已启动: http://{host}:{port}  (device={device})")
    print(f"   诊断模型: {svc.health()['diagnosis_models'] or '(无)'}")
    print(f"   趋势预测: {'已加载' if svc._pred else '未加载'}")
    print(f"   TKGL    : {'已加载' if svc._tkgl else '未加载'}")
    print(f"   接口: GET /health  POST /diagnosis  POST /trend  "
          f"POST /predict  POST /evaluate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 收到中断，关闭服务。")
    finally:
        server.server_close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一推理 —— HTTP 服务（故障诊断 + 时序趋势预测 + TKGL 链接预测）")
    parser.add_argument("--host", type=str,
                        default=_default_inference_config().host,
                        help="HTTP 服务绑定地址，默认来自 config.yaml")
    parser.add_argument("--port", type=int,
                        default=_default_inference_config().port,
                        help="HTTP 服务端口，默认来自 config.yaml")

    parser.add_argument(
        "--diagnosis-models-dir", type=str,
        default=_link_prediction_config().model_path,
        help="故障诊断模型目录（自动发现 .pt/.pth），默认来自 config.yaml"
    )
    parser.add_argument(
        "--prediction-model-dir", type=str,
        default=_trend_prediction_config().model_path,
        help="时序趋势预测模型/图谱目录（含 temporal_*.pt），默认来自 config.yaml"
    )
    parser.add_argument(
        "--model-path", type=str,
        default=_tkgl_prediction_config().model_path,
        help="TKGL 链接预测已训练模型 .pt 路径（可选）"
    )

    parser.add_argument(
        "--device", type=str, default=_default_inference_config().device,
        help="推理设备 auto/cpu/cuda，默认来自 config.yaml"
    )
    parser.add_argument(
        "--top-k", type=int, default=_default_inference_config().top_k,
        help="诊断默认返回 Top-K 故障数，默认来自 config.yaml"
    )
    parser.add_argument(
        "--symptom-relation", type=str, default=None,
        help="症状→故障语义关系名（默认从 config.yaml 读取）"
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    symptom_relation = (args.symptom_relation
                        if args.symptom_relation is not None
                        else _default_relation_mapping().get("symptoms"))
    run_server(
        diagnosis_models_dir=args.diagnosis_models_dir,
        prediction_model_dir=args.prediction_model_dir,
        tkgl_model_path=args.model_path,
        host=args.host,
        port=args.port,
        device=args.device,
        top_k=args.top_k,
        symptom_relation=symptom_relation,
    )


if __name__ == "__main__":
    main()

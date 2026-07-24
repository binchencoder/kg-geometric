# 统一推理 HTTP 服务（src/api）

对外提供静态链接预测、时序属性预测与 TLP 时序链接预测的 REST 接口。
基于 Python 标准库 `http.server` 实现，**零额外 Web 框架依赖**（无需 FastAPI / Flask）。

核心实现见 [`inference_server.py`](./inference_server.py)，集成自：

- `src/slp/predict.py` —— 静态链接预测（四阶段推理）
- `src/tap/predict.py` —— 时序属性预测（R-GCN + TGN）
- `src/tlp/predict.py` —— TLP 链接预测

---

## 快速开始

```bash
# 三类能力全部启用（缺省某项参数即不加载该项，其余照常工作）
python3 -m src.api.inference_server \
    --slp-model-dir ./models \
    --tap-model-dir ./trained_models/trend \
    --model-path trained_models/tlp/tkgl_smallpedia_model.pt \
    --host 0.0.0.0 --port 8000

# 仅启用静态链接预测 + 属性预测
python3 -m src.api.inference_server \
    --slp-model-dir ./models \
    --tap-model-dir ./trained_models/trend
```

启动后控制台会打印各能力加载状态与可用接口列表。

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slp-model-dir` | `config.yaml` 中 `inference.models_dir` | 静态链接预测模型目录，自动发现 `.pt` / `.pth` |
| `--tap-model-dir` | `./trained_models/trend` | 时序属性预测模型/图谱目录（含 `temporal_*.pt`） |
| `--model-path` | `None` | TLP 链接预测已训练模型 `.pt` 路径（可选） |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--device` | `auto` | 推理设备 `auto` / `cpu` / `cuda` |
| `--top-k` | `3` | 诊断默认返回 Top-K 故障数 |
| `--symptom-relation` | 读 `config.yaml` | 症状→故障语义关系名 |

> 三类能力相互独立：未提供对应参数则该能力不加载，其它能力不受影响。

---

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health`   | 健康检查，返回各能力加载状态 |
| POST | `/slp`| 静态链接预测（症状文本 → 结构化诊断 JSON） |
| POST | `/tap`    | 时序属性预测（健康状态 + 油温趋势 + 故障风险） |
| POST | `/predict`  | TLP 链接预测（头/关系/时间 → Top-K 尾实体） |
| POST | `/evaluate` | TLP 过滤式 MRR 评测 |

请求体均为 JSON，响应均为 UTF-8 JSON。

---

### 1. 健康检查 `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "device": "cpu",
  "slp_models": ["slp_model_xxx"],
  "tap_config_loaded": true,
  "tlp_config_loaded": true
}
```

---

### 2. 静态链接预测 `POST /slp`

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `symptoms` | string | 是* | 症状文本，多症状用换行分隔（也可用 `instance` 字段） |
| `top_k` | int | 否 | 返回 Top-K 故障数，默认服务端 `--top-k` |
| `model` | string | 否 | 指定诊断模型名，默认首个发现的模型 |

\* `symptoms` 与 `instance` 二选一。

```bash
curl -X POST http://localhost:8000/slp \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "振动过高\n温度过高", "top_k": 3}'
```

**响应**（结构化诊断 JSON，含 `causes` / `actions` / `tools` 等字段）

```json
{
  "diagnosis": "...",
  "top_fault": "...",
  "causes": ["..."],
  "actions": ["..."],
  "tools": ["..."]
}
```

---

### 3. 时序属性预测 `POST /tap`

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `slice_idx` | int | 否 | 时序切片索引，默认最新一个切片 |

```bash
curl -X POST http://localhost:8000/tap \
  -H "Content-Type: application/json" \
  -d '{"slice_idx": 100}'
```

```json
{
  "slice_idx": 100,
  "date": "2024-05-12",
  "features": {"HUFL": 12.3, "OT": 48.1, "...": "..."},
  "health_state": "正常运行",
  "health_state_id": 0,
  "health_probabilities": {"正常运行": 0.92, "...": "..."},
  "rules": ["无异常特征，符合正常运行状态"],
  "future_oil_temperature": 42.13,
  "fault_risk": 0.05,
  "risk_level": "低风险",
  "suggestion": "设备运行正常，维持常规巡检周期"
}
```

---

### 4. TLP 链接预测 `POST /predict`

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `head` | string \| int | 是 | 头实体：`Q-ID` / `P-ID` 字符串或整数 ID |
| `relation` | string \| int | 是 | 关系：`P-ID` 字符串或整数 ID |
| `time` | int | 是 | 查询年份（整数） |
| `topk` | int | 否 | 返回 Top-K 尾实体，默认 5 |
| `temporal_bias` | float | 否 | 时间邻近性偏置权重，默认 50.0 |
| `temporal_sigma` | float | 否 | 偏置高斯衰减尺度，默认 8.0 |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"head": "Q648", "relation": "P27", "time": 2008, "topk": 5}'
```

```json
{
  "query": {"head": "Q648", "relation": "P27", "time": 2008},
  "predictions": [
    {"tail": "Q42", "tail_id": 12, "score": 1.234},
    {"tail": "Q5",  "tail_id": 7,  "score": 0.987}
  ]
}
```

---

### 5. TLP 评测 `POST /evaluate`

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `num_eval` | int | 否 | 评测样本数，默认 2000 |
| `k_neg` | int | 否 | 负采样数，默认 500 |

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"num_eval": 2000, "k_neg": 500}'
```

```json
{"mrr": 0.31, "hits@1": 0.21, "hits@3": 0.36, "hits@10": 0.52}
```

---

## 模型文件要求

| 能力 | 所需文件 | 位置 |
|------|----------|------|
| 静态链接预测 | `*.pt` / `*.pth`（前缀 `slp_model`） | `--slp-model-dir` 目录下，按文件名前缀 `slp_model` 自动发现并加载缓存 |
| 时序属性预测 | `temporal_kg_graph.pt`、`temporal_diag_rgcn.pth`、`temporal_tgn.pth` | `--tap-model-dir` 目录 |
| TLP 链接预测 | 已训练 `.pt` checkpoint | `--model-path` 指定 |

---

## 错误码

| 状态码 | 含义 |
|--------|------|
| 400 | 请求参数错误（缺必填字段 / 未知实体或关系 / 切片越界 / 未知模型） |
| 503 | 对应能力未加载（服务未初始化或缺少模型文件） |
| 500 | 推理内部异常 |
| 404 | 未知路径 |

---

## 实现说明

- **线程安全**：模型参数只读，推理在 `torch.no_grad()` 下执行，并以 `threading.Lock` 保护并发前向传播；底层使用 `ThreadingHTTPServer` 处理并发连接。
- **按需加载**：能力仅在启动时按参数加载，加载失败仅告警/跳过，不阻断其它能力。
- **程序化调用**：除 HTTP 外，也可直接 `from src.api import InferenceService, run_server` 在进程内使用 `InferenceService.predict_slp / predict_tap / predict_link / evaluate_link`。

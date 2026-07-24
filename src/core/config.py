"""全局配置与基础数据结构。

包含：
- ESConfig: Elasticsearch 连接配置
- KnowledgeGraphSchema: 知识图谱索引/字段映射配置
- BatchProgress: 批量处理进度统计
- logger: 全局日志器

所有配置从 ``config/config.yaml`` 加载，避免把账号密码、索引名、字段名
硬编码在源代码中。YAML 文件缺失 / 解析失败 / 段缺失 时直接抛出明确错误，
防止在未知配置状态下继续运行。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ESKGReader")

# -------------------- 配置文件路径 --------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

# ES 连接配置字段白名单（同时做存在性检查与类型校正）
_ES_FIELDS_INT: tuple = ("port", "timeout", "max_retries")
_ES_FIELDS_BOOL: tuple = ("retry_on_timeout",)
_ES_FIELDS_STR: tuple = ("host", "username", "password", "scheme")
_ES_REQUIRED: frozenset = frozenset(
    _ES_FIELDS_INT + _ES_FIELDS_BOOL + _ES_FIELDS_STR
)

# 知识图谱配置字段白名单
_KG_FIELDS_INT: tuple = ("batch_size",)
_KG_FIELDS_STR: tuple = (
    "graph_id", "ontology_id",
    "entity_index", "relation_index", "relation_type_index",
    "entity_id_field", "entity_name_field",
    "head_id_field", "tail_id_field", "relation_field",
    "relation_type_id_field", "relation_type_name_field",
)
_KG_REQUIRED: frozenset = frozenset(_KG_FIELDS_INT + _KG_FIELDS_STR)


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """从 YAML 加载完整配置树；失败时抛出明确错误。

    Raises
    ------
    ImportError
        未安装 PyYAML
    FileNotFoundError
        配置文件不存在
    RuntimeError
        YAML 顶层不是 mapping 或解析失败
    """
    try:
        import yaml  # PyYAML
    except ImportError as e:
        raise ImportError(
            "未安装 PyYAML，无法加载配置文件。"
            "请先执行 `pip install PyYAML`。"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"配置文件不存在: {path}。"
            f"请确认 config/config.yaml 已正确放置。"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - yaml 底层异常类型不稳定
        raise RuntimeError(f"配置文件 {path} 解析失败: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(
            f"配置文件 {path} 顶层不是 mapping（YAML 字典），"
            f"请检查格式。"
        )
    return data


def _require_section(root: Dict[str, Any], section: str, path: str) -> Dict[str, Any]:
    """从配置树根中取出指定段；不存在或不是 mapping 时抛错。"""
    section_data = root.get(section, None)
    if not isinstance(section_data, dict):
        raise RuntimeError(
            f"配置文件 {path} 缺少 `{section}:` 段，或其值不是 mapping。"
            f"请在 YAML 中添加 `{section}:` 段并填写完整字段。"
        )
    return section_data


def _validate_and_coerce(
        data: Dict[str, Any],
        section: str,
        required: frozenset,
        int_fields: tuple,
        bool_fields: tuple,
        str_fields: tuple,
        path: str,
) -> Dict[str, Any]:
    """通用的字段存在性检查 + 类型校正。"""
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise RuntimeError(
            f"配置文件 {path} 的 `{section}:` 段缺少字段: {missing}。"
            f"必需字段: {sorted(required)}"
        )

    merged: Dict[str, Any] = {k: data[k] for k in required}
    for key in int_fields:
        merged[key] = int(merged[key])
    for key in bool_fields:
        merged[key] = bool(merged[key])
    for key in str_fields:
        merged[key] = str(merged[key])
    return merged


def _resolve_es_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """从 YAML 的 `elasticsearch` 段读取 ES 连接配置。

    与旧版本不同：**不再提供回退默认值**。YAML 中必须显式声明
    全部必需字段，否则直接抛错。
    """
    root = _load_yaml_config(path)
    section_data = _require_section(root, "elasticsearch", path)
    merged = _validate_and_coerce(
        section_data,
        section="elasticsearch",
        required=_ES_REQUIRED,
        int_fields=_ES_FIELDS_INT,
        bool_fields=_ES_FIELDS_BOOL,
        str_fields=_ES_FIELDS_STR,
        path=path,
    )
    logger.info(
        "已加载 ES 连接配置: %s:%s (%s)",
        merged["host"], merged["port"], merged["scheme"],
    )
    return merged


def _resolve_kg_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """从 YAML 的 `knowledge_graph` 段读取知识图谱 schema。

    与旧版本不同：**不再提供回退默认值**。YAML 中必须显式声明
    全部必需字段，否则直接抛错。
    """
    root = _load_yaml_config(path)
    section_data = _require_section(root, "knowledge_graph", path)
    merged = _validate_and_coerce(
        section_data,
        section="knowledge_graph",
        required=_KG_REQUIRED,
        int_fields=_KG_FIELDS_INT,
        bool_fields=(),
        str_fields=_KG_FIELDS_STR,
        path=path,
    )
    logger.info(
        "已加载知识图谱 schema | entity=%s | relation=%s | type=%s",
        merged["entity_index"],
        merged["relation_index"],
        merged["relation_type_index"],
    )
    return merged


# -------------------- 训练脚本配置 --------------------

_TRAINING_FIELDS_INT: tuple = ("seed",)
_TRAINING_FIELDS_STR: tuple = ("mode", "device")
_TRAINING_REQUIRED: frozenset = frozenset(_TRAINING_FIELDS_INT + _TRAINING_FIELDS_STR)

_DIAGNOSIS_FIELDS_INT: tuple = ("epochs", "hidden_dim", "num_layers")
_DIAGNOSIS_FIELDS_FLOAT: tuple = ("dropout", "lr", "weight_decay")
_DIAGNOSIS_FIELDS_OPT_STR: tuple = ("save_model_path",)
_DIAGNOSIS_REQUIRED: frozenset = frozenset(
    _DIAGNOSIS_FIELDS_INT + _DIAGNOSIS_FIELDS_FLOAT
)

_PREDICTION_FIELDS_INT: tuple = (
    "transformer_id", "hold_out", "epochs", "hidden_dim",
)
_PREDICTION_FIELDS_FLOAT: tuple = ("test_ratio", "lr")
_PREDICTION_FIELDS_STR: tuple = ("csv_path",)
_PREDICTION_FIELDS_OPT_STR: tuple = ("save_model_path",)
_PREDICTION_REQUIRED: frozenset = frozenset(
    _PREDICTION_FIELDS_INT + _PREDICTION_FIELDS_FLOAT + _PREDICTION_FIELDS_STR
)


def _validate_section_opt(
        data: Dict[str, Any],
        section: str,
        required: frozenset,
        int_fields: tuple,
        float_fields: tuple,
        str_fields: tuple,
        opt_str_fields: tuple,
        path: str,
) -> Dict[str, Any]:
    """带可选字段的字段存在性检查 + 类型校正。"""
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise RuntimeError(
            f"配置文件 {path} 的 `{section}:` 段缺少字段: {missing}。"
            f"必需字段: {sorted(required)}"
        )
    merged: Dict[str, Any] = {k: data[k] for k in required}
    for key in int_fields:
        merged[key] = int(merged[key])
    for key in float_fields:
        merged[key] = float(merged[key])
    for key in str_fields:
        merged[key] = str(merged[key])
    for key in opt_str_fields:
        v = data.get(key, None)
        merged[key] = str(v) if v is not None else None
    return merged


def _resolve_training_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    root = _load_yaml_config(path)
    section_data = _require_section(root, "training", path)
    merged = _validate_and_coerce(
        section_data,
        section="training",
        required=_TRAINING_REQUIRED,
        int_fields=_TRAINING_FIELDS_INT,
        bool_fields=(),
        str_fields=_TRAINING_FIELDS_STR,
        path=path,
    )
    logger.info("已加载 training 配置 | mode=%s | seed=%s | device=%s",
                merged["mode"], merged["seed"], merged["device"])
    return merged


def _resolve_slp_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    root = _load_yaml_config(path)
    section_data = _require_section(root, "slp_prediction", path)
    merged = _validate_section_opt(
        section_data,
        section="slp_prediction",
        required=_DIAGNOSIS_REQUIRED,
        int_fields=_DIAGNOSIS_FIELDS_INT,
        float_fields=_DIAGNOSIS_FIELDS_FLOAT,
        str_fields=(),
        opt_str_fields=_DIAGNOSIS_FIELDS_OPT_STR,
        path=path,
    )
    logger.info("已加载 diagnosis 配置 | epochs=%s | hidden_dim=%s | lr=%s",
                merged["epochs"], merged["hidden_dim"], merged["lr"])
    return merged


def _resolve_tap_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    root = _load_yaml_config(path)
    section_data = _require_section(root, "tap_prediction", path)
    merged = _validate_section_opt(
        section_data,
        section="tap_prediction",
        required=_PREDICTION_REQUIRED,
        int_fields=_PREDICTION_FIELDS_INT,
        float_fields=_PREDICTION_FIELDS_FLOAT,
        str_fields=_PREDICTION_FIELDS_STR,
        opt_str_fields=_PREDICTION_FIELDS_OPT_STR,
        path=path,
    )
    logger.info("已加载 prediction 配置 | csv_path=%s | epochs=%s | lr=%s",
                merged["csv_path"], merged["epochs"], merged["lr"])
    return merged


@dataclass(frozen=True)
class TrainingConfig:
    """训练脚本顶层配置（从 ``config/config.yaml`` 加载）。"""
    mode: str
    seed: int
    device: str

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        merged = _resolve_training_config(path)
        return cls(mode=merged["mode"], seed=int(merged["seed"]),
                   device=merged["device"])


@dataclass(frozen=True)
class SLPTrainingConfig:
    """静态链接预测训练配置（从 ``config/config.yaml`` 加载）。"""
    epochs: int
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    save_model_path: Optional[str]

    @classmethod
    def default(cls) -> "SLPTrainingConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "SLPTrainingConfig":
        m = _resolve_slp_config(path)
        return cls(
            epochs=int(m["epochs"]),
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
            lr=float(m["lr"]),
            weight_decay=float(m["weight_decay"]),
            save_model_path=m.get("save_model_path"),
        )


@dataclass(frozen=True)
class TAPTrainingConfig:
    """时序属性预测训练配置（从 ``config/config.yaml`` 加载）。"""
    csv_path: str
    transformer_id: int
    hold_out: int
    test_ratio: float
    epochs: int
    lr: float
    hidden_dim: int
    save_model_path: Optional[str]

    @classmethod
    def default(cls) -> "TAPTrainingConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "TAPTrainingConfig":
        m = _resolve_tap_config(path)
        return cls(
            csv_path=m["csv_path"],
            transformer_id=int(m["transformer_id"]),
            hold_out=int(m["hold_out"]),
            test_ratio=float(m["test_ratio"]),
            epochs=int(m["epochs"]),
            lr=float(m["lr"]),
            hidden_dim=int(m["hidden_dim"]),
            save_model_path=m.get("save_model_path"),
        )


def load_training_configs(
        path: str = DEFAULT_CONFIG_PATH,
) -> tuple[TrainingConfig, SLPTrainingConfig, TAPTrainingConfig]:
    """一次性加载全部训练配置。"""
    return (
        TrainingConfig.from_yaml(path),
        SLPTrainingConfig.from_yaml(path),
        TAPTrainingConfig.from_yaml(path),
    )


def _resolve_health_mapping_config(
        path: str = DEFAULT_CONFIG_PATH,
) -> Dict[int, str]:
    """从 YAML 的 `health_mapping` 段读取健康状态标签映射。

    YAML 格式为有序列表，列表下标即为标签 ID。例如::

        health_mapping:
          - "正常运行"
          - "轻微过热"
          - "严重过热"
          - "过载故障"

    空字符串/全空白/None 条目会被跳过并记录警告。
    """
    root = _load_yaml_config(path)
    raw = root.get("health_mapping", None)
    if raw is None:
        raise RuntimeError(
            f"配置文件 {path} 缺少 `health_mapping:` 段，"
            f"请在 YAML 中添加列表形式的健康状态描述。"
        )
    if not isinstance(raw, list):
        raise RuntimeError(
            f"配置文件 {path} 的 `health_mapping:` 段必须是列表形式 "
            f"(当前类型: {type(raw).__name__})"
        )

    mapping: Dict[int, str] = {}
    for idx, val in enumerate(raw):
        if val is None:
            logger.warning("health_mapping 第 %d 项为 None，跳过该条目", idx)
            continue
        if not isinstance(val, str) or not val.strip():
            logger.warning(
                "health_mapping 第 %d 项 (%r) 不是有效字符串，跳过",
                idx, val,
            )
            continue
        mapping[idx] = val.strip()

    if not mapping:
        raise RuntimeError(
            f"config 文件 {path} 的 `health_mapping:` 列表为空或全部条目无效"
        )
    logger.info(
        "已加载 health_mapping | 共 %d 个标签: %s",
        len(mapping), mapping,
    )
    return mapping


@dataclass(frozen=True)
class HealthMappingConfig:
    """健康状态标签映射配置（从 ``config/config.yaml`` 加载）。

    字段:
        mapping: Dict[int, str] —— 标签 ID -> 中文描述
        health_num: int —— 标签总数（自动计算）
    """
    mapping: Dict[int, str]
    health_num: int

    @classmethod
    def default(cls) -> "HealthMappingConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "HealthMappingConfig":
        m = _resolve_health_mapping_config(path)
        return cls(mapping=m, health_num=len(m))


def load_health_mapping(
        path: str = DEFAULT_CONFIG_PATH,
) -> Dict[int, str]:
    """便捷函数：直接返回 {标签 id: 中文描述} 的映射。"""
    return HealthMappingConfig.from_yaml(path).mapping


def _resolve_inference_config(
        path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """从 YAML 的 `inference:` 段读取推理引擎参数（predict.py）。

    当前结构（支持两路子配置：slp_prediction 与 tap_prediction）::

        inference:
          device: cpu
          slp_prediction:
            model_path: /path/to/kg_fault_model.pth
            instance: "振动过高\\n温度过高"
            top_k: 3
          tap_prediction:
            model_path: /path/to/kg_trend_model.pth
    """
    root = _load_yaml_config(path)
    section_data = _require_section(root, "inference", path)

    host = str(section_data.get("host", "0.0.0.0"))
    port = int(section_data.get("port", 8000))
    if not isinstance(host, str) or not host.strip():
        host = "0.0.0.0"

    device = section_data.get("device", "cpu")
    if not isinstance(device, str) or not device.strip():
        device = "cpu"

    slp = section_data.get("slp_prediction") or {}
    if not isinstance(slp, dict):
        slp = {}

    model_path = slp.get("model_path", "")
    instance = slp.get("instance", "")
    top_k = int(slp.get("top_k", 3))

    if not isinstance(model_path, str):
        model_path = str(model_path) if model_path is not None else ""
    if not isinstance(instance, str):
        instance = str(instance) if instance is not None else ""

    # 兼容字段：models_dir 由 slp.model_path 的父目录推导
    if model_path.strip():
        models_dir = str(Path(model_path).parent)
    else:
        models_dir = "./models"
    if not models_dir.strip():
        models_dir = "./models"

    tap = section_data.get("tap_prediction") or {}
    if not isinstance(tap, dict):
        tap = {}
    tap_model_path = tap.get("model_path", "")
    if not isinstance(tap_model_path, str):
        tap_model_path = str(tap_model_path) if tap_model_path is not None else ""

    tlp = section_data.get("tlp_prediction") or {}
    if not isinstance(tlp, dict):
        tlp = {}
    tlp_model_path = tlp.get("model_path", "")
    if not isinstance(tlp_model_path, str):
        tlp_model_path = str(tlp_model_path) if tlp_model_path is not None else ""

    merged: Dict[str, Any] = {
        "host": host.strip(),
        "port": port,
        "device": device.strip(),
        "models_dir": models_dir.strip(),
        "slp_config": {
            "model_path": model_path.strip(),
            "instance": instance,
            "top_k": top_k,
        },
        "tap_config": {
            "model_path": tap_model_path.strip(),
        },
        "tlp": {
            "model_path": tlp_model_path.strip(),
        },
    }
    logger.info(
        "已加载 inference 配置 | device=%s | models_dir=%s | slp_config.top_k=%d",
        merged["device"], merged["models_dir"],
        merged["slp_config"]["top_k"],
    )
    return merged


@dataclass(frozen=True)
class SLPConfig:
    """SLP（静态链接预测）推理子配置。"""
    model_path: str
    instance: str
    top_k: int


@dataclass(frozen=True)
class TAPConfig:
    """TAP（时序属性预测）推理子配置。"""
    model_path: str


@dataclass(frozen=True)
class TLPConfig:
    """TLP（时序链接预测）推理子配置。"""
    model_path: str


@dataclass(frozen=True)
class InferenceConfig:
    """推理引擎配置（predict.py 的默认参数，从 ``config/config.yaml`` 加载）。"""
    host: str
    port: int
    device: str

    # 静态链接预测推理配置
    slp_config: SLPConfig
    # 动态链接预测推理配置
    tlp_config: TLPConfig
    # 时序属性预测推理配置
    tap_config: TAPConfig

    @property
    def instance(self) -> str:
        """兼容字段：返回 slp_config.instance。"""
        return self.slp_config.instance

    @property
    def top_k(self) -> int:
        """兼容字段：返回 slp_config.top_k。"""
        return self.slp_config.top_k

    @classmethod
    def default(cls) -> "InferenceConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        m = _resolve_inference_config(path)
        return cls(
            host=str(m["host"]),
            port=int(m["port"]),
            device=str(m["device"]),
            slp_config=SLPConfig(
                model_path=str(m["slp_config"]["model_path"]),
                instance=str(m["slp_config"]["instance"]),
                top_k=int(m["slp_config"]["top_k"]),
            ),
            tlp_config=TLPConfig(
                model_path=str(m["tlp"]["model_path"]),
            ),
            tap_config=TAPConfig(
                model_path=str(m["tap_config"]["model_path"]),
            ),
        )


def load_inference_config(
        path: str = DEFAULT_CONFIG_PATH,
) -> InferenceConfig:
    """便捷函数：加载 InferenceConfig。"""
    return InferenceConfig.from_yaml(path)


# -------------------- ES 连接配置 --------------------
@dataclass(frozen=True)
class ESConfig:
    """Elasticsearch 连接配置（从 ``config/config.yaml`` 加载）。

    用法::

        cfg = ESConfig.default()                      # 从 config/config.yaml 加载
        cfg = ESConfig.from_yaml("path/to/custom.yaml")  # 指定自定义 YAML
        cfg = ESConfig(host=..., port=..., ...)       # 显式传参构造

    字段含义与 Elasticsearch 客户端一致：
    - host/port/scheme: 连接端点
    - username/password: HTTP Basic 认证凭据
    - timeout: 单请求超时（秒）
    - max_retries / retry_on_timeout: 失败重试策略

    YAML 格式示例（``config/config.yaml``）::

        elasticsearch:
          host: "10.1.13.30"
          port: 30920
          username: "elastic"
          password: "E1OfAx4Nf55513tU4i40eQbA"
          scheme: "http"
          timeout: 60
          max_retries: 3
          retry_on_timeout: true
    """

    host: str
    port: int
    username: str
    password: str
    scheme: str
    timeout: int
    max_retries: int
    retry_on_timeout: bool

    @classmethod
    def default(cls) -> "ESConfig":
        """从默认路径 ``config/config.yaml`` 加载 ES 配置。"""
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "ESConfig":
        """从指定 YAML 路径构造 ESConfig。"""
        merged = _resolve_es_config(path)
        return cls(
            host=merged["host"],
            port=merged["port"],
            username=merged["username"],
            password=merged["password"],
            scheme=merged["scheme"],
            timeout=merged["timeout"],
            max_retries=merged["max_retries"],
            retry_on_timeout=merged["retry_on_timeout"],
        )


# 为了兼容原先随处可见的 `ESConfig()`（零参数构造）用法，
# 提供一个模块级便捷工厂。`ESConfig.default()` 是更显式的推荐写法。
def load_es_config(path: str = DEFAULT_CONFIG_PATH) -> ESConfig:
    """从 YAML 加载 ESConfig；与 ``ESConfig.from_yaml(path)`` 等价。"""
    return ESConfig.from_yaml(path)


# -------------------- 知识图谱索引/字段映射 --------------------
@dataclass(frozen=True)
class KnowledgeGraphSchema:
    """知识图谱索引/字段映射（从 ``config/config.yaml`` 加载）。

    用法::

        schema = KnowledgeGraphSchema.default()               # 从 config/config.yaml 加载
        schema = KnowledgeGraphSchema.from_yaml("custom.yaml") # 自定义路径
        schema = KnowledgeGraphSchema(graph_id=..., ...)      # 显式传参

    包含字段：
    - graph_id / ontology_id: 图谱标识
    - batch_size: scan 每批文档数
    - entity_index / relation_index / relation_type_index: 三个 ES 索引名
    - entity_id_field / entity_name_field: 实体索引的 ID/名称字段
    - head_id_field / tail_id_field / relation_field: 关系索引的头/尾/类型字段
    - relation_type_id_field / relation_type_name_field: 关系类型索引的 ID/名称字段
    """

    graph_id: str
    ontology_id: str
    batch_size: int
    entity_index: str
    relation_index: str
    relation_type_index: str
    entity_id_field: str
    entity_name_field: str
    head_id_field: str
    tail_id_field: str
    relation_field: str
    relation_type_id_field: str
    relation_type_name_field: str

    @classmethod
    def default(cls) -> "KnowledgeGraphSchema":
        """从默认路径 ``config/config.yaml`` 加载知识图谱 schema。"""
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "KnowledgeGraphSchema":
        """从指定 YAML 路径构造 KnowledgeGraphSchema。"""
        merged = _resolve_kg_config(path)
        return cls(
            graph_id=merged["graph_id"],
            ontology_id=merged["ontology_id"],
            batch_size=merged["batch_size"],
            entity_index=merged["entity_index"],
            relation_index=merged["relation_index"],
            relation_type_index=merged["relation_type_index"],
            entity_id_field=merged["entity_id_field"],
            entity_name_field=merged["entity_name_field"],
            head_id_field=merged["head_id_field"],
            tail_id_field=merged["tail_id_field"],
            relation_field=merged["relation_field"],
            relation_type_id_field=merged["relation_type_id_field"],
            relation_type_name_field=merged["relation_type_name_field"],
        )

    def override(
            self,
            graph_id: Optional[str] = None,
            ontology_id: Optional[str] = None,
            batch_size: Optional[int] = None,
            entity_index: Optional[str] = None,
            relation_index: Optional[str] = None,
            relation_type_index: Optional[str] = None,
            entity_id_field: Optional[str] = None,
            entity_name_field: Optional[str] = None,
            head_id_field: Optional[str] = None,
            tail_id_field: Optional[str] = None,
            relation_field: Optional[str] = None,
            relation_type_id_field: Optional[str] = None,
            relation_type_name_field: Optional[str] = None,
    ) -> "KnowledgeGraphSchema":
        """返回一个新 schema，部分字段被覆盖；为 None 的字段保持原值。"""
        return KnowledgeGraphSchema(
            graph_id=self.graph_id if graph_id is None else graph_id,
            ontology_id=self.ontology_id if ontology_id is None else ontology_id,
            batch_size=self.batch_size if batch_size is None else batch_size,
            entity_index=self.entity_index if entity_index is None else entity_index,
            relation_index=self.relation_index if relation_index is None else relation_index,
            relation_type_index=(
                self.relation_type_index if relation_type_index is None
                else relation_type_index
            ),
            entity_id_field=(
                self.entity_id_field if entity_id_field is None else entity_id_field
            ),
            entity_name_field=(
                self.entity_name_field if entity_name_field is None else entity_name_field
            ),
            head_id_field=self.head_id_field if head_id_field is None else head_id_field,
            tail_id_field=self.tail_id_field if tail_id_field is None else tail_id_field,
            relation_field=(
                self.relation_field if relation_field is None else relation_field
            ),
            relation_type_id_field=(
                self.relation_type_id_field if relation_type_id_field is None
                else relation_type_id_field
            ),
            relation_type_name_field=(
                self.relation_type_name_field if relation_type_name_field is None
                else relation_type_name_field
            ),
        )


def load_kg_config(path: str = DEFAULT_CONFIG_PATH) -> KnowledgeGraphSchema:
    """从 YAML 加载 KnowledgeGraphSchema；与 ``KnowledgeGraphSchema.from_yaml(path)`` 等价。"""
    return KnowledgeGraphSchema.from_yaml(path)


# -------------------- 静态链接预测关系映射 --------------------
def _resolve_relation_mapping_config(
        path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, str]:
    """从 YAML 的 `relation_mapping:` 段读取语义角色 → 关系名映射。

    该映射用于 ``KGTripleDataset.get_node_relations`` 等场景，将
    causes/actions/tools 等语义角色映射到图谱中实际的关系名称。

    YAML 格式示例::

        relation_mapping:
          symptoms: "表现为"
          causes: "由...引起"
          actions: "维修措施"
          tools: "需要工具"
          system: "属于系统"
          category: "属于类别"

    段缺失时回退到与历史硬编码一致的内置默认映射（保证向后兼容）；
    段存在但为空/全部无效时直接抛错。
    """
    root = _load_yaml_config(path)
    raw = root.get("relation_mapping", None)
    if raw is None:
        logger.warning(
            "配置文件 %s 缺少 `relation_mapping:` 段，使用内置默认关系映射",
            path,
        )
        return {
            "symptoms": "表现为",
            "causes": "由...引起",
            "actions": "维修措施",
            "tools": "需要工具",
            "system": "属于系统",
            "category": "属于类别",
        }
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"配置文件 {path} 的 `relation_mapping:` 段必须是 mapping "
            f"(当前类型: {type(raw).__name__})"
        )

    mapping: Dict[str, str] = {}
    for key, val in raw.items():
        if val is None or not isinstance(val, str) or not val.strip():
            logger.warning(
                "relation_mapping 的 %r 项无效 (%r)，跳过", key, val,
            )
            continue
        mapping[str(key)] = val.strip()

    if not mapping:
        raise RuntimeError(
            f"config 文件 {path} 的 `relation_mapping:` 段为空或全部条目无效"
        )
    logger.info(
        "已加载 relation_mapping | 共 %d 条: %s", len(mapping), mapping,
    )
    return mapping


@dataclass(frozen=True)
class RelationMappingConfig:
    """静态链接预测关系映射配置（从 ``config/config.yaml`` 加载）。

    字段:
        mapping: Dict[str, str] —— 语义角色 -> 图谱关系名
    """
    mapping: Dict[str, str]

    @classmethod
    def default(cls) -> "RelationMappingConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    @classmethod
    def from_yaml(cls, path: str) -> "RelationMappingConfig":
        return cls(mapping=_resolve_relation_mapping_config(path))


def load_relation_mapping(
        path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, str]:
    """便捷函数：直接返回 {语义角色: 关系名} 的映射。"""
    return RelationMappingConfig.from_yaml(path).mapping


# -------------------- 批量进度回调 --------------------
@dataclass
class BatchProgress:
    """批量处理进度统计。"""
    total_docs: int = 0
    valid_triples: int = 0
    skipped_docs: int = 0
    errors: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        if self.total_docs == 0:
            return 0.0
        return self.valid_triples / self.total_docs

    def summary(self) -> str:
        return (
            f"总文档: {self.total_docs} | 有效三元组: {self.valid_triples} | "
            f"跳过: {self.skipped_docs} | 错误: {self.errors} | "
            f"有效率: {self.valid_ratio:.2%}"
        )

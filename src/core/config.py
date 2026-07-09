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
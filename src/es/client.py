"""ES 底层工具函数集。

提供 ES 客户端创建和基础操作（索引检测、扫描、安全取值），
供 ESKnowledgeGraphReader 和 IDNameResolver 等模块复用。
"""

from __future__ import annotations

from typing import List, Optional

from elasticsearch import Elasticsearch

from ..core.config import ESConfig, logger


def create_es_client(config: Optional[ESConfig] = None) -> Elasticsearch:
    """根据配置创建 ES 客户端。

    Parameters
    ----------
    config : Optional[ESConfig]
        ES 连接配置，为 None 时使用默认配置。

    Returns
    -------
    Elasticsearch
    """
    config = config or ESConfig()
    return Elasticsearch(
        hosts=[{
            "host": config.host,
            "port": config.port,
            "scheme": config.scheme,
        }],
        http_auth=(config.username, config.password),
        request_timeout=config.timeout,
        max_retries=config.max_retries,
        retry_on_timeout=config.retry_on_timeout,
        sniff_on_start=False,
        sniff_on_connection_fail=False,
        verify_certs=False,
    )


def ping_es(client: Elasticsearch, host: str, port: int) -> None:
    """检测 ES 连通性，失败时警告但不阻断。"""
    try:
        if client.ping():
            logger.info("Elasticsearch 连接成功: %s:%s", host, port)
        else:
            logger.warning(
                "Elasticsearch ping 返回 False（%s:%s），可能是代理/网关拦截，"
                "后续索引操作仍会重试", host, port,
            )
    except Exception as e:
        logger.warning(
            "Elasticsearch ping 异常（%s:%s）: %s。"
            "将跳过连通性检测，实际读写时如失败会抛出异常",
            host, port, e,
        )


def index_exists(client: Elasticsearch, index: str) -> bool:
    """检测索引是否存在。"""
    try:
        return client.indices.exists(index=index)
    except Exception:
        return False


def list_indices(client: Elasticsearch, pattern: str = "*") -> List[str]:
    """列出匹配模式的所有索引。"""
    try:
        cat = client.cat.indices(index=pattern, format="json")
        return [item["index"] for item in cat]
    except Exception as e:
        logger.error("列出索引失败: %s", e)
        return []


def get_index_mapping(client: Elasticsearch, index_name: str) -> dict:
    """获取索引的 mapping 信息。"""
    try:
        resp = client.indices.get_mapping(index=index_name)
        return resp.body if hasattr(resp, "body") else resp
    except Exception as e:
        logger.error("获取索引 %s mapping 失败: %s", index_name, e)
        return {}


def safe_get(source: dict, field: str, default: str = "") -> str:
    """安全地从文档中取值，支持嵌套字段（如 obj.nested.key）。"""
    if field in source:
        val = source[field]
        return str(val).strip() if val is not None else default
    # 尝试嵌套路径
    parts = field.split(".")
    current = source
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return str(current).strip() if current is not None else default

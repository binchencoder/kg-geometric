"""ES 层：Elasticsearch 连接、读取、流式提取、ID 解析与词汇表。

包含：
- create_es_client / ping_es / safe_get 等底层工具函数
- ESKnowledgeGraphReader: 传统 scroll/scan 模式读取器
- IDNameResolver: ID→名称双向解析器
- ESTripletStreamer: search_after 流式三元组提取器
- KGVocabulary: 全局实体/关系词汇表管理器
"""

from .client import create_es_client, ping_es, index_exists, list_indices, get_index_mapping, safe_get
from .reader import ESKnowledgeGraphReader
from .resolver import IDNameResolver
from .streamer import ESTripletStreamer
from .vocabulary import KGVocabulary

__all__ = [
    "create_es_client", "ping_es", "index_exists", "list_indices",
    "get_index_mapping", "safe_get",
    "ESKnowledgeGraphReader",
    "IDNameResolver",
    "ESTripletStreamer",
    "KGVocabulary",
]

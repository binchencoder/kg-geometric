"""全局配置与基础数据结构。

包含：
- ESConfig: Elasticsearch 连接配置
- BatchProgress: 批量处理进度统计
- logger: 全局日志器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ESKGReader")


# -------------------- ES 连接配置 --------------------
@dataclass(frozen=True)
class ESConfig:
    """Elasticsearch 连接配置。"""
    host: str = "10.1.13.30"
    port: int = 30920
    username: str = "elastic"
    password: str = "E1OfAx4Nf55513tU4i40eQbA"
    scheme: str = "http"
    timeout: int = 60
    max_retries: int = 3
    retry_on_timeout: bool = True


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

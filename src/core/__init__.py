"""核心模块：全局配置与基础数据类型。

包含：
- ESConfig: Elasticsearch 连接配置
- Triple: 知识图谱三元组数据类
- BatchProgress: 批量处理进度统计
- logger: 全局日志器
"""

from .config import ESConfig, BatchProgress, logger
from .types import Triple

__all__ = ["ESConfig", "Triple", "BatchProgress", "logger"]

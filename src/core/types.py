"""核心数据类型定义。

包含：
- Triple: 知识图谱三元组数据类
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    """知识图谱三元组：头实体 --关系--> 尾实体。"""
    head: str
    relation: str
    tail: str

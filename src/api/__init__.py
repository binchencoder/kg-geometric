# -*- coding: utf-8 -*-
"""
src/api/__init__.py
=====================================================================
统一推理 HTTP 服务包。

对外提供故障诊断、时序趋势预测与 TKGL 链接预测的 HTTP API（基于
Python 标准库 http.server，零额外依赖）。核心实现见 inference_server.py：

  - POST /diagnosis  故障诊断（症状文本 → 结构化诊断 JSON）
  - POST /trend      时序趋势预测（健康状态 + 油温趋势 + 故障风险）
  - POST /predict    TKGL 链接预测（头/关系/时间 → Top-K 尾实体）
  - POST /evaluate   TKGL 过滤式 MRR 评测
  - GET  /health     健康检查
"""

from .inference_server import run_server, InferenceService  # noqa: F401

__all__ = ["run_server", "InferenceService"]

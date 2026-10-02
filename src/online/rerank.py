# -*- coding: utf-8 -*-
"""W5 - rerank.py:重排(进阶,当前架构唯一缺失的精排环节)

两方案(架构 §6-R4):
    方案 B(先试):LLM-as-reranker —— 把 top-20 交给 qwen-plus 排序
    方案 A(备选):torch + bge-reranker-base(CPU)—— cross-encoder 逐对精算
    注意:Ollama 不支持 cross-encoder,此模块无法走 Ollama。

对应:路线计划 W5 | 架构 §4 步骤③
"""
from common import config


def rerank(query: str, hits: list[dict], top_m: int = config.TOP_M) -> list[dict]:
    """从召回的 hits 中精排出 top_m。TODO: W5 实现(先用方案 B)"""
    raise NotImplementedError

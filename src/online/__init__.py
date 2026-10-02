# -*- coding: utf-8 -*-
"""online —— 在线流水线包(查询):问题 → 向量 → 检索 → (重排) → 生成

模块调用顺序(路线计划 D6):
    retrieve.py → (rerank.py, W5 进阶) → qwen.py
由 src/pipeline.py 统一编排。
"""

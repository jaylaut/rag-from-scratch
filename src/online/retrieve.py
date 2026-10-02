# -*- coding: utf-8 -*-
"""D6 - retrieve.py:向量检索(在线流水线第①②步)

职责:
    1. 加载 index/wiki.faiss(进程内只加载一次)
    2. 查询向量 → IndexFlatIP 检索 top-k(RECALL_K=50,宽口径)
    3. 按 faiss_row 回填元数据(chunk_id/title/text),过滤 -1

对应:路线计划 D6 | 架构 §4
"""
from typing import Union

import numpy as np

from common import config


def load_index():
    """加载 FAISS 索引并缓存(模块级单例)。TODO: D6 实现"""
    raise NotImplementedError


def search(query_vec: np.ndarray, k: int = config.RECALL_K) -> list[dict]:
    """top-k 检索 + 元数据回填 → [{score, chunk_id, title, text}, ...]。TODO: D6 实现"""
    raise NotImplementedError

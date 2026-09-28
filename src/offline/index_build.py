# -*- coding: utf-8 -*-
"""D4 - index_build.py:建 FAISS 索引(离线流水线终点)

职责:
    1. 归一化向量 → faiss.IndexFlatIP(1024) → write_index 到 index/wiki.faiss
    2. 元数据同步落盘 index/chunks_meta.jsonl(第 i 行 ↔ FAISS 第 i 行,铁律)
    3. sanity_check:拿库内已有 chunk 重新 embed 检索,top1 必须是自己(>0.999)

对应:路线计划 D4 | 架构 §3.5
运行:python -m offline.index_build
"""
from pathlib import Path

import faiss

from common import config


def build_index(vectors: "np.ndarray", meta_source: Path) -> None:
    """建 IndexFlatIP 并落盘索引 + 元数据。TODO: D4 实现"""
    raise NotImplementedError


def sanity_check(k: int = 5) -> None:
    """随机抽 3 个已有 chunk 验证 top1 命中自身。TODO: D4 实现"""
    raise NotImplementedError

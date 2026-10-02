# -*- coding: utf-8 -*-
"""offline —— 离线流水线包(建库):dump → 文章 → 块 → 向量 → FAISS 索引

模块执行顺序(路线计划 D0→D4):
    download.py → parse.py → chunk.py → embed.py → index_build.py
产物:data/{raw,cleaned,chunks}/ + index/
"""

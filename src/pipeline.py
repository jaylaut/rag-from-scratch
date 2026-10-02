# -*- coding: utf-8 -*-
"""D5/D6 - pipeline.py:端到端编排(系统的门面)

调用链:
    ask(question)
      ├─ offline.embed.embed_query(question)     # ① 问题向量化
      ├─ online.retrieve.search(q_vec, k=50)     # ② 宽口径召回
      ├─ online.rerank.rerank(...)               # ③ (W5 接入,当前直通)
      ├─ online.qwen.build_prompt + generate     # ④ 窄口径生成
      └─ 返回 {answer, sources: [{title, chunk_id, score}]}

对应:路线计划 D5(子集极简版)→ D6(完整版)
用法(venv 激活后,任意目录):
    python -c "from pipeline import ask; print(ask('京剧的起源是什么?'))"
"""
from common import config


def ask(question: str, k: int = config.RECALL_K,
        top_m: int = config.TOP_M, stream: bool = True) -> dict:
    """端到端问答:问题 → 带引用的答案。TODO: D5 极简版 → D6 完整版"""
    raise NotImplementedError

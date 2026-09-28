# -*- coding: utf-8 -*-
"""D6 - qwen.py:prompt 组装 + Qwen 生成(在线流水线最后一步)

职责:
    1. build_prompt:系统指令 + [来源N] 正文(单来源截断 800 字)+ 问题
    2. count_tokens:tiktoken(cl100k)估算;超预算削减顺序:单来源截断 → 来源个数
    3. generate:调 dashscope Generation(qwen-plus),支持流式

对应:路线计划 D6 | 架构 §4 步骤④
"""
from common import config


def build_prompt(question: str, hits: list[dict],
                 max_src: int = config.TOP_M, src_max_chars: int = 800) -> str:
    """组装带 [来源N] 标注的完整 prompt。TODO: D6 实现"""
    raise NotImplementedError


def count_tokens(text: str) -> int:
    """tiktoken 估算 token 数(中文为近似值)。TODO: D6 实现"""
    raise NotImplementedError


def generate(prompt: str, stream: bool = True):
    """调用 qwen-plus;stream=True 返回逐段迭代器。TODO: D6 实现"""
    raise NotImplementedError

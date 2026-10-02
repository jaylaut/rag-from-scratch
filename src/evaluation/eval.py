# -*- coding: utf-8 -*-
"""D8 - eval.py:评估闭环(检索质量的唯一裁判)

职责:
    1. build_eval_set:语料挑 n 篇 → qwen-plus 生成问题 → 人工筛选
       → data/eval/eval_set.jsonl({question, target_title})
    2. evaluate:hit@5 / hit@10 / MRR + 最差 10 个 case 报告
    3. ab_test:改切片/k/重排等单变量 A/B 对比框架

对应:路线计划 D8 | 架构 §9
运行:python -m evaluation.eval
"""
from pathlib import Path

from common import config


def build_eval_set(n: int = 60, out: Path = config.DATA_EVAL / "eval_set.jsonl") -> None:
    """构造人工过目的评测集。TODO: D8 实现"""
    raise NotImplementedError


def evaluate(eval_set: Path, k_list: list[int] | None = None) -> dict:
    """计算 hit@k 与 MRR,输出报告 + 失败案例。TODO: D8 实现"""
    raise NotImplementedError


def ab_test(**variants) -> None:
    """单变量 A/B 对比,一张表出结果。TODO: D8 实现"""
    raise NotImplementedError

# -*- coding: utf-8 -*-
'''tests/unit/evaluation/test_eval.py —— 对应 src/evaluation/eval.py(D8 评测)

当前状态:占位骨架。src/evaluation/eval.py 目前还是 raise NotImplementedError。

到时候怎么写(先记下来):
  · 指标计算是纯函数,最适合单测:hit@k / MRR / NDCG 这些务必用**手算过的小例子**
    做断言 —— 指标公式写错是这类代码最常见的 bug,而指标错了没人看得出来。
    例子:3 个候选里正确答案排第 2 位,MRR 应为 1/2、hit@1 为 0、hit@2 为 1。
  · 评测集构造(JSONL 读写)复用 common/utils 那套流式写法。
  · A/B 对比的统计显著性部分(W5)可以后补。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D8 eval.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/evaluation/eval.py (D8) 实现后再编写')

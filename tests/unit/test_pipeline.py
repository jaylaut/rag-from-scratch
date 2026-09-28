# -*- coding: utf-8 -*-
'''tests/unit/test_pipeline.py —— 对应 src/pipeline.py(D5/D6 端到端门面 ask())

当前状态:占位骨架。src/pipeline.py 目前只有 22 行,ask() 还是 raise NotImplementedError。

说明:pipeline 是端到端入口(把检索 + 重排 + 生成串起来),它本身的单测价值不高
——"串起来能不能跑通"是集成测试的活(见 integration/)。所以本文件的定位是**小
单元**,只测那些可以拆出来独立验证的东西:
  · 参数传递:RECALL_K / TOP_M 这两层口径(召回宽、精读窄)有没有被正确透传;
  · 异常路径:索引文件不存在时是不是给了人话报错,而不是裸 Traceback;
  · 返回结构:带引用的答案字典的字段是否齐全(chunk_id / title / text)。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D5/D6 pipeline.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/pipeline.py (D5/D6) 实现后再编写')

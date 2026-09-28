# -*- coding: utf-8 -*-
'''tests/unit/online/test_rerank.py —— 对应 src/online/rerank.py(W5 进阶·重排)

当前状态:占位骨架。src/online/rerank.py 目前还是 raise NotImplementedError。

背景:这是 W5 的进阶能力,按 `系统架构说明.md` §6-R4 方案 B→A 演进,
优先级低于 D2~D6 主线,所以它的测试也是最晚补的。

到时候怎么写(先记下来):
  · 纯函数部分(prompt 组装、top-m 截断、分数归一化)用假数据测,不联网;
  · 两种实现路径要分别测:
      - 方案 A:cross-encoder / bge-reranker(需要额外依赖,跑 CPU);
      - 方案 B:LLM-as-reranker(Qwen 列表重排,**会花钱**,必须打
        @pytest.mark.network,默认过滤跳过);
  · 断言重点:重排只改变**顺序**、不改变候选集合 —— 这条能挡住绝大多数实现 bug。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 W5 rerank.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/online/rerank.py (W5 进阶) 实现后再编写')

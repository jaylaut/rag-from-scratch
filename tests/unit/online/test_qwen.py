# -*- coding: utf-8 -*-
'''tests/unit/online/test_qwen.py —— 对应 src/online/qwen.py(D6 prompt 组装 + 生成)

当前状态:占位骨架。src/online/qwen.py 目前还是 raise NotImplementedError。

到时候怎么写(先记下来):
  · **重点测纯函数部分**:prompt 组装、引用编号 [来源N] 的插入、token 预算截断。
    这些是最容易出 bug 且完全不需要联网的部分,性价比最高:
      - 超长上下文被截断后,引用编号必须仍然是连续的、且没有被截成半个;
      - 引用编号要能对应回 chunk_id(错一位就等于答非所问的来源);
      - 空候选 / 单候选这些边界不要崩。
  · 真正调用 Qwen API 的用例:**一律打 @pytest.mark.network**,默认被
    -m "not network" 过滤 —— 那是唯一真的会花钱的东西,必须能一键关掉。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D6 qwen.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/online/qwen.py (D6) 实现后再编写')

# -*- coding: utf-8 -*-
'''tests/unit/online/test_retrieve.py —— 对应 src/online/retrieve.py(D6 向量检索)

当前状态:占位骨架。src/online/retrieve.py 目前还是 raise NotImplementedError,
等 D6 落地后再写真实用例。

到时候怎么写(先记下来):
  · 检索的核心(prompt 拼装之前的排序/截断)用假数据测:
      - top-k 截断边界:k 大于候选数时不能崩;
      - 分数 desc 排序是否正确(别默认 faiss 返回就是有序的);
      - 元数据回填:chunk_id -> 原文的映射不能错位(错位是最常见也最隐蔽的 bug)。
  · 需要真实 FAISS 索引的用例归 tests/integration/;要连 Ollama 的额外打
    @pytest.mark.network(network 用例默认被 -m "not network" 过滤掉)。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D6 retrieve.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/online/retrieve.py (D6) 实现后再编写')

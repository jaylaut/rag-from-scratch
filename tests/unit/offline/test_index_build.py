# -*- coding: utf-8 -*-
'''tests/unit/offline/test_index_build.py —— 对应 src/offline/index_build.py(D4 建索引)

当前状态:占位骨架。src/offline/index_build.py 已写入 **8 个内部函数**
(_fmt_bytes / _default_paths / _infer_rows / _open_vectors / _audit_vectors /
_new_index / _add_shards / _count_rows),但 `build_index` / `sanity_check` 仍是
`NotImplementedError`,且 `:404` 代码截断(整文件当前无法 import)、`:148` 有一处
字节数校验错误 —— 两处均见 D4建索引开发指引.md §12。故本文件暂不写真实用例。

到时候怎么写(先记下来):
  · 用小矩阵造假向量(比如 8 条 1024 维),断言:
      - 索引 ntotal 与输入条数一致;
      - search 出来的 id 能正确回填到元数据(这是最容易写错下标的地方);
      - 归一化向量做 IP 检索时,自己跟自己的相似度是 1.0。
  · 元数据落盘/回读的那部分(JSONL),可以复用 tests/unit/common/test_utils.py
    那套 iter_jsonl 的经验。
  · 真实语料(GB 级)建库的那次 perf 验证归 tests/integration/,打 bigdata 标记。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D4 index_build.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/offline/index_build.py (D4) 实现后再编写')

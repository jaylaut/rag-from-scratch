# -*- coding: utf-8 -*-
'''tests/unit/offline/test_embed.py —— 对应 src/offline/embed.py(D3 向量化)

当前状态:占位骨架。src/offline/embed.py 的**第 1 层 7 个工具函数已落地**
(_get_session / _count_rows / _progress_path / _load_progress / _save_progress /
_post_batch / _l2_normalize),但**第 2~4 层尚未实现**(embed_texts / embed_query /
cosine / embed_chunks / __main__),故本文件暂不写真实用例。
(注:embed.py:76 有一处已知缺陷 `requests.Session` 缺调用括号,见 D3开发指引.md §12)

到时候怎么写(先记下来,免得临场现想):
  · 向量计算本身:喂 2~3 条小样本字符串,断言返回形状是 (n, 1024)、dtype 是
    float32、且**已 L2 归一化**(每条向量模长为 1.0,tolerance 给 1e-6)。
    ★ 归一化这条最重要:faiss 的 IndexFlatIP 检索质量完全依赖它,漏了不会报错,
      只会悄悄变差 —— 属于"必须显式断言"的典型。
  · HTTP 请求:用 monkeypatch 把 requests.post 顶掉,别真连 Ollama。
    真连服务的那些归入 tests/integration/,并打 @pytest.mark.network。
  · 批大小 / 空输入 / 超长文本怎么分批,也可以顺手写。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:等 D3 embed.py 实现后再替换为真实断言。'''
    pytest.skip('等待 src/offline/embed.py (D3) 实现后再编写')

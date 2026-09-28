# -*- coding: utf-8 -*-
'''tests/integration/test_offline_chain.py —— D2→D3→D4 离线链路串联、跑真实产物

和 unit/ 的区别:
  · unit/ 喂手搓的小字符串,秒级返回,每次改代码都跑;
  · integration/ 读 data/ 下的真实产物,慢、可能需要 Ollama / Qwen,
    只在"某个阶段做完要验收"时跑。

本文件将来的内容对应 `D2切片开发指引.md §7` 的 V1~V6 验收标准:
  V1 块长分布(>550 占比 <=8%,且成因可解释)
  V2 重叠:相邻块"上块尾部 == 下块头部",重叠长度 ∈ [50, 100]
  V3 断句:块首字符的前一字符 ∈ 。！？ 或 \\n(硬切白名单块除外)
  V4 主键:chunk_id 全局唯一、每篇 chunk_index 从 0 连续
  V5 无损性(最强的一条):块0 + Σ(块i 去掉重叠前缀) == 原文,逐字相等
  V6 分布对齐:长文章 450~550 占比 70%+、块数/篇数 ≈ 1.9

★ 黑盒原则(指引 §8 第 5 条):这里只读产物文件做验证,**不要 import 被测模块的
  私有函数**来做断言 —— 否则"实现和验证共享同一个 bug",验证就白做了。
'''
import pytest

pytestmark = pytest.mark.integration


def test_placeholder() -> None:
    '''占位用例:等 D2 全量切片跑完后,把 V1~V6 的验收断言搬到这里。'''
    pytest.skip('等待 data/chunks/wiki_chunks.jsonl 产出后再编写 V1~V6 验收')

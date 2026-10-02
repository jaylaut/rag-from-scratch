# -*- coding: utf-8 -*-
'''tests/unit/offline/test_chunk.py —— 对应 src/offline/chunk.py(D2 切片,命门模块)

当前状态:已覆盖 Step 3 的 `_pick_overlap` / `_assemble`(重叠回填)。
          Step 2 `_pack` 的 T1~T11 尚未编写(见下方来源)。

还不写的部分 —— 真实用例来源:项目根 `D2切片开发指引.md` §6 的 T1~T11:
      T1  460 字单段多句              -> 1 块,长 460
      T2  600 字单段(多句)          -> 2 块,首块 450~550
      T3  三段各 200 字              -> 1 块 604(R4 强制续句,白名单,不是 bug)
      T4  两段 200+300               -> 1 块 502(段落优选断点)
      T5  单句 700 字                -> 硬切 2 片,均 <=500
      T6  前块 500 + 尾块 30(跨段) -> 并入 -> 532
      T7  前块 640 + 尾块 30(跨段) -> 672 > 650 -> 尾块独立保留
      T10 两篇文章各 2 块             -> chunk_id 无碰撞、各自从 0 连续
      T11 随机 5 篇重建              -> 重建文本与原文逐字相等
    (T8/T9 两条重叠相关的用例已经由本文件的 test_pick_overlap_* 系列覆盖了)
    注:T3/T4/T6/T7 的长度含跨段 \\n\\n 的 2 个字符,与实现里 cur_len 同口径。

§7 的 V1~V6 是全量跑完后的黑盒验收(要读 2.32GB 产物),归 tests/integration/。

写用例的三条硬要求:
  1. 只喂手搓的小字符串。禁止在这个文件里读 data/articles/wiki_zh.jsonl。
  2. 不许恒真断言(项目铁律)。每条断言都要预设"把实现改错就必须 FAIL"的反例。
     ★ 这个文件刚建立就证明了它的牙齿:下面的 test_pick_overlap_* 一跑起来
       立刻抓出实现里把 prev[start:] 写成了 prev[:start](抄成了"头"而不是"尾")。
  3. 阈值一律从 config 取,不在测试里写死 50/100 —— 参数调整时测试要跟着变,
     而不是继续守护一个早就作废的数字。
'''
import pytest

from common import config
from offline import chunk

pytestmark = pytest.mark.unit          # 整个文件都打 unit 标记

# 阈值从 config 取:项目约定是"参数单一来源",测试里不要再抄一份常数
OVERLAP = config.CHUNK_OVERLAP          # 50  —— 目标重叠(分支 a/c 的结果)
OVERLAP_MAX = config.CHUNK_OVERLAP_MAX  # 100 —— 句对齐回退的最远距离(分支 b 的上限)


# ======================================================================
# _pick_overlap:R7 重叠决策的三条分支
# ======================================================================
def test_pick_overlap_branch_a_exact_50() -> None:
    '''分支 a:候选切点左边刚好是句号 -> 重叠 = 精确 50 字

    构造:100 个"甲" + 句号 + 50 个"乙",总长 151。
    切点 = 151 - 50 = 101,而下标 100 正是那个句号 -> 落在断点上,直接抄最后 50 字。
    ★ 关键断言是 ov == prev[-50:]:它证明抄的是【尾部】。写成 prev[:50] 时这里必挂。
    '''
    prev = '甲' * 100 + '。' + '乙' * 50

    ov = chunk._pick_overlap(prev)

    assert len(ov) == OVERLAP, f'期望精确 {OVERLAP} 字,实际 {len(ov)} 字'
    assert ov == '乙' * 50, '重叠内容应当是末尾那 50 个"乙"'
    assert ov == prev[-OVERLAP:], '重叠必须是 prev 的【尾部】,写反切片这里会挂'


def test_pick_overlap_branch_b_sentence_aligned() -> None:
    '''分支 b:切点落在句中 -> 回退到最近的句号之后,重叠落在 (50, 100]

    构造:80 个"甲" + 句号 + 80 个"乙",总长 161。
    切点 = 111,落在第 31 个"乙"上(句中)-> 回退到句号之后(start = 81),
    重叠 = 161 - 81 = 80 字,刚好是完整的后半句。
    '''
    prev = '甲' * 80 + '。' + '乙' * 80

    ov = chunk._pick_overlap(prev)
    start = len(prev) - len(ov)          # 重叠的起点

    assert ov == '乙' * 80, '应当整句退回(抄回句号之后的全部 80 个"乙")'
    assert OVERLAP < len(ov) <= OVERLAP_MAX, (
        f'分支 b 的重叠必须落在 ({OVERLAP}, {OVERLAP_MAX}] 之间,实际 {len(ov)} 字'
    )
    assert prev[start - 1] == '。', '重叠必须以句首对齐:它的前一个字符得是句号'
    assert prev[-(len(ov)):] == ov, '重叠必须是 prev 的尾部'


def test_pick_overlap_branch_c_no_break_at_all() -> None:
    '''分支 c:前面一个断点都没有(超长句)-> 退无可退,接受精确 50 字的句中截断

    构造:500 个"丙",没有任何句号 -> rfind 全部返回 -1 -> 就地截断 50 字。
    '''
    prev = '丙' * 500

    ov = chunk._pick_overlap(prev)

    assert len(ov) == OVERLAP
    assert ov == '丙' * 50
    assert ov == prev[-OVERLAP:], '即使整句都没有断点,抄的也必须是尾部'


def test_pick_overlap_break_too_far_falls_back_to_50() -> None:
    '''分支 c 的另一种成因:断点是有的,但离得太远(回退会超过 100 字)-> 放弃对齐

    构造:300 个"丁" + 句号 + 300 个"戊",总长 601。
    最近的断点在下标 300,回退过去要抄 601 - 301 = 300 字,远超 100 -> 不划算,
    于是放弃句对齐,退化成精确 50 字。
    ★ 这一条和 test_pick_overlap_branch_c 的区别:那条是"没找到",这条是"找到了但太远"。
    '''
    prev = '丁' * 300 + '。' + '戊' * 300

    ov = chunk._pick_overlap(prev)

    # 先自证这个用例确实构造出了"回退会超过上限"的局面
    assert len(prev) - (prev.rfind('。') + 1) > OVERLAP_MAX
    assert len(ov) == OVERLAP, '断点太远时必须退化成精确 50'


def test_pick_overlap_short_prev_returns_empty() -> None:
    '''防御分支:上一块还不到 50 字 -> 返回空串

    为什么必须拦下来:此时 cut = len - 50 会是负数,Python 的负下标会绕到字符串
    末尾取字符,切出来的结果既错又不报错(python 里最难受的那类 bug)。
    '''
    assert chunk._pick_overlap('短') == ''
    assert chunk._pick_overlap('短' * OVERLAP) == ''      # 刚好等于 50 也不重叠


def test_pick_overlap_always_suffix() -> None:
    '''通用性质:无论走哪条分支,返回值都必须是 prev 的【后缀】

    ★ 这条是防"切片写反"的总闸门。历史上(以及此刻的实现里)最容易犯的错就是把
      prev[start:] 写成 prev[:start] —— 那样会得到"头",长度还经常碰巧对,
      单点用例很难发现,但一条 endswith 就能钉死。
    '''
    cases = [
        '甲' * 100 + '。' + '乙' * 50,        # 分支 a
        '甲' * 80 + '。' + '乙' * 80,         # 分支 b
        '丙' * 500,                          # 分支 c(无断点)
        '丁' * 300 + '。' + '戊' * 300,       # 分支 c(断点太远)
        '第一句。' + '己' * 400 + '。' + '庚' * 60,
    ]
    for prev in cases:
        ov = chunk._pick_overlap(prev)
        assert prev.endswith(ov), (
            f'返回值不是 prev 的后缀,很可能把切片写反了。'
            f'prev 尾 20 字={prev[-20:]!r},ov 头 20 字={ov[:20]!r}'
        )


# ======================================================================
# _assemble:把重叠拼回下一块的开头
# ======================================================================
def test_assemble_second_block_starts_with_prev_tail() -> None:
    '''同段相邻块:第二块必须以【上一块的尾部 50 字】开头,中间不加分隔'''
    first = '甲' * 100 + '。' + '乙' * 50      # 151 字,分支 a 触发
    second = '丙' * 60
    # 手写 _pack 的产物格式:[[(段落号, 句子), ...], ...];这里每块只有一个句子
    chunks = [[(0, first)], [(0, second)]]

    texts = chunk._assemble(chunks)

    assert len(texts) == 2
    assert texts[0] == first, '第一块没有上一块,不该被改动'
    assert texts[1] == ('乙' * 50) + '丙' * 60, '同段相邻 -> 中间不加任何东西'


def test_assemble_cross_paragraph_keeps_blank_line() -> None:
    '''跨段落相邻块:第二块 = 上一块尾 50 字 + \\n\\n + 本块正文

    ★ \\n\\n 不属于任何块的内部(_chunk_text 看不见它),只能由 _assemble 按
      "两块交界处的段落号变了"补回来。少了它,重建原文会把相邻段落粘在一起。
    '''
    first = '甲' * 100 + '。' + '乙' * 50
    second = '丙' * 60
    chunks = [[(0, first)], [(1, second)]]     # 段落号 0 -> 1,表示跨段

    texts = chunk._assemble(chunks)

    assert texts[1] == ('乙' * 50) + '\n\n' + '丙' * 60


# ======================================================================
# 无损性(V5)回归:2026-09-27 在真实语料上抓到的两个"静默丢字"bug
#
# 这两个 bug 的共同特点:不报错、不崩、肉眼看产物也完全正常,
# 只有把各块去掉重叠拼回去跟原文逐字比对(V5)才会暴露。
# 前 300 篇抽样只暴露了第 1 个,第 2 个要到 2 万篇才现身 ——
# 所以这里必须留用例,不能指望抽样。
# ======================================================================
def _rebuild(texts: list[str]) -> str:
    '''把各块去掉重叠前缀后拼回原文(与验收脚本同一口径)

    重叠长度怎么定:取 [0, OVERLAP_MAX] 里【最小的】k,
    使"上一块以 cur[:k] 结尾"且"去掉 k 之后能接上原文"。
    ★ 这里只用到 texts 自身,不依赖实现内部状态(黑盒)。
    '''
    out = texts[0]
    prev = texts[0]
    for cur in texts[1:]:
        k = None
        for cand in range(0, OVERLAP_MAX + 1):
            if len(cur) < cand:
                break
            if cand > 0 and not prev.endswith(cur[:cand]):
                continue
            k = cand
            break
        assert k is not None, '找不到重叠长度,说明产物结构本身有问题'
        out += cur[k:]
        prev = cur
    return out


def test_v5_leading_punctuation_not_lost() -> None:
    '''bug 1:段首/句首是全角句号时,旧正则会把它吞掉

    旧正则 `[^。！？]+[。！？]?` 要求一句里【至少有一个非标点字符】,
    于是 '。这种潮流最早…' 里的句首 '。' 匹配不上,被静默丢弃。
    真实语料里这是 D1 清掉维基模板/引用后留下的残句,前 300 篇就有 41 篇中招。
    ★ 反例:把 _SENT_RE 改回旧写法,这里的长度断言立刻挂。
    '''
    text = '第一段的正常内容。\n\n。这是段首带句号的一段，后面还有别的内容。'

    texts = chunk.sliding_window(text)

    assert ''.join(texts).count('。这是段首') >= 1 or _rebuild(texts) == text, \
        '段首的句号被吃掉了'
    assert _rebuild(texts) == text, '丢字后重建必然不等于原文'


def test_v5_paragraph_trailing_space_not_lost() -> None:
    '''bug 2:段落末尾的空格被 split_paragraphs 的 strip() 吃掉

    旧代码 `para = para.strip()`,原文段落 '连音 ' 被变成 '连音',少 1 个字。
    ★ 反例:把 strip() 加回去,这条立刻挂(该段会少一个空格)。
    '''
    text = '第一段的内容在这里。\n\n连音 \n\n以下词组说明别的规则。'

    texts = chunk.sliding_window(text)

    assert _rebuild(texts) == text, '段落首尾空白必须原样保留'
    # 直接钉死那个空格:它必须出现在某一块里
    assert '连音 ' in ''.join(texts), '段尾空格丢了'


def test_v5_whitespace_only_paragraph_not_lost() -> None:
    '''边界:一个段落全是空格时,也不能被当成"空句"过滤掉

    _stream 里旧的 `if s.strip()` 会把纯空白句子判成空而丢弃。
    '''
    text = '第一段在这里。\n\n   \n\n第三段在这里。'

    texts = chunk.sliding_window(text)

    assert _rebuild(texts) == text

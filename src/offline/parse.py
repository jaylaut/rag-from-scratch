# -*- coding: utf-8 -*-
"""D1 - parse.py:清洗提取(原始 XML → 干净文章 JSONL)

职责:
    1. 调 wikiextractor(仅支持 Python ≤3.12)把 dump 剥成正文 JSON 分片
    2. 二次清洗: 过滤 <50 字文章、去控制字符与维基转换标记、繁简统一(opencc t2s)
    3. 流式合并全部分片 → 单个 wiki_zh.jsonl(一行一篇, 交给 D2 切片) —— to_jsonl

对应:路线计划 D1 | 架构 §3.2
运行:python -m offline.parse            # 断点续跑(已存在的输出跳过)
      python -m offline.parse --force   # 全量重跑(覆盖已有输出, 改了清洗规则后使用)
依赖:标准库 + opencc-python-reimplemented(繁简统一用; 不想装就把 _ENABLE_OPENCC 改成 False)
"""
from __future__ import annotations

import html
from pathlib import Path
import subprocess
import sys
import logging
import time
import json # 解析分片里的每行JSON

from multiprocessing import Pool
import unicodedata #NFC 规范化
import os # os.cpu_count()
import re
# ======================================================================
# 二次清洗:常量与正则(私有助手必须位于模块顶层)
# ======================================================================
'''
【为什么这些常量和助手函数必须写在"模块顶层",不能塞进类或函数内部】
    Windows 下 multiprocessing 用 spawn 方式创建子进程,有两处硬约束:
      1) 子进程会【重新 import 一次本模块】(以 __mp_main__ 之名执行模块体),
         所以模块顶层的 _RE_* 正则会再编译一遍,子进程里这些名字天然存在;
      2) 传给子进程执行的 worker 函数是按 "模块名.函数名" 序列化(pickle)的,
         子进程必须能按这个名字找回"同一个对象"。
    若 _clean_one_shard / _clean_article 被定义在类里或某个函数内部,
    子进程找不到对应对象,会抛:
        PicklingError: Can't pickle <function ...>: it's not the same object
    因此本文件所有二次清洗助手函数一律留在模块顶层。

【为什么日志不放在 _clean_article 里逐篇打印】
    实测语料共 4,677 个分片 / 1,553,051 篇文章(约 155 万篇)。
    若在清洗函数里逐篇 logger.info,日志会刷出上百万行,比数据本身还大。
    所以: _clean_article 保持"纯函数"(不做 IO、不打日志,方便单独测试),
          进度与异常日志全部收敛到"分片级"(_clean_one_shard)和
          "总控级"(clean_shards),每个分片只留一行汇总。
'''

# ---------------- 路径与阈值(改行为只动这里) ----------------
'''
_OUTPUT_DIR
    二次清洗的默认输出目录(Path 对象,后面可直接用 / 运算符拼路径)。
    每篇保留的文章写成一行 JSON,落到 _OUTPUT_DIR / (分片名).jsonl,
    如 .../data/articles/AA_wiki_00.jsonl。

_MIN_CHARS = 50
    有效字数下限,低于它判废(计入 dropped)。
    依据: 对 130 篇真实样本做过长度分布分析 —— 垃圾簇集中在 <40 字
    (消歧义页 5~31 字),100 附近并没有天然断档;卡 100 会误杀约 7% 的
    真实短条目(年号、小电影、小人物这类 26~75 字的 stub)。
    垃圾主要由"消歧义句式"和"魔词"两条模式规则拦截,阈值只做兜底。

_MAX_CHARS = 50000
    单篇字符数上限,超出则截断。保护下游向量化:
    超长文章会让 embedding 又慢又糊,而这类文章在语料里极少。

_HEARTBEAT = 20000
    心跳日志间隔:一个分片每处理满 2 万篇就打一条 INFO,
    用来判断"进程还活着、没卡死"。常规分片只有几十~几百篇,一般不触发。

_MERGE_BATCH = 10000
    to_jsonl 合并阶段的"攒批"大小:攒够 1 万行才调一次 write 落盘。
    为什么需要:全量约 136 万行,若逐行调 write, 就是 136 万次 Python 层调用,
    即便有文件缓冲挡掉一部分系统调用, 纯开销也不小;攒批到 1 万行后
    write 次数降到约 140 次, 合并更快(实测全量 24.8s 完成)。
    与 _HEARTBEAT 的区别:那个是"日志间隔", 这个是"写盘粒度", 别混用。

_MAX_BADLINES = 5
    每个分片最多"详细记录"几条坏行(JSON 解析失败的原文)。
    超出后只累加计数、不再打印,防止一个损坏文件把日志刷爆。

_FORCE = False
    是否覆盖已存在的输出文件。
    False(默认)= 断点续跑: 输出已存在就跳过该分片;
    True        = 全量重跑: 覆盖已有输出(改了清洗规则后需要重跑时用)。
    开关方式: clean_shards(..., force=True) 或命令行 python -m offline.parse --force。

_ENABLE_OPENCC = True
    繁简统一开关。True = 把正文里的繁体字统一转成简体(二期功能,现已启用)。
    ★ 打开它需要先装依赖: pip install opencc-python-reimplemented
      (纯 Python 轮子,Windows 免编译;只用 3 个标准库之外的这一个包)
    ★ 没装包又打开开关时,clean_shards 会在启动阶段就报错并给出安装命令 ——
      这是刻意的"快速失败",防止你以为做了繁简统一、其实产出还是繁体。

_OPENCC_CONFIG = 't2s'
    opencc 的转换配置名(字符串):
        t2s   繁体 -> 简体(选它: 只做字形归并, 不擅自替换地区词汇)
        tw2sp 台湾正体 -> 简体(额外做"计程车->出租车"这类地区词替换, 改动更大)
        s2t   简体 -> 繁体(本项目不用)

_CONV_MAX_ROUNDS = 8
    字词转换标记"交替迭代"的最大轮数(规则 2 / 2b / 3 三种规则轮流跑)。
    为什么需要多轮: 语料里存在嵌套标记, 外层要等内层先被拆掉才匹配得上。
    正常文本 2 轮内收敛(第 2 轮一个都没替换 => 提前 break), 8 只是安全上限。
'''
_OUTPUT_DIR   = Path('E:/python/RAG/rag-from-scratch/data/articles')
_MIN_CHARS    = 50
_MAX_CHARS    = 50000
_HEARTBEAT    = 20000
_MERGE_BATCH  = 10000
_MAX_BADLINES = 5
_FORCE        = False
_ENABLE_OPENCC = True
_OPENCC_CONFIG = 't2s'
_CONV_MAX_ROUNDS = 8

# ---------------- 正则规则集(模块加载时编译一次,155 万篇复用) ----------------
r'''
re.compile(模式字符串)
    把正则【预先编译】成 Pattern 对象。为什么不在函数里每次写 re.sub('...'):
    那样每处理一篇文章都要重新解析一遍正则表达式;提前编译一次、全程复用,
    可以省掉上百万次重复解析 —— 这是本清洗环节最主要的性能优化手段。

★ 下面每条说明块都写成"字母 r + 三单引号"的原始字符串形式,原因有二:
    1) 说明里含 \s \x00 等反斜杠,若用普通字符串,Python 3.12 会报
       SyntaxWarning: invalid escape sequence;
    2) 原始字符串里反斜杠原样保留,不会被当成转义符解释掉。
'''

_RE_HTML_TAG = re.compile(r'<[^<>]{1,200}>')
r'''
1. HTML 标签壳
    <            字面量:左尖括号
    [^<>]        否定字符类:任何一个"不是 < 也不是 >"的字符(类首 ^ 表示取反)
    {1,200}      前面的字符类重复 1~200 次 —— 设上限是防止极端情况贪婪吞掉整段正文
    >            字面量:右尖括号
    匹配: <mark class="template-facttext" title="需要提供文獻來源"> 、</p>
    用法: 必须先 html.unescape() 把 &lt; 还原成 <,否则匹配不到转义形态;
          替换成空串 = 只删"标签壳",标签包裹的正文(如 二战)被保留。
    风险: 正文里若出现 "1 < 2 > 0" 这种数学写法会被误删(概率极低),
          要更严格可写成 <[a-zA-Z/][^<>]{0,199}>
    ★ 本规则【刻意只跑一遍】, 不要改成"迭代到不动点"。原因:
      规则 2/2b/3 与规则 4 反复执行都安全(它们只删"标记"); 但本规则一旦迭代,
      会把"删掉内层 <x> 之后才露出成对形状"的**代码/数学内容**一并删掉。
      实测: 归并排序里的 `Array[idxLeft] < RightSubArray[idxRight]) {` 这类 C++ 片段,
      迭代后会被整段吃掉(全量实测残留 89 篇 → 迭代后归零, 代价却是丢正文)。
      那 89 篇的残留全是源代码/数学式里的尖括号(如 i<MAX_TREE_SIZE、
      template< typename T >、0<2<4<...<1<3<5), 属真实内容, 留着比删掉好。
'''

_RE_CONV_BLOCK = re.compile(r'-\{[^{}|]*\|[^{}]*\}-')
r'''
2. 整块字词转换标记(维基的繁简/地区词转换语法, 带竖线的那一类)
    -\{          字面量 -{   ({ 在字符类之外是量词元字符,必须转义)
    [^{}|]*      规则标志位: 任意"不含花括号、也不含竖线"的字符,0 个或多个
                 ★ 这里刻意放宽成"任意": 真实语料里标志位不限于一个字母, 还有
                   -{zh;zh-hans;zh-hant;zh-cn;zh-hk;zh-sg;zh-tw|zh:;zh-hans:;zh-hant:;}- 
                   这种多语言"空变体清单"(模板残留, 整块都是垃圾, 应当整体删除)。
                   若按旧写法限定单个字母, 这类块匹配不上, 会被规则 3 拆壳,
                   把 "zh;zh-hans;...;zh-hant:;" 留在正文里(实测污染 3365 篇)。
    \|           字面量竖线 (| 是"或"运算符,必须转义)
    [^{}]*       标记内容: 任意不含花括号的字符
                 (排除花括号 => 遇到第一个 } 就停下,不会跨块吞噬后面的正文)
    \}-          字面量 }-
    匹配: -{H|zh-cn:重定向;zh-tw:重新導向;}-      (样本实测 7.2% 的文章含它)
    用法: 与规则 2b、3 交替迭代替换,直到某一轮"一个都没替换"为止
          (见 _clean_article 第 5 步; 迭代是为了收敛嵌套块并保证幂等)。
'''

_RE_CONV_VARIANT = re.compile(r'-\{([^{}|]*[:;][^{}|]*)\}-?')
r'''
2b. 变体列表(不带竖线, 但内容里含 : 或 ;)
    形状: -{zh-cn:域;zh-tw:體}-          -{zh-hans:震波;zh-hant:激波}-
          -{zh-hant:;zh-hans:}-         (值为空 => 整块是垃圾, 替换成空串)
    含义: 同一个词在不同地区变体下的不同写法。既然目标是"简体统一",
          正确做法不是把整块删掉, 而是按优先级挑出简体那一支的值
          (取值逻辑见 _pick_variant; 例如 'zh-cn:域;zh-tw:體' -> '域')。
    正则: ([^{}|]*[:;][^{}|]*)  —— 捕获组里必须出现 : 或 ; 之一,
          这样才能与规则 3 的普通行内标记 -{出}- 区分开
          (那个没有冒号分号, 要保留中间那个字)。
          末尾的 \}-? 表示结尾短横可有可无(语料里偶有漏写尾横的情况)。
    实测: 2.7% 的文章含它(样本 235/8746), 多出现在句子中间, 例如
          "這些物件的結構性質被探討於群、環、-{zh-cn:域;zh-tw:體}-等抽象系統中",
          只拆壳不取值的话, 句中会留下 "zh-cn:域;zh-tw:體" 这种垃圾。
'''

_RE_CONV_INLINE = re.compile(r'-\{([^{}]*)\}-')
r'''
3. 行内转换标记(删掉标记、保留花括号里的字)
    -\{  \}-     首尾字面量,同上
    ([^{}]*)     捕获组(圆括号):把花括号里的内容"抓"出来,
                 替换时用反向引用 \1 把它放回原处
    行为: -{}-   -> 空       (捕获组抓到空串)
          軍-{}-團 -> 軍團     (删掉标记,保留前后两个字)
          -{出}-  -> 出       (抓到"出"并还原)
    ★ 执行顺序必须是"规则 2 / 2b 之后":
      那两类的内容外形上也满足本规则,若先跑本规则,会把整块转换标记
      当成行内标记处理,把 "H|zh-cn:重定向;..." 当正文留下来。
'''

_RE_CONV_DANGLING = re.compile(r'-\{[^{}\n]*$', re.M)
r'''
3b. 残缺的转换标记(这一行里再也找不到闭合的 })
    形状: 行尾只剩 -{H        (wikiextractor 偶发把 -{H| 的开头吃掉或截断)
    正则: -\{[^{}\n]*$      配合 re.M(多行模式): $ 在这里表示"行尾"而不是"整串结尾";
          [^{}\n]* 限定"这一行剩下的部分既没有 } 也没有换行"。
    为什么需要它: 残缺标记永远不成对, 任何"成对匹配"的规则都碰不到它;
          它只会腐蚀正文、污染向量, 所以按行尾整段清掉(样本实测 0.06% 的文章)。
    ★ 必须放在"交替迭代"【之后】执行: 要先把成对的块处理干净, 剩下的才是真残缺。
'''

_RE_EMPTY_PAREN = re.compile(r'（[\s，、;；/()（）]*）|\([\s,;]*\)')
r'''
4. 空括号噪声(模板参数为空时留下的壳)
    左分支(全角): （ [\s，、;；/()（）]* ）
        括号内只允许: 空白 / 全角逗号，/ 顿号、/ 分号;；/ 斜杠 / 半角括号 () / 全角括号（）
        0 个或多个 —— 只要括号里有汉字或数字就【不匹配】,所以实质内容安全
    右分支(半角): \( [\s,;]* \)
        半角括号写在字符类外必须转义成 \( \) (否则会被当成"分组括号")
    |             "或": 两个分支任一命中,整段(含括号本身)一起删除
    匹配:   （，）（）( )(,;)   —— 实测真实语料里高频出现
    不匹配: （1955年2月17日—） 这类有内容的括号会被完整保留
    ★ 踩坑记录: 本行曾被误写成 r'（[\s，、;；/()（）]*）|$[\s,;]*$'
      (全角括号丢失、半角括号写成了 $ 行尾锚点)。后果极严重: 左分支会退化成
      "删除所有空白/逗号/顿号/分号/斜杠",中文逗号和 \n\n 段落边界会被整片吃掉。
      改动这一行后务必自测,确认 "，" 和 "\n\n" 还在,（，）已被删除。
'''

_RE_INVISIBLE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f'
                           r'\u200b-\u200f\u202a-\u202e\u2060-\u2064\u00ad\ufeff]')
r'''
5. 控制字符 + 不可见字符(两行相邻的字符串字面量会自动拼成一条,只扫一遍文本)
    \x00-\x08    C0 控制符,【故意跳过】\x09 制表符与 \x0a 换行 —— 保住文本结构
    \x0b \x0c    垂直制表符、换页符(这两个要删)
    \x0e-\x1f    继续到 0x1f (\x0d 回车夹在中间,但前面已把 \r\n 统一成 \n,不会漏)
    \x7f         DEL 删除符
    \x80-\x9f    C1 控制字符
    \u200b-\u200f 零宽空格、零宽连接符、左右方向标记(LRM/RLM)
    \u202a-\u202e Unicode 双向文本嵌入/覆盖符(可用来隐藏文字方向)
    \u2060-\u2064 文字连接符等不可见字符
    \u00ad       软连字符(看不见,却会让中文分词与检索错位)
    \ufeff       BOM / 零宽不换行空格
    这类字符"看着像空白、实际是垃圾",合并成一条正则一次清完。
'''

_RE_WS = re.compile(r'\s')
r'''
6. 任意空白字符
    \s  空白类: 空格 / 制表 / 换行 / 回车 / 换页 / 垂直制表,
        Python3 下还包含全角空格 \u3000 等所有 Unicode 空白
    用途: _effective_len() 里删掉全部空白后数字符,衡量"真实信息量"
          (避免空行多的文章用 len() 量出虚高的长度)
'''

_RE_MAGIC_WORD = re.compile(r'__[A-Z][A-Z0-9_]+__')
r'''
7. 魔词(magic words)
    __           两个下划线开头
    [A-Z]        首字符必须是大写字母 —— 排除 "____" 这类纯下划线序列的误配
    [A-Z0-9_]+   大写字母 / 数字 / 下划线,1 个或多个
    __           两个下划线结尾
    匹配: __EXPECTSHORTPAGE__ (wikiextractor 官方给"短页"打的标签,命中即判废)、
          __NOTOC__、__NOEDITSECTION__ 等
    取舍: 限定全大写,正文里的英文单词几乎不会两侧带双下划线,误伤率极低。
'''

_RE_REDIRECT = re.compile(r'^\s*#(?:REDIRECT|重定向|重新導向)', re.I)
r'''
8. 重定向页首行
    ^            字符串开头(未加 re.M 标志,所以是"整篇文本的开头",不是每行开头)
    \s*          允许开头有任意空白
    #            字面量井号
    (?:...)      非捕获组: 只分组不抓取 —— 不做反向引用时用它,更省内存也更清晰
    REDIRECT|重定向|重新導向   三个备选,任一命中即可
    re.I         (第二个参数)忽略大小写 => redirect / Redirect / REDIRECT 都命中;
                 该标志对汉字无影响
    用在 _clean_article 最前面: multistream dump 含全部页面,重定向页的正文
    只有一行 "#重定向 目标",毫无检索价值,直接判废。
'''

_RE_DISAMBIG = re.compile(r'(可以指|可能指|可能意指|可能是指|意指事物)')
r'''
10. 消歧义句式
    (A|B|C|...)  捕获组 + 五个中文备选(此处不用 \1,写成 (?:...) 更规范):
                 可以指 / 可能指 / 可能意指 / 可能是指 / 意指事物
                 —— 130 篇样本实测归纳出的完整集合
    ★ 必须与"有效字数 < 200"联合判定(见 _clean_article 第 12 步),
      否则正文里顺带提到"可以指"的正常长文会被误判成消歧义页丢掉。
'''

# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------
'''
logging.basicConfig(...)
    - level: 全局最低级别。设为 INFO 时，INFO/WARNING/ERROR 都会输出；
             DEBUG 级别的内容被过滤掉。
    - format: 日志格式串。
        %(asctime)s  时间戳
        %(levelname)s 级别名（INFO/WARNING/ERROR）
        %(message)s  正文
      datefmt 指定时间戳格式，精确到毫秒 %f。
    - 若希望每次 import 本模块都重新配置，调用 logging.basicConfig(force=True)；
      此处未加 force，避免覆盖调用方已有的日志配置。
'''
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _human_size(n: int) -> str:
    """
    把字节数格式化成人类可读的字符串（B / KiB / MiB / GiB / TiB）。
    仅用于进度日志，不参与业务逻辑。
    """
    '''
    从 B 开始逐级除以 1024，直到小于 1024 或单位耗尽。
    size 为 float 以便显示小数。
    '''
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} EiB"


def _human_time(seconds: float) -> str:
    """
    把秒数格式化为 "1h23m45s" 形式，便于阅读长任务的耗时。
    """
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def extract_raw( dump: Path,
                 out_dir: Path,
                 *,
                verbose: bool = True,) -> list[Path]:
    '''
    # 调用 wikiextractor 把wikipedia xml dump 抽取成 JSON分片文件
    param dump: pathlib.Path,wikipedia dump 文件路径
    param out_dir: pathlib.Path
                   wikiextractor的输出目录。不存在时会自动创建，抽取结果形式如下：
                   out_dir/AA/wiki_00.jsonl
                   out_dir/AA/wiki_01.jsonl
                   ...
                   out_dir/ZZ/wiki_99.jsonl`

    return:[pathlib.Path]:排序后的、所有生成的json分片文件路径列表
    异常：
        FileNotFoundError:dump文件不存在
        RuntimeError:返回非零退出码，或未产生任何输出文件
    '''

    # 1.入参校验与规范化
    '''
    Path(x).expanduser()
        - 把开头的"~" 或 "Users"展开为对应用户的根目录
        - 例如：Path("~/data/a.xml").expanduser() → /Users/username/data/a.xml
        - 非 "~" 开头的路径原样返回。
    Path(x).resolve()
        - 把相对路径转换成绝对路径，并解析"." ".." 与符号链接
    '''
    t_start = time.perf_counter()  # 用于统计总耗时；perf_counter 单调递增，不受系统时间调整影响
    logger.info("=" * 70)
    logger.info("[0/4] 准备：规范化路径并校验入参")
    logger.info("=" * 70)

    dump = Path(dump).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    logger.info(f"      输入 dump : {dump}")
    logger.info(f"      输出目录  : {out_dir}")

    if not dump.is_file():
        # 这里额外打印实际解析到的绝对路径，便于发现反斜杠 \r 之类的转义问题
        logger.error(f"      dump 文件不存在: {dump}")
        raise FileNotFoundError(f"文件不存在: {dump}")

    '''
    Path.mkdir(parents=True, exist_ok=True)
        - parents = True: 创建父目录(如果不存在),一并递归创建
        - exist_ok = True: 如果目录已存在，则不会报错
        - 必要性：wikiextractor 的 -o 要求输出目录已经存在，否则报错推出
    '''
    dump_size = dump.stat().st_size
    logger.info(f"      dump 大小 : {_human_size(dump_size)}")
    logger.info(f"      dump 后缀 : {dump.suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"      输出目录已就绪（存在或已创建）")


    logger.info("-" * 70)
    logger.info("[1/4] 构造 wikiextractor 命令行")
    logger.info("-" * 70)
    '''
     2.构造wikiextractor 命令行
     使用 `python -m wikiextractor` 形式启动
        (1) sys.executable 
            - 当前python解释器的绝对路径，如/user/bin/python3.12
            - 用它替代裸'python'，可保证调用的是[当前正在运行本脚本]的解释器，从而命中同一个虚拟幻境里安装的Wikiextractor，避免环境错乱
        (2) '-m'
            - 等价于让解释器执行 python -m<module>
            - 会把 <module>当作包内模块搜索并执行，且 sys.path[0]为当前目录
            - 相比直接给脚本路径，可避免手动写 wikiextractor/EikiExtractor.py的绝对路径
        (3) "wikiextractor.WikiExtractor" 目标模块名
            - 包 wikiextractor 中的 WikiExtractor 模块。
            - 该模块带有 if __name__ == "__main__": 入口，能直接以模块方式执行。
            
        (4) --json  输出格式参数（关键）
            - 让 wikiextractor 每行输出一个 JSON 对象，而不是默认的 <doc>...</doc>。
            - 每个 JSON 对象的字段：
                { "id": <str>, "url": <str>, "title": <str>, "text": <str> }
            - 后续可直接 json.loads(line) 逐行解析，方便流式处理。
            - 注意：PyPI 上的 2.75 版本对 --json 支持不完整，建议装 GitHub master。
        (5) --bytes 1M 
            - 设置分片大小为 1M
            - 实际分片会略大于此值，因为按完整文档切分，不会把一篇文档拆开。
        (6) --no-templates
            - 不展开 {{...}} 模板，直接保留原始 wikitext。
            - 大幅提速、显著减小输出体积；代价是正文里会残留大量模板符号。
            
        (7) --processes N 
            - 并行工作进程数，默认取 CPU 核心数
            - 大 dump（如 enwiki 全量）建议设为物理核心数，能线性加速抽取阶段。
            
        (8) --filter_disambig_pages
            - 过滤掉“消歧义页”（disambiguation pages）。
            - 这些页面通常只有一堆同名条目链接，对训练/检索价值低。
            
        (9) --quiet 
             - 关闭进度条 / 进度信息输出，减少 stderr 噪音。
             
        (10) -o <OUTDIR>  输出目录参数
             - 指定抽取结果的根输出目录。
            - 要求该目录必须事先存在（上面已用 mkdir 保证），否则 wikiextractor 会报错。
            - 最终输出结构：
                <OUTDIR>/AA/wiki_00
                <OUTDIR>/AA/wiki_01
                <OUTDIR>/AB/wiki_00
                ...
              子目录名 AA、AB、... 依次递增；每个子目录默认最多 100 个分片（可调 --files）。
        (11) str(dump) 位置参数：输入 dump 文件路径
            - 支持 .xml、.xml.bz2、.xml.gz；wikiextractor 会自动识别压缩格式并解压。
            - 必须放在所有选项参数之后。
            - 用 str(dump) 转换，因为 subprocess 需要字符串而不是 Path 对象
              （subprocess 自 3.6 起其实也接受 PathLike，但显式 str 更清晰）。
    '''
    cmd: list[str] = [
        sys.executable,
        '-m',
        'wikiextractor.WikiExtractor',
        '--json',
        '-o',str(out_dir),
        str(dump),
        '--processes', '2'
    ]

    '''
     3.执行子进程
     subprocess.run(args,**kwargs) -- 运行子进程并等到结束，返回ComplatedProcess
     参数说明;
        1.args:
            - 命令及其参数组成的 list[str]。
            - 用列表而不是单个字符串，可避免 shell 解析、无需手动加引号、
              并规避 shell 注入（不经过 /bin/sh）
        2.capture_output=True
          - 等价于同时设置 stdout=subprocess.PIPE 和 stderr=subprocess.PIPE，
            把子进程的 stdout / stderr 收集到内存，供后面读取。
          - 副作用：大 dump 时进度输出可能很大，会占用内存；如担心可改为
            stdout=subprocess.DEVNULL。
        3.text=True
          - 让 proc.stdout / proc.stderr 返回 str 而不是 bytes。
          - 等价于旧写法 universal_newlines=True。

        4.encoding="utf-8"
          - 显式指定解码编码。
          - text=True 时的默认编码取决于 locale（Windows 上常是 GBK/cp1252），
            显式指定 UTF-8 更可移植。

        5.errors="replace"
          - 遇到非法字节用 U+FFFD（�）替代，不抛 UnicodeDecodeError。
          - wikiextractor 的进度输出偶尔混入非 UTF-8 字节，这样更健壮。

        6.check=False
          - 不因为非零返回码而自动抛 CalledProcessError。
          - 我们自己检查 returncode，这样才能把 stderr 一并写进异常，
            给调用者更有用的排查信息。

        返回值：subprocess.CompletedProcess，关键属性：
              .returncode  子进程退出码，0 表示成功
              .stdout      子进程标准输出（这里是 str）
              .stderr      子进程标准错误（这里是 str）
              .args        实际执行的命令列表
    '''
    logger.info(f"      完整命令  : {' '.join(cmd)}")

    logger.info("-" * 70)
    logger.info("[2/4] 启动 wikiextractor 子进程（大 dump 可能耗时数小时，请耐心等待）")
    logger.info("-" * 70)
    t_sub = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False
    )

    elapsed_sub = time.perf_counter() - t_sub

    logger.info(f"      子进程返回码: {proc.returncode}")
    logger.info(f"      子进程耗时  : {_human_time(elapsed_sub)}")
    '''
       proc.returncode
           - 0 表示成功；非 0 表示 wikiextractor 出错。
       proc.stderr
           - 捕获到的 stderr 文本，wikiextractor 的报错信息主要在这里。
           - 只取末尾 4000 字符（stderr[-4000:]）避免异常信息过长刷屏，
             因为真正的错误往往在最后几行。
    '''
    if proc.returncode != 0:
        raise RuntimeError(
            'wikiextractor 执行失败 \n'
            f"命令:{' '.join(cmd)}\n"
            f"返回码:{proc.returncode}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )

    logger.info("-" * 70)
    logger.info("[3/4] 收集输出分片文件")
    logger.info("-" * 70)
    '''
     4.收集输出文件
     Path.rglob(pattern)
        - 递归遍历 out_dir 下所有子孙目录，返回匹配 glob 模式的文件/目录 Path。
        - 返回的是生成器，配合 sorted() 才会一次性物化。
        - 模式 "wiki_*" 匹配文件 "wiki_00"、"wiki_01"…，
          忽略 wikiextractor 写在同目录下的其它文件（如日志）。

    Path.is_file()
        - 过滤掉名字形如 "wiki_xxx" 但其实是目录的条目，只保留常规文件。

    sorted(...)
        - 按 Path 的自然顺序（字符串序）排序。
        - 因为子目录是 AA、AB…ZZ 的字典序，文件名是两位数字 wiki_00..wiki_99，
          所以字符串序恰好等于“字典序 + 序号序”，结果稳定可复现。
    '''
    files: list[Path] = sorted(
        p for p in out_dir.rglob('wiki_*')
        if p.is_file()
    )
    '''
    若一个文件都没有，说明 wikiextractor 虽然返回 0 但没产出结果
    （例如 dump 为空、被过滤光），这通常也是异常情况，抛出以便上游感知。
    '''
    if not files:
        raise  RuntimeError(
            f"wikiextractor 未生成任何输出文件: {out_dir} \n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return files
def _effective_len(text:str) -> int:
    r'''
    计算"有效字数": 删掉全部空白字符后的字符数。

    param text: str,待统计的正文
    return: int,有效字符数

    为什么不直接用 len(text):
        len() 会把空格、制表符、空行也算成长度,一篇正文很短但排版空行很多的
        文章会量出虚高的长度,导致"长度门控"失效。去空白后再数更接近信息量。

    实现:
        _RE_WS 是模块级预编译的 \s (任意 Unicode 空白);
        Pattern.sub(替换成什么, 在哪替换) 把每个空白替换成空串;
        最后 len() 数剩下的字符。
    '''
    #有效字数 = 删除全部空白后的字符串长度(去空白后数字,比 len() 更能反映信息量)
    return len(_RE_WS.sub('',text))

# ---------------- 变体列表取值 + 繁简统一(opencc) ----------------
_VARIANT_PRIORITY = ('zh-cn', 'zh-hans', 'zh-sg', 'zh-my', 'zh',
                     'zh-hant', 'zh-tw', 'zh-hk', 'zh-mo')
r'''
变体优先级(用于 _pick_variant, 从前往后找, 命中即取):
    zh-cn  中国大陆简体   ┐
    zh-hans 简体          │ 目标语料是简体, 所以简体各支排最前
    zh-sg  新加坡简体     │
    zh-my  马来西亚简体   ┘
    zh     不转换(原文写法, 通常已是简体)
    zh-hant 繁体 / zh-tw 台湾 / zh-hk 香港 / zh-mo 澳门  -- 只在没有任何简体分支时才兜底
'''

def _pick_variant(content: str) -> str:
    r'''
    从变体列表里挑出"简体那一支"的值。

    param content: str —— 由正则 2b 的捕获组拿到的花括号内文, 形如 'zh-cn:域;zh-tw:體'
    return: str —— 挑中的值, 例如 '域'; 值为空时返回空串

    例子:
        'zh-cn:域;zh-tw:體'                     -> '域'
        'zh-hans:震波;zh-hant:激波'             -> '震波'
        'zh-hans:里瓦尔多;zh-hk:李華度;zh-tw:里瓦爾多' -> '里瓦尔多'
        'zh-cn:互联网档案馆;zh-tw:網際網路檔案館' -> '互联网档案馆'
        'zh-hant:;zh-hans:;'                    -> ''   (都是空值 = 垃圾, 清掉)

    实现要点(都是标准库字符串方法):
        str.split(';')       按分号拆项: 'zh-cn:域;zh-tw:體' -> ['zh-cn:域', 'zh-tw:體']
        str.partition(':')   把每项按【第一个】冒号拆成 (键, ':', 值) 三元组。
                             ★ 这里特意用 partition 而不是 split(':'), 因为
                               值里可能还带冒号(如 'zh-cn:12:00'), split 会拆成 3 段出错。
        str.strip()          去掉键值两侧空白: 语料里常见 'zh-hans:文件; zh-hant:檔案'
        字典 + for 循环      存成 {键: 值}, 再按 _VARIANT_PRIORITY 顺序查, 命中即返回
    '''
    pairs = {}
    for item in content.split(';'):
        key, sep, val = item.partition(':')
        if sep:                        # sep 非空 => 这一项确实含冒号, 是"键:值"
            pairs[key.strip()] = val.strip()
    for key in _VARIANT_PRIORITY:      # 按简优先序找
        if key in pairs:
            return pairs[key]
    return content                     # 一个已知键都没有 => 原样保留, 不瞎猜(宁可多留信息)


def _variant_repl(m: re.Match) -> str:
    r'''
    _RE_CONV_VARIANT.sub(...) 用的"替换函数"。

    param m: re.Match —— 正则匹配对象(re 模块匹配成功时交给回调的那个对象)
    return: str —— 用来替换本次匹配的文本

    知识点: Pattern.sub(第二个参数, 目标字符串)
        第二个参数既可以是字符串(可用 r'\1' 反向引用), 也可以是【函数】;
        传函数时每匹配到一处就调用一次, 参数是 Match 对象, 返回值即替换文本。
        这里必须用函数而不能用 r'\1' —— 因为要按变体优先级"挑一个值",
        不是简单地把捕获组原样填回去。
        m.group(1) 取第 1 个捕获组, 即花括号里的变体列表文本。
    '''
    return _pick_variant(m.group(1))


_OPENCC = None
r'''
opencc 转换器实例(模块级单例, 初值 None 表示"还没构造")。

为什么用"模块级变量 + 懒加载"而不是在 _clean_article 里直接 OpenCC(...):
    1) 构造 OpenCC 要读词表, 有成本; 全量 155 万篇每篇 new 一次会慢到不可接受;
    2) 多进程下每个子进程有独立的模块副本 => 每个进程各构造一次, 互不影响, 正合适;
    3) 初值 None 表示按需构造: 不启用繁简统一时, 连 opencc 包都不会被 import。
'''

def _get_opencc():
    r'''
    取本进程的 OpenCC 单例(首次调用时才真正构造)。

    return: opencc.OpenCC —— 繁体转简体的转换器实例
    依赖:   pip install opencc-python-reimplemented   (纯 Python 轮子, Windows 免编译)
    异常:   未安装时抛 ImportError; clean_shards 会在启动阶段就检查并给出安装提示。
    '''
    global _OPENCC                     # 声明要修改的是模块级变量
    if _OPENCC is None:                # 只有第一次会进这个分支
        from opencc import OpenCC      # ★ 放在函数内部 import: 不启用繁简统一时零依赖
        _OPENCC = OpenCC(_OPENCC_CONFIG)   # 配置名见顶部 _OPENCC_CONFIG
    return _OPENCC


def _clean_article(text: str) -> str | None:
    r'''
    二次清洗单篇正文 —— 纯函数: 不读写磁盘、不打日志,便于单独写测试。
    (之所以不打日志: 全量约 155 万篇,逐篇打 INFO 会刷出上百万行,见文件头说明)

    param text: str,wikiextractor 抽出的原始正文(含各类标记残留)
    return: str  —— 清洗后的正文,保留 \n\n 段落边界供下游 chunk 切分
            None —— 判废,调用方 _clean_one_shard 会把它计入 dropped、不落盘

    判废出口(按判定顺序):
        1) 空文档 / 全是空白
        2) 重定向页          正文形如 "#重定向 目标"
        3) 含魔词            __EXPECTSHORTPAGE__ 等 wikiextractor 官方标记
        4) 消歧义桩          命中"可以指/可能指/…"且有效字数 < 200
        5) 太短              有效字数 < _MIN_CHARS(50)
    通过全部判废出口后, 返回前还会做两件事(见第 14 步):
        · 繁简统一            _ENABLE_OPENCC 为 True 时, 把正文里的繁体转成简体
        · 超长截断            超过 _MAX_CHARS(50000) 时只保留前 5 万个字符

    ★ 各步顺序不可随意调换(如行内转换必须在整块转换之后、长度门控放在最后),
      每一步的"为什么"都写在对应注释里。
    '''
    # 第 1 步: 空值兜底。strip() 去掉首尾空白后若为空(falsy) => 没内容,直接判废
    if not text or not text.strip():
        return None

    # 第 2 步: Unicode 规范化 + 统一换行
    #   unicodedata.normalize('NFC', s): 把"一个汉字由基础字符+组合符号两个码位
    #   表示"的写法合并成单个标准字符。不做这步,后面的正则会因码位不一致而漏匹配。
    #   replace 把 Windows 的 \r\n 以及孤立的 \r 统一成 \n,
    #   保证后面 split('\n') 拆出的"行"和真实段落一一对应。
    text = unicodedata.normalize('NFC',text).replace('\r\n','\n').replace('\r','\n')

    # 第 3 步: 重定向页判废。
    #   Pattern.match(字符串) 只尝试从【开头】匹配(配合正则里的 ^ 与 \s*),
    #   比 search() 更快,也避免正文中部的 "#重定向" 字样被误判。
    if _RE_REDIRECT.match(text):
        return None

    # 第 4 步: HTML 实体反转义,再删标签壳
    #   html.unescape(s): &lt; -> < , &gt; -> > , &amp; -> & , &nbsp; -> 空格
    #   ★ 必须先反转义: 真实语料里标签是以转义形态存的(&lt;mark ...&gt;),
    #     不还原成正的 < > 就没法用 _RE_HTML_TAG 匹配到它。
    text = html.unescape(text)
    # 删标签壳、保留标签包裹的正文: <mark ...>二战</mark> -> 二战
    text = _RE_HTML_TAG.sub('',text)

    # 第 5 步: 处理三类字词转换标记(维基繁简/地区词语法)
    #   三种规则【交替】执行, 每一轮记下各自替换了多少处;
    #   某一轮"三处都是 0"就说明已经没有任何标记了 => 提前 break(不动点)。
    #     · 规则 2   -{H|zh-cn:重定向;zh-tw:重新導向;}-  整块删除(是规则条文, 不是正文)
    #     · 规则 2b  -{zh-cn:域;zh-tw:體}-              取简体那一支的值 -> "域"
    #     · 规则 3   -{}- / -{出}- / 軍-{}-團             删标记, 保留中间的字
    #   Pattern.subn(替换成什么, 目标) 返回元组 (新字符串, 替换次数) ——
    #   比"先 search 判断有没有、再 sub 替换"少扫一遍文本, 所以用它来当"是否需要再来一轮"的判据。
    #   ★ 为什么必须"交替 + 迭代", 不能各跑一遍就完事:
    #     语料里存在嵌套, 例如 -{zh;zh-hans;...|-{zh:;zh-hans:;}-}- ,
    #     外层因为 [^{}]* 不能跨花括号而暂时匹配不上; 交替执行时内层先被规则 2b 拆掉,
    #     上一轮结束后外层就没有内层花括号了, 下一轮规则 2 才能整块删掉它。
    #     这也是"幂等性"的要求: 洗过一遍的文本再洗一次, 结果必须不再变化。
    #   ★ 分成两个阶段, 顺序不能混:
    #     阶段一 反复跑两条【无损】规则(整块删除、变体取值), 直到一轮下来都没变化;
    #     阶段二 再跑【有损】的行内规则(删壳留字), 也迭代到不动点。
    #   ★ 为什么必须分开: 行内规则会把壳拆掉、把花括号里的内容当正文留下。
    #     若三个阶段混在同一个循环里, 会出现这种坏顺序 ——
    #       变体规则刚把内层拆掉 -> 外层刚"长得像"带竖线的整块 -> 行内规则抢先拆壳,
    #       于是一个本该整块删除的模板标记变成了留在正文里的垃圾(实测 'zh;zh-hans;zh-tw|')。
    #     所以: 先让无损的两条把该删的删干净, 最后才允许行内规则动刀。
    for _round in range(_CONV_MAX_ROUNDS):                     # ---- 阶段一: 无损两级 ----
        text, n_block   = _RE_CONV_BLOCK.subn('', text)             # 规则 2  : 整块删
        text, n_variant = _RE_CONV_VARIANT.subn(_variant_repl, text)  # 规则 2b : 取简体值
        if not (n_block or n_variant):                         # 两级都没替换 => 已收敛
            break
    for _round in range(_CONV_MAX_ROUNDS):                     # ---- 阶段二: 行内标记 ----
        text, n_inline = _RE_CONV_INLINE.subn(r'\1', text)          # 规则 3  : 删壳留字
        if not n_inline:                                       # 没替换 => 已收敛
            break
        # 迭代是为了处理 -{-{a}-}- 这类套娃: 拆掉内层后外层又成了 -{}-, 需要再拆一次。
        # 若循环跑满 _CONV_MAX_ROUNDS 还在变, 说明遇到预期外的形态; 正常语料 2 轮内收敛,
        # 这里不做额外处理(标记会留在正文里), 由验证脚本的残留检查兜底。

    # 第 6 步: 清掉"行内再无闭合 } "的残缺标记(规则 3b)
    #   ★ 必须在第 5 步的迭代之后: 成对的块先处理干净, 剩下的才是真残缺。
    text = _RE_CONV_DANGLING.sub('', text)

    # 第 7 步: 删控制字符与不可见字符(零宽空格、BOM、软连字符、方向标记等)
    text = _RE_INVISIBLE.sub('',text)

    # 第 8 步: 删空括号噪声(模板参数为空留下的壳,如 （，） （） ( ) )
    #   ★ 必须迭代到【不动点】, 不能只跑一遍 —— 与第 5 步同一个道理:
    #     有些括号是"套着的"(如 (（）) ), 第一遍删掉内层后, 外层才露出成对形状,
    #     而单遍 sub 从左到右只扫一次、不回头, 于是留下一个 () 残留(全量实测 2 篇)。
    #     跑不动点后这类残留归零, 且本规则只删"括号里没有任何实义字符"的壳,
    #     反复执行不会误伤正文(与 _RE_HTML_TAG 不同, 见该条注释末尾的说明)。
    for _round in range(_CONV_MAX_ROUNDS):
        text, n_paren = _RE_EMPTY_PAREN.subn('', text)
        if not n_paren:                                        # 没替换 => 已收敛
            break

    # 第 9 步: 魔词判废。search() 在整个字符串里找,一旦命中说明这是
    #   wikiextractor 标记过的短页/特殊页,比任何字数规则都可靠,直接丢。
    if _RE_MAGIC_WORD.search(text):
        return None

    # 第 9b 步: 重定向页【二次】判废(补漏)。
    #   既然第 3 步已经判过一次, 为什么还要再判? 因为有些重定向页的开头先挂了别的标记, 例如
    #       -{H|zh-cn:...;}-⏎#重定向 上海市
    #   第 3 步执行时 "#重定向" 并不在正文最前面(前面还有个转换块), match() 自然匹配不到;
    #   等第 4~6 步把这些前置标记删干净后, "#重定向" 才露到串首。
    #   在这里补判一次就能把这类漏网页捞出来(旧规则全量实测漏掉落库 117 篇)。
    if _RE_REDIRECT.match(text):
        return None

    # 第 10 步: 逐行清理
    #   str.split('\n') 按换行拆成行列表;wikiextractor 的正文里一行≈一个段落。
    lines = []
    for line in text.split('\n'):
        # strip() 必须放在"尾点判断"【之前】: wikiextractor 的标题行偶尔带尾随空格
        #   ("生平. "),若先判断 endswith('.') 会因末位是空格而漏判,标题尾点就留在正文里。
        #   先 strip 再判断,才不漏。空行不收集(段落边界在第 11 步统一重建)。
        line = line.strip()
        # wikiextractor 把 == 生平 == 渲染成 "生平."(ASCII 尾点);
        # 而中文正文用 "。" 结尾,所以"短行 + ASCII 尾点"可判定为章节标题。
        # 处理方式: 去掉尾点、保留标题文字 —— 标题是 chunk 的天然语义锚点,
        # 比整行删掉更有价值。len(line) <= 30 用来排除 "This is a sentence." 这类正文。
        if line.endswith('.') and len(line) <= 30:
            # ★ rstrip('.') 之后再 strip 一次: 删空括号后原位置会留下一个空格
            #   ("省 ()."  ->删 ()->  "省 ."  ->去尾点->  "省 "), 只去尾点的话
            #   这个空格就留在行尾, 最终变成正文末尾的尾随空格(全量实测 7 篇)。
            line = line.rstrip('.').strip()
        if line:
            lines.append(line)

    # 第 11 步: 用 '\n\n' 重新拼接 => 每个非空行变成一个段落,
    #   给下游 chunk 阶段提供可靠的段落边界(清洗阶段不把段落拍平成一行)。
    #   此处【无需】再压缩连续空行: lines 里的元素都已 strip 且非空,
    #   join 出来的相邻换行恒为 2 个,'\n\n\n' 在构造上就不可能产生
    #   (原先这里多挂了一次 r'\n{3,}' 替换,它永远不会命中,已删除)。
    text = '\n\n'.join(lines)

    # 第 12 步: 消歧义桩判废(联合判定,两个条件必须同时满足)
    #   只看"可以指"会误杀正文里顺带提到它的正常长文,所以加上
    #   "有效字数 < 200": 样本里的消歧义页正文都只有一句话(5~31 字)。
    if _RE_DISAMBIG.search(text) and _effective_len(text) < 200:
        return None

    # 第 13 步: 长度门控
    #   注意 _effective_len(text) 在这里只算一次并复用,避免重复扫全串。
    n = _effective_len(text)
    if n < _MIN_CHARS:
        return None

    # 第 14 步: 繁简统一(繁体 -> 简体)
    #   ★ 位置很关键, 两句话记住:
    #     · 放在长度门控【之后】—— 马上要丢掉的垃圾文档不必做昂贵转换;
    #     · 放在超长截断【之前】—— 保证截断后的正文也已经是简体。
    #   _get_opencc() 返回本进程的转换器单例(见该函数说明: 构造有成本, 每进程只做一次);
    #   .convert(字符串) 是 opencc 的转换方法, 把整段文本里的繁体统一成简体。
    #   性能参考: 全量 2.33 GB 实测约 0.9 MB/s(纯 Python 实现), 是本链路最重的一步,
    #   必须靠多进程分摊; 想关掉就把顶部 _ENABLE_OPENCC 改成 False。
    if _ENABLE_OPENCC:
        text = _get_opencc().convert(text)

    # 超长截断: text[:N] 是切片语法,取前 N 个字符(不改变类型),保护下游 embedding。
    #   ★ 截断点可能正好落在段落边界上 => 切完会留下尾随的 '\n\n'; 顺手 rstrip 掉,
    #     让"正文首尾不含空白"这条不变量对【所有】文章都成立
    #     (前面第 10~11 步已经保证常规文章首尾干净, 只有被截断的这篇会破例)。
    #     截断只截一次、rstrip 只去空白, 所以结果长度恒 <= _MAX_CHARS。
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS].rstrip()
    return text

def _clean_one_shard(task: tuple[Path, str]) -> dict:
    '''
    多进程 worker: 负责一个分片的全部工作(读文件 → 逐行清洗 → 写盘),
    并就地记录日志。它是 multiprocessing.Pool 调用的函数,
    因此必须位于模块顶层(Windows pickle 的硬约束,见文件头说明)。

    param task: tuple[Path, str],由 _discover_shards() 打包好的任务元组
                (in_path, out_name)
                    in_path : 输入分片路径,如 .../data/cleaned/AA/wiki_00
                    out_name: 输出文件名,  如 AA_wiki_00.jsonl
                ★ 为什么用元组: Pool 的 worker 只接受"一个参数",
                  需要一个对象同时携带"读哪个文件、写哪个文件"两条信息。
    return: dict,本分片的统计结果
                status  'ok'     正常完成
                        'skip'   输出已存在(断点续跑,直接跳过)
                        'failed' 本分片发生致命异常
                file    输入分片路径字符串(总控汇总失败清单时用)
                total / kept / dropped / badlines   各项计数
            ★ 内部的 records(待写入的正文行)与 _logged(已详记的坏行数)
              在成功返回前会被 pop 掉 —— 不把几万行正文回传主进程,
              否则多进程管道序列化这些数据会非常慢。

    日志(全部在本函数内):
        开始跳过 / 坏行 WARNING(最多 _MAX_BADLINES 条) / 心跳 INFO /
        分片汇总 INFO / 致命异常 ERROR(带完整堆栈)
    '''
    # 解包任务元组: in_path = 输入分片, out_name = 输出文件名
    in_path, out_name = task
    t0 = time.perf_counter()               # 本分片耗时统计起点
    # 输出路径 = 全局输出目录 / 输出文件名。
    # _OUTPUT_DIR 由 _init_worker() 在【子进程启动时】设好(见该函数说明),
    # 这样主进程传入自定义 out_dir 时子进程也能写对位置。
    out_path = _OUTPUT_DIR / out_name
    # 断点续跑: 该分片已清洗过就跳过(想重跑就删掉输出文件, 或用 clean_shards(force=True));
    # _FORCE 由 clean_shards(force=...) -> Pool(initializer=_init_worker) 传到子进程, 见 _init_worker。
    if out_path.exists() and not _FORCE:
        logger.info(f"      {in_path.name}:输出已存在，跳过")
        return {'status':'skip'}
    # stats 是本分片的统计容器;
    # 'records' 暂存待写入的行,'_logged' 是内部计数器(返回前两者都会被删掉)
    stats = {'status':'ok','file':str(in_path),'total':0,'kept':0,
             'dropped':0,'badlines':0,'records':[],'_logged':0}
    try:
        # Path.open(encoding='utf-8'): 以只读文本模式打开并显式指定 UTF-8
        #   (Windows 默认编码是 GBK,不指定会乱码或报 UnicodeDecodeError);
        #   with 语句保证即使中途异常也会关闭文件句柄。
        with in_path.open( encoding='utf-8') as fin:
            # enumerate(fin, start=1): 直接迭代文件对象是【逐行读取】(惰性,不整载内存),
            # 同时给出从 1 开始的行号,方便出错时定位是第几行。
            for lineno,line in enumerate(fin,start=1):
                line = line.strip()
                if not line:            # 跳过空行
                    continue
                stats['total'] +=1
                try:
                    # json.loads(字符串): JSON 文本 -> Python 字典。
                    # 每行就是一个对象的序列化结果,字段为 {id, url, title, text}
                    page = json.loads(line)
                except json.JSONDecodeError as e:
                    # 坏行不中断整个分片: 计数 + 记 WARNING(前几条附行号与原文片段),
                    # 之后只累加计数,继续处理下一行。
                    # e.msg 是解析器的错误说明,如 "Expecting value"。
                    stats['badlines'] += 1
                    if stats['_logged'] < _MAX_BADLINES:
                        stats['_logged'] += 1
                        # !r 用 repr 输出原文,里面的引号/换行不会被误解;
                        # line[:80] 只截前 80 个字符,避免一条超长坏行刷屏。
                        logger.warning(f"{in_path.name}第{lineno}行解析失败"
                                       f"({e.msg}):{line[:80]!r}")
                    continue
                # dict.get(键, 默认值): 键不存在时返回默认值而不抛 KeyError
                cleaned = _clean_article(page.get('text',''))
                if cleaned is None:          # 判废的文章计入 dropped,不落盘
                    stats['dropped'] += 1
                    continue
                stats['kept'] += 1
                # json.dumps(对象, ensure_ascii=False): Python 字典 -> JSON 文本。
                #   ensure_ascii=False 让中文原样写出(而不是 \u8607 这种转义),
                #   文件体积更小、人工排查时可直接阅读。
                # 先把每行攒进 records 列表,循环结束后一次性写盘 ——
                #   避免"每篇一次小写入"产生几万次磁盘 IO。
                stats['records'].append(json.dumps(
                    {
                     'id':page.get('id'),
                     'title':page.get('title'),
                     'url':page.get('url'),                 # 原始条目链接,便于溯源
                     'n_chars':_effective_len(cleaned),     # 清洗后的有效字数
                     'text':cleaned
                     },ensure_ascii=False))
                # 心跳日志: 超大分片时每 _HEARTBEAT(2 万)篇打一条,确认进程没卡死
                if stats['total'] % _HEARTBEAT == 0:
                    logger.info(f"      {in_path.name}:已处理{stats['total']}篇")
        # 一次性写盘: '\n'.join(records) 把每行 JSON 用换行连成整个文件内容,
        #   Path.write_text(内容, encoding='utf-8') 内部完成"打开→写入→关闭"。
        out_path.write_text('\n'.join(stats['records']),encoding='utf-8')
    except Exception:
        # 致命异常(磁盘满、权限不足、文件损坏、编码错误…):
        #   记录 ERROR + 完整堆栈,把本分片标记为 failed 后返回。
        # ★ 这里【故意不 raise】: 一个分片失败不应终止其余几千个分片,
        #   失败清单交给总控 clean_shards 汇总后统一报告。
        #   exc_info=True 让 logging 把当前异常的完整 traceback 一并打出来。
        logger.error(f"分片处理失败:{in_path}",exc_info=True)
        stats['status'] = 'failed'
        return stats
    # 成功返回前清理: 几万行正文(records)与内部计数器不参与向主进程的汇总
    stats.pop('records');stats.pop('_logged')
    # 分片级汇总日志: 一行看清"读了多少、留了多少、丢了几个、花了多久"
    logger.info(f"      {in_path.name}:共{stats['total']} 留{stats['kept']} "
                f"弃{stats['dropped']} 耗时 {_human_time(time.perf_counter() - t0)}")
    return stats

def _discover_shards(input_dir: Path) -> list[tuple[Path, str]]:
    '''
    递归发现 input_dir 下所有子目录(wikiextractor 的 AA/AB/... 桶)里的分片文件,
    并把它们打包成"任务元组"列表,交给多进程执行。

    ★ 本函数在【主进程】里运行(不在 worker 里): 分片清单只扫描一次,
      避免每个子进程各自 rglob 一遍目录,白白浪费 IO。

    param input_dir: pathlib.Path,分片根目录,如 .../data/cleaned
    return: [(分片路径, 输出文件名), ...] 元组列表,直接喂给 Pool.imap_unordered
            (Path 与 str 都能被 pickle 序列化,可以安全地传进子进程)

    为什么返回元组而不是单纯路径:
        _clean_one_shard(task) 里的 task 是 (in_path, out_name) 两元组;
        Pool 的 worker 只接受"单参数",所以必须打包成一个对象;
        输出文件名在这里一次性定好,worker 端不再关心命名逻辑。

    为什么做重名检测:
        AA/wiki_00 与 AB/wiki_00 的 stem 都是 wiki_00,若都用 wiki_00.jsonl
        会互相覆盖。全部唯一 → 用扁平名 wiki_XX.jsonl(与输入一一对应);
        存在重名 → 退化为带桶名前缀 AA_wiki_XX.jsonl,保证永不覆盖。
        (本项目实测 47 个桶 / 4,677 个分片,必然同名,走的是带前缀分支)

    异常: FileNotFoundError —— 目录下一个分片都没有。多半是 extract_raw() 没跑过,
          或输入目录传错了;显式报错胜过静默产出一个空结果。
    '''
    '''
    Path.rglob('wiki_*')
        - 递归遍历所有子孙目录,返回匹配模式的文件与目录(生成器,惰性求值)。
        - 模式 wiki_* 命中 wikiextractor 的分片,忽略同目录下的日志等其它文件。
    Path.is_file()
        - 排除名字形如 wiki_xxx 但实际是目录的条目。
    sorted(...)
        - 物化生成器并按字典序排序,保证每次运行处理顺序一致、结果可复现。
    Path.stem
        - 不含扩展名的文件名,如 wiki_00。
    Path.relative_to(input_dir)
        - 取相对路径,如 AA/wiki_00;.as_posix() 统一成正斜杠,
          再 replace('/', '_') 压平成 AA_wiki_00,用作带桶名前缀的输出名。
    '''
    # 递归收集: rglob 惰性遍历 -> is_file 过滤掉同名目录 -> sorted 物化成有序列表
    shards = sorted(p for p in input_dir.rglob('wiki_*') if p.is_file())
    if not shards:
        # 目录存在但没有任何分片: 多半是 extract_raw() 没跑,或输入目录传错了
        raise FileNotFoundError(
            f"未在 {input_dir} 下发现任何 wikiextractor 分片(模式 wiki_*)。\n"
            f"  请先运行 extract_raw() 生成分片,或确认输入目录是否正确。")

    # 重名检测: set() 去重后长度变短 => 存在跨桶同名分片
    stems = [p.stem for p in shards]
    duplicated = len(stems) != len(set(stems))   # set 去重后长度变短 = 存在同名
    if duplicated:
        logger.warning("      检测到跨桶同名分片(如 AA/wiki_00 与 AB/wiki_00),"
                       "输出文件名将带桶名前缀以避免覆盖")

    # 逐分片生成 (输入路径, 输出文件名) 任务元组
    tasks: list[tuple[Path, str]] = []
    for p in shards:
        if duplicated:
            # AA/wiki_00 -> AA_wiki_00.jsonl
            #   relative_to: 取相对 input_dir 的路径(AA/wiki_00)
            #   as_posix():  统一成正斜杠(Windows 的 \ 在字符串替换里容易踩坑)
            #   replace('/', '_'): 压平成 AA_wiki_00,得到永不重复的文件名
            out_name = p.relative_to(input_dir).as_posix().replace('/', '_') + '.jsonl'
        else:
            # wiki_00 -> wiki_00.jsonl(与输入分片同名一一对应)
            out_name = p.stem + '.jsonl'
        tasks.append((p, out_name))
    return tasks


def _init_worker(out_dir: Path, force: bool = False) -> None:
    '''
    Pool 子进程初始化钩子(initializer),每个子进程启动时调用一次。

    背景: Windows 用 spawn 方式创建子进程时会【重新 import 本模块】,
    子进程里的模块级变量取的是代码里写死的默认值,
    主进程里 "global xxx; xxx = ..." 的赋值【传不进子进程】。
    所以凡是要让"每个子进程都知道"的设置, 都必须通过 initargs 在这里重新赋一次:
        · _OUTPUT_DIR  子进程该往哪个目录写
        · _FORCE       是否覆盖已存在的输出

    param out_dir: 主进程解析后的输出目录,由 Pool(initargs=...) 传给每个子进程
    param force:   是否覆盖已存在的输出(断点续跑 vs 全量重跑),同样来自 initargs
    作用: 1) 把子进程的 _OUTPUT_DIR 设成主进程指定的目录, 这样
             _clean_one_shard 内 out_path = _OUTPUT_DIR / out_name 才正确;
          2) 把 _FORCE 同步给子进程, 否则传了 force=True 也不会覆盖;
          3) 预热 opencc: 子进程一启动就把转换器构造好(读词表有成本),
             避免第一个分片白白慢一截, 也让"装包缺失"早暴露。
    '''
    global _OUTPUT_DIR, _FORCE
    _OUTPUT_DIR = out_dir              # 赋值后,本子进程内 _clean_one_shard 写盘就用这个目录了
    _FORCE = force                     # 子进程内是否覆盖已有输出
    if _ENABLE_OPENCC:
        _get_opencc()                  # 预热繁简转换器(每个子进程各构造一次)


def clean_shards(input_dir: Path,
                 out_dir: Path = _OUTPUT_DIR,
                 processes: int | None = None,
                 force: bool = False) -> dict:
    '''
    二次清洗总入口: 清洗【目录】下所有子路径里的分片,结果写入 out_dir。

    param input_dir: pathlib.Path,分片根目录(不是分片列表!),
                     如 E:/python/RAG/rag-from-scratch/data/cleaned
                     内部用 rglob('wiki_*') 递归遍历 AA/AB/... 全部子目录,
                     调用方不需要自己收集文件列表。
    param out_dir:   pathlib.Path,输出目录(默认 data/articles),不存在会自动创建。
    param processes: int 或 None,并行进程数;None 时取 CPU 核数 - 2(留核给系统)。
    param force:     bool,是否覆盖已存在的输出文件。
                     False(默认)= 断点续跑: 输出已存在就跳过该分片(跑一半中断了可以接着跑);
                     True        = 全量重跑: 逐片覆盖写入(改了清洗规则/开关繁简统一后用)。
                     命令行等价写法: python -m offline.parse --force
    return: dict —— 汇总统计:
            total   本次处理的总篇数 / kept 保留篇数 / dropped 判废篇数
            badlines JSON 解析失败的坏行数
            failed  失败分片的文件名列表(_clean_one_shard 捕获到异常的那些)

    异常:
        NotADirectoryError: input_dir 不是有效目录
        FileNotFoundError : 目录下没有任何分片(提示先跑 extract_raw)

    ★ 形参 out_dir 的默认值 _OUTPUT_DIR 是在"函数定义时"求值一次后固定下来的,
      所以它等于文件顶部那个硬编码目录;调用方传了自定义 out_dir 就走调用方的值。

    日志: [4/4] 阶段头 / 分片发现数 / 并行进程数 / 每分片汇总(worker 内) /
          坏行 WARNING / 失败分片 ERROR(堆栈在子进程打) / None率>15% 警告

    用法: grand = clean_shards(Path(r'E:/python/RAG/rag-from-scratch/data/cleaned'))
    '''
    global _OUTPUT_DIR, _FORCE         # 主进程内也同步一次(子进程由 _init_worker 同步)
    _OUTPUT_DIR = out_dir
    _FORCE = force
    t_start = time.perf_counter()      # perf_counter 单调递增,不受系统时间调整影响

    # 入参规范化: expanduser 展开开头的 ~;resolve 转绝对路径并解析 . 与 ..
    input_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    logger.info('=' * 70)
    logger.info("[4/4] 二次清洗: 递归发现分片 -> 正则清理 -> 质量过滤(多进程)")
    logger.info('=' * 70)
    if not input_dir.is_dir():
        logger.error(f"      输入目录不存在或不是目录: {input_dir}")
        raise NotADirectoryError(f"不是有效目录: {input_dir}")

    # ★ 调用方只给"根目录",分片发现与命名由 _discover_shards 在内部完成;
    #   返回 [(分片路径, 输出文件名), ...],无分片时会抛 FileNotFoundError
    tasks = _discover_shards(input_dir)          # ★ 分片发现在方法内部完成
    logger.info(f"      输入目录  : {input_dir}")
    logger.info(f"      发现分片  : {len(tasks)} 个(含各子目录)")
    logger.info(f"      输出目录  : {out_dir}")

    # ---- 繁简统一(opencc)启动前检查 ----
    # ★ 快速失败: 开关打开却没装包时立刻报错并给出安装命令, 而不是"静默跳过"——
    #   静默跳过会让你以为产出已经是简体, 实际仍是繁体, 属于最难排查的错。
    if _ENABLE_OPENCC:
        try:
            t_cc = time.perf_counter()
            _get_opencc()                        # 主进程预热一次(同时验证依赖是否可用)
            logger.info(f"      繁简统一  : 已启用(配置 {_OPENCC_CONFIG}, "
                        f"转换器就绪 {_human_time(time.perf_counter() - t_cc)})")
        except ImportError:
            logger.error("      繁简统一已开启, 但 opencc 未安装。请任选其一处理:")
            logger.error(f"        1) 安装依赖: {sys.executable} -m pip install "
                         "opencc-python-reimplemented")
            logger.error("        2) 关闭开关: 把 parse.py 顶部的 _ENABLE_OPENCC 改成 False")
            raise                                # 直接抛出, 避免产出"假简体"
    else:
        logger.info("      繁简统一  : 未启用(_ENABLE_OPENCC=False, 输出保留原始繁简)")

    # ---- 覆盖/续跑模式提示 ----
    if force:
        logger.warning("      覆盖模式  : force=True —— 已存在的输出文件将被逐片覆盖重写")
    else:
        logger.info("      续跑模式  : 已存在的输出文件会被跳过(要全量重跑请加 force=True)")

    '''
    Path.mkdir(parents=True, exist_ok=True)
        - parents=True: 父目录不存在时一并递归创建
        - exist_ok=True: 目录已存在时不报错
    '''
    out_dir.mkdir(parents=True, exist_ok=True)
    '''
    计算并行进程数 n_proc:
        os.cpu_count()     取逻辑 CPU 核数(4 核 8 线程的机器返回 8);
                           "or 4" 兜底:极少数环境会返回 None
        (… - 2)            留 2 个核算力给系统与本进程,避免机器被跑满卡死
        max(…, 1)          至少 1 个,防止单核机器算出 0 或负数
        调用方显式传了 processes 则用调用方的值(or 短路: 非 0/None 时直接取它)
    '''
    n_proc = processes or max((os.cpu_count() or 4) - 2, 1)
    logger.info(f"      并行进程  : {n_proc}")

    # 总统计容器: 各子进程只返回自己那片的计数,这里做累加;
    # failed 收集"处理失败的分片路径",跑完统一报告
    grand = {'total': 0, 'kept': 0, 'dropped': 0, 'badlines': 0, 'failed': []}

    '''
    multiprocessing.Pool(processes, initializer, initargs)
        - processes  : 工作进程数
        - initializer: 每个子进程【启动时】调用一次的函数(此处 _init_worker)
        - initargs   : 传给 initializer 的参数元组(须可 pickle,Path 与 bool 都可以)
        作用: 在子进程里把 _OUTPUT_DIR / _FORCE 设成主进程指定的值,
              并预热 opencc 转换器 —— 修掉 Windows spawn 下
              "模块级全局变量传不进子进程"的隐患(详见 _init_worker 说明)。

    pool.imap_unordered(func, iterable, chunksize)
        - 与 map 的区别: 不保证返回顺序,谁先干完先返回 -> 进度实时滚动
        - chunksize=1: 一次派一个分片,进程间负载均衡最好
        - 迭代产出的是 worker 函数 return 的 dict
    '''
    with Pool(processes=n_proc, initializer=_init_worker,
              initargs=(out_dir, force)) as pool:
        # tasks 形如 [(分片路径, 输出文件名), ...]
        # ★ with 语句: 出这个代码块时会自动调用 pool.close() + pool.join(),
        #   保证所有子进程都干完活并回收,不会留下僵尸进程。
        for s in pool.imap_unordered(_clean_one_shard, tasks, chunksize=1):
            # s 就是 _clean_one_shard 返回的统计 dict,三种情况分别处理:
            if s['status'] == 'skip':        # 输出已存在(断点续跑),不计入统计
                continue
            if s['status'] == 'failed':      # 堆栈已在子进程内 logger.error 打过
                grand['failed'].append(s['file'])
                continue
            # 正常完成: 把本分片的计数累加进总统计
            for k in ('total', 'kept', 'dropped', 'badlines'):
                grand[k] += s[k]             # 累加各分片统计

    rate = grand['dropped'] / max(grand['total'], 1)   # max 防除零
    # 汇总日志: 总计 / 保留 / 丢弃 / None率 / 坏行 / 总耗时
    #   {rate:.1%} 表示百分比保留 1 位小数,如 0.1456 -> "14.6%"
    logger.info('-' * 70)
    logger.info(f"      合计 {grand['total']} 篇 | 留 {grand['kept']} | "
                f"弃 {grand['dropped']} (None率 {rate:.1%}) | 坏行 {grand['badlines']}"
                f" | 耗时 {_human_time(time.perf_counter() - t_start)} ({n_proc} 进程)")
    if grand['failed']:
        # Path(f).name 只取文件名(去掉长长的绝对路径),日志更短
        logger.error(f"      失败 {len(grand['failed'])} 片: "
                     f"{[Path(f).name for f in grand['failed']]},"
                     f"修复后删除对应输出文件可断点续跑")
    if rate > 0.15:
        # 质量检查点: None 率异常升高(>15%)通常意味着上游 wikiextractor 抽取出了问题
        # (如 dump 不完整、--json 版本差异),此时应抽样人工复核,而不是直接入库
        logger.warning("      None 率超 15%,建议抽查 wikiextractor 抽取质量")
    logger.info('-' * 70)
    return grand




def to_jsonl(shards: 'list[Path] | Path', out: Path) -> int:
    r'''
    把清洗好的全部分片,按固定顺序【流式】合并成单个 JSONL 文件(一行一篇)。

    param shards: 分片来源,两种写法都支持 ——
                    · list[Path]: 分片文件列表(架构规格里的原始签名);
                    · Path      : 清洗输出目录(如 data/articles/),
                                  函数内部自动按 *.jsonl 展开并排序。
                  ★ 为什么允许直接传目录: 调用方少写一行 glob,更重要的是
                    避免各处自己 glob 时忘记 sorted(顺序变了产物就不可复现)、
                    忘记排除输出文件自身(重跑会把上次的产物又合进来)。
    param out:    合并产物的路径,如 data/articles/wiki_zh.jsonl。
                  ★ 允许与分片同目录: 函数内部会把 out 自己从输入里剔除。
                  推荐落点 data/articles/wiki_zh.jsonl(与清洗分片同目录)。
                  注意架构文档写的是 data/cleaned/wiki_zh.jsonl,但当前
                  data/cleaned/ 被 wikiextractor 的分片占用了(见文件头),
                  口径需另行统一,故本函数不把路径写死、由调用方传入。
    return: 写入的文章总数(int)。应与 clean_shards() 汇总的 kept 相等
            (当前语料 = 1,362,848),调用方可用它做一次交叉校验。

    ★ 执行流程(4 步, 与函数体内的 [1]~[4] 分段注释一一对应):
        [1] 归一化入参: 传目录则展开成 *.jsonl 文件列表并排序;
                        传列表则原样使用(顺序由调用方负责)
        [2] 准备输出: 建父目录, 并求出 out 的绝对路径(留着判断"谁是自己")
        [3] 先筛分片: 去掉"文件不存在"与"就是输出自己"的, 并确认还有东西可合。
            ★ 这一步必须【在打开输出之前】做完(原因见下面第 3 条细节)
        [4] 逐分片逐行搬运: 攒够 _MERGE_BATCH 行写一次盘, 收尾时补上换行

    ★ 与二次清洗的边界: 本函数【不做任何内容处理】。
      _clean_one_shard 写出的每一行已经是
          json.dumps({id, title, url, n_chars, text}, ensure_ascii=False)
      的成品行,这里只做"搬运 + 补换行",连 json.loads/dumps 都不需要
      (省掉 136 万次解析与序列化)。
      ★ 千万不要在这里再调 _clean_article: opencc 自身非幂等(已简体的
        文本再转还会变),重洗会平白引入变更,还要多烧十几分钟。

    ★ 三个必须小心的细节:
      1) 分片文件是 write_text('\n'.join(records)) 写出来的,【末行没有换行符】。
         所以不能直接 fout.write(line) —— 那样后一分片的首行会和前一分片的
         末行粘成一行: 静默少一行,而且那一行不是合法 JSON。必须逐行补 '\n'。
      2) 全程逐行读写,不把任何分片整载进内存: 语料 2.32 GB,整载后算上
         str 对象开销要 8~10 GB。内存占用只与"单行长度"和攒批大小有关。
      3) 【先筛后开】: out.open('w') 一执行, 输出文件就被立刻截断成空。
         若把"筛掉不存在的分片 / 排除输出自身"放在打开之后做, 一旦筛完发现
         一个可用分片都没有(例: 分片被删过、目录里只剩上一次的产物),
         就会"先清空输出、再发现没内容可写", 静默丢掉几 GB 的产物。
         因此第 [3] 步必须提前, 并且筛空时直接抛 FileNotFoundError,
         让输出文件保持原样 —— 宁可报错, 不要静默毁数据。

    为什么单进程、不开 Pool:
        合并是纯顺序 IO,并行写同一个文件必须加锁,反而更慢;
        真正的重活在二次清洗阶段(已多进程跑完),这里只需顺序搬运。
    '''
    # ---- 1. 归一化输入: 传目录就按 *.jsonl 展开(排序保证结果可复现) ----
    # isinstance(x, (str, Path)) 成立 => 调用方传的是"一个路径"(目录),
    #   而不是"路径列表", 由本函数负责展开; 否则按文件列表原样使用。
    if isinstance(shards, (str, Path)):
        src = Path(shards)
        # 显式校验: 传进来不存在/不是目录时立刻报错, 胜过静默产出一个空文件
        if not src.is_dir():
            raise NotADirectoryError(f"to_jsonl 收到的不是存在的目录: {src}")
        # glob('*.jsonl'): 只取本层的 .jsonl(清洗产物是扁平的, 不递归子目录);
        #   is_file() 排除名字形如 xxx.jsonl 的目录;
        #   sorted() 物化生成器并排序 —— 顺序决定产物的行序, 必须稳定,
        #   否则每次跑出来的文件都不一样(build_subset 取前 N 篇会漂移)。
        shard_list = sorted(p for p in src.glob('*.jsonl') if p.is_file())
    else:
        shard_list = list(shards)
    if not shard_list:
        # 一个分片都没有: 多半是二次清洗还没跑, 或目录传错了
        raise FileNotFoundError(
            f"to_jsonl 没有拿到任何分片: {shards}\n"
            f"  请先运行 clean_shards() 生成清洗分片。")

    # ---- 2. 准备输出 ----
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)   # 父目录不存在就建(如 data/articles)
    # resolve() 标准化成绝对路径, 便于可靠判断"哪个文件是输出自己"
    out_self = out.resolve()

    # ---- 3. 先筛出真正要合的分片, 并且【赶在打开输出之前】确认还有东西可合 ----
    # ★ 为什么必须提前筛: out.open('w') 一旦执行, 输出文件就被【立刻截断成空】。
    #   若等打开之后再筛、筛完发现一个可用分片都没有(典型场景: 分片被删过、
    #   目录里只剩上一次的产物), 就会"先清空输出、再发现无内容可写",
    #   静默丢掉几 GB 的产物。所以宁可在这里提前抛错, 让输出文件保持原样。
    missing = [p for p in shard_list if not p.is_file()]
    for p in missing:
        logger.warning(f"      跳过不存在的分片: {p}")
    # 排除输出文件自身: 产物就落在分片目录里时, 不排除的话重跑会把上次的产物
    #   也当成输入再合一遍(第一次 136 万行 → 第二次 273 万行, 越滚越大)。
    usable = [p for p in shard_list
              if p.is_file() and p.resolve() != out_self]
    if len(usable) < len(shard_list) - len(missing):
        logger.info(f"      跳过输出文件自身: {out.name}")
    if not usable:
        raise FileNotFoundError(
            f"to_jsonl 没有任何可用的分片可合并: {out}\n"
            f"  候选 {len(shard_list)} 个, 其中不存在 {len(missing)} 个, "
            f"其余均为输出文件自身。")

    # ---- 4. 逐分片、逐行搬运(唯一真正的循环) ----
    total = 0
    t0 = time.perf_counter()
    # newline='\n': 文本写入时不把 '\n' 自动翻译成 Windows 的 '\r\n',
    #               保证产物跨平台逐字节一致(下游按 \n 切行)
    with out.open('w', encoding='utf-8', newline='\n') as fout:
        buf: list[str] = []           # 攒批缓冲(见 _MERGE_BATCH 说明)
        for shard in usable:
            # with 保证分片句柄及时关闭(4677 个文件, 靠引用计数也得及时释放)
            with shard.open(encoding='utf-8') as fin:
                # 直接迭代文件对象 = 逐行读取(惰性,不整载);
                # 除了最后一行, 每行都自带结尾 '\n' —— 这正是要 rstrip 掉的原因
                for line in fin:
                    line = line.rstrip('\r\n')
                    if not line:      # 分片构造上不会出现空行, 此处仅防御
                        continue
                    buf.append(line)
                    total += 1
                    if len(buf) >= _MERGE_BATCH:
                        fout.write('\n'.join(buf))
                        fout.write('\n')   # ★ 补换行: 防相邻分片首尾粘成一行
                        buf.clear()
            logger.info(f"      {shard.name}: 累计已合并 {total} 篇")
        if buf:                       # 收尾: 写出不足一批的尾巴
            fout.write('\n'.join(buf))
            fout.write('\n')          # ★ 让产物以换行结尾(规范 JSONL 语义)
    # 汇总日志: 片数用 usable(真正参与合并的数量), 不是 shard_list(候选数量)——
    #   两者在"目录里还有上一次的产物/有文件缺失"时会差 1 个或几个。
    skipped = len(shard_list) - len(usable)
    logger.info(f"      合并完成: {len(usable)} 片 → {out.name} | "
                f"{total} 篇 | {_human_size(out.stat().st_size)} | "
                f"耗时 {_human_time(time.perf_counter() - t0)}"
                + (f" | 已跳过 {skipped} 个" if skipped else ""))
    return total

if __name__ == '__main__':
    # ★ 这个判断是 Windows 多进程的【硬性要求】:
    #   Windows 用 spawn 创建子进程时,子进程会重新执行本模块;
    #   没有这层保护,子进程会再次执行下面的 clean_shards,又建一层 Pool,
    #   直接抛 "An attempt has been made to start a new process before
    #   the current process has finished its bootstrapping phase"。
    cleaned_dir = Path(r'E:/python/RAG/rag-from-scratch/data/cleaned')  # 分片根目录
    # 是否覆盖重跑: 命令行带 --force 就是全量重跑, 不带就是断点续跑。
    #   python -m offline.parse            # 接着跑: 已存在的输出文件全部跳过
    #   python -m offline.parse --force    # 全量重跑: 覆盖重写(改了清洗规则/开关繁简统一后用)
    # sys.argv 是命令行参数列表(第 0 个是脚本名), in 判断它里面有没有 '--force'
    force = '--force' in sys.argv
    grand = clean_shards(cleaned_dir, force=force)  # 内部自动递归 AA/AB/... 下所有分片
    logger.info(f"二次清洗完成: 保留 {grand['kept']} 篇 / 丢弃 {grand['dropped']} 篇")

    # ---- 可选第 3 步: 把所有清洗分片合并成单个 wiki_zh.jsonl(供 D2 切片) ----
    # 为什么默认不自动跑: 清洗与合并可以分开重跑 —— 只改清洗规则时重跑清洗,
    #   合并再单独执行一次即可; 且产物约 2.3 GB, 不该在每次断点续跑时都重写。
    # 用法(取消下面几行注释即可):
    #   n = to_jsonl(_OUTPUT_DIR, _OUTPUT_DIR / 'wiki_zh.jsonl')
    #   # ★ 注意: grand['kept'] 只是【本次运行】的保留数, 断点续跑时被跳过的分片
    #   #   不计入(见 _clean_one_shard 的 'skip' 分支), 所以不带 --force 时它是 0,
    #   #   这行断言只在 --force 全量重跑之后才成立。
    #   assert n == grand['kept'], f"合并篇数 {n} 与本次清洗保留数 {grand['kept']} 不一致"


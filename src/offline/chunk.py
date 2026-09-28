# -*- coding: utf-8 -*-
""" chunk.py:切片(检索质量的命门)

职责:
    1. 句子流 -> 滑窗封块(450~550 弹性区间,段落优选断点,超长句硬切)
    2. 相邻块重叠 50~100 字(精确 50 / 句对齐动态分支)
    3. 每块生成全局唯一 chunk_id(f"wiki{doc_id}_c{chunk_index:03d}")
    4. 输出 data/chunks/wiki_chunks.jsonl(一行一块)
    5. build_subset:只处理前 N 万篇 -> chunks_subset.jsonl(D3~D6 调试用)

对应:路线计划 D2 | 架构 §3.3 | 开发指引 D2切片开发指引.md(2026-09-20 定稿)
运行: python -m offline.chunk          (全量)
     python -m offline.chunk --subset (前 10 万篇调试子集)
"""
import re
import json                         #把每块字典json.dumps生成一行，产物是JSON
import logging                      # 日志
import sys                          # 新增main
import time                         # 计时
from pathlib import Path            # 路径
from common import config
from common.utils import iter_jsonl  # 流式逐行读jsonl
# ======================================================================
# 正则清单(D2,共 1 条)
# ======================================================================
# R1 句子切分定稿:按全角句号/叹号/问号切句,标点跟随前句。
# ★★ 2026-09-27 修过一个静默丢字 bug,改正则前务必先看这段说明 ★★
#
# 【旧写法(有 bug,已废弃)】: r'[^。！？]+[。！？]?'
#   它要求一句里【至少有一个非标点字符】,于是"以标点开头"的文本匹配不上:
#     findall('。这种潮流最早...') -> ['这种潮流最早...']   ← 句首的 '。' 被吞掉
#   实测前 300 篇:41 篇(13.7%)受影响、共丢 63 字,
#   直接导致 V5 无损性验证(各块去掉重叠后拼回 == 原文)失败 41 篇。
#   成因:原文里存在"段首就是句号"的段落(维基模板/引用被 D1 清掉后留下的残句)。
#
# 【新写法(现行)】: r'[^。！？]*[。！？]|[^。！？]+$'
#   模式拆解(两条分支,findall 从左到右逐段匹配):
#   分支一 [^。！？]*[。！？]
#        * 表示"可以 0 个非标点字符" —— 这正是修 bug 的关键:
#        允许匹配"光秃秃一个句号",从而保住句首/段首的标点。
#   分支二 [^。！？]+$
#        尾部没有句号的残句(文章最后一句常无标点),用 $ 锚定兜住,保证不丢字。
#   边界实测:
#     '。这种潮流最早。'  -> ['。', '这种潮流最早。']   (旧: ['这种潮流最早。'] 丢字)
#     '。'                -> ['。']                    (旧: [] 整个段落消失)
#     '。。连续'          -> ['。', '。', '连续']       (旧: ['连续'] 丢两字)
#     '正常句子。第二句'   -> ['正常句子。', '第二句']   (与旧写法一致)
#   全量校正:前 300 篇共 21785 个段落,新正则重建后与原文逐字相等的段落 = 全部
#            (旧正则有 63 个段落不等)。
# 注意:分号、逗号不算句子边界(方案 R1 定稿口径,验收 V3 按此检查)。
_SENT_RE = re.compile(r'[^。！？]*[。！？]|[^。！？]+$')

#模块级 logger
logger = logging.getLogger(__name__)
def split_paragraphs(text: str) -> list[str]:
    """
    按空行把整篇正文切成段落列表，跨段累加模式下，碎段(章节、标题等)会被滑窗阶段自然吸收进所属块；若在这里删除，原文会缺字。

    """
    # ★★ 2026-09-27 修过一个静默丢字 bug:这里【不能】对段落做 strip() ★★
    # 旧代码是 `para = para.strip()`,它会吃掉段落首尾的空白字符。
    # 实测 2 万篇里就有这种数据:原文段落是 '连音 '(末尾带一个空格),
    # 被 strip 成 '连音' -> 该段少了 1 个字 -> V5 无损性验证(重建 == 原文)失败。
    # 本模块的第一原则是"只搬运重组、不改一个字符",
    # 所以段落一律【原样保留】,连首尾空格都不许动。
    paragraphs = []
    for para in text.split('\n\n'):
        # 只过滤"零长度"段落。它只在 4 个以上连续换行时才会出现,
        # 而 D1 已全量校验过语料里没有 3+ 连续换行,所以这个分支正常不会命中;
        # 留着它只是防御——真出现了,丢掉空串不会造成额外丢字。
        if para:
            paragraphs.append(para)
    return paragraphs

def _stream(text: str) ->list[list[str]]:
    '''
    整篇正文 ->句子流：外层是段落，内层是该段的句子列表
    为什么不拍平成一维列表:段落号(外层下标)是后续三个机制的依据——
    R2 跨段拼接时补 \n\n、R3b 段落末尾优选断点、_assemble 判断块边界
    是否需要补段落分隔。拍平了这些信息就丢了。
    句子保留原样、不做 strip:这样 len(句子) 才等于它在原文里占的字数,
    后续 V5 无损性验证(重建文本 == 原文)才能逐字成立。
    '''
    stream = []
    for para in split_paragraphs(text):
        # ★ 过滤条件用 `if s` 而不是 `if s.strip()`:
        #   strip() 会把"纯空白句子"(比如一个段落就是几个空格)判成空而丢掉,
        #   那些空格也就跟着没了 —— 又一处静默丢字。
        #   新正则的两条分支都要求至少 1 个字符,findall 不可能返回空串,
        #   所以 `if s` 既能兜住防御目的,又不会误杀任何内容。
        sents = [s for s in _SENT_RE.findall(para) if s]
        stream.append(sents)
    return stream

def _hardcut(para_no:int,sent:str) ->list[tuple[int,str]]:
    '''
    超长单句硬切 —— 均分成 ceil(L/500) 片,每片 <=500 字
    para_no  该句所属的段落号(硬切片不改变段落归属)
    sent     超长句子本身
    返回     [(段落号, 片), ...],仍是 Step 2 统一的数据结构

    切法:前 n-1 片每片固定 size 字(向上取整的均分),最后一片吃掉全部余数。
    数学上可证:last = L - size*(n-1) <= size <= 500,不会出现"最后一片反而超标"。
    '''
    L = len(sent)
    n = -(-L // config.CHUNK_HARDCUT_CHARS) #负号翻转技巧,-(-7//2)=4
    size = -(-L // n) # 每片基础长度(向上取整的均分)
    pieces = []
    for j in range(n):
        if j < n - 1:
            pieces.append((para_no,sent[size * j:size * (j + 1)])) # 前 n-1 片:等长
        else:
            pieces.append((para_no,sent[size * (n - 1):]))  # 最后一片:带余数
    return pieces

def _chunk_text(pieces:list[tuple[int,str]]) -> list[str]:
    '''
    块 -> 文本:同段落直连,跨段落补 \n\n(R2)

    pieces  [(段落号, 句子), ...],段落号单调不减
    返回    该块的最终文本(与 _pack 里的 cur_len 逐字一致)

    为什么"补 \n\n"放在这里而不是切句时:段落分隔符不属于任何句子,
    它只在"块内跨段"时才存在——块边界的分隔是 Step 3 _assemble 的事。
    '''
    parts = [] #按段落分组后的文本段
    buf = [] # 当前段落累积的句子
    last = None # 上一句的段落号
    for i,s in pieces:
        if last is not None and i != last: # 段落号变了 -> 上一个段落到此为止
            parts.append(''.join(buf))
            buf = []   # 开新缓冲
        buf.append(s)  # 同段直连(句子自带标点,不加任何东西)
        last = i
    parts.append(''.join(buf))   # 收尾:最后一个段落
    return '\n\n'.join(parts)  # 段与段之间补回空行,还原 D1 保留的边界

def _pack(stream:list[list[str]]) ->list[list[tuple[int,str]]]:
    '''
    核心滑窗封块(R3a/R3b/R3c/R4/R6)，参数说明：
    stream : 段落列表(_stream 的产物)，每段 = 该段的句子列表 list[str]；
             一维化后的 pieces 才是 [(段落号，句子), ...]
    返回   : 块列表，每块 = [(段落号，句子), ...] -> 文本拼装延迟到 _chunk_text()
    恒等式：任何时刻，cur_len == len(_chunk_text(cur))——
        因为跨段加句时，把'\n\n' 的两个字符也计入了
    '''
    MIN = config.CHUNK_MIN_CHARS    # 450：下限，不足必须加句(R4 无条件续句)
    MAX = config.CHUNK_MAX_CHARS    # 550：上限，放不下就封块(R3a)

    # 一维化：把二维句子流 摊平成[(段落号，句子),...]的有序序列
    # 外层下表就是段落号，摊平后循环里只需看"下一个句子"的段落号是否变了
    pieces = [(i,s) for i,para in enumerate(stream) for s in para]
    chunks = []  # 已封的块(每块是pieces子列表：一律是 list，硬切片也包 list)
    lengths = [] # 与chunks平行的块长表（避免封块后反复重拼文本计算长度）
    cur = []  # 当前正在累积的块
    cur_len = 0  # 当前块的精确文本长度(含跨段\n\n)

    def close():
        '''
        封块：把cur存入chunks，清空当前累积，内层函数用nonlocal改外部变量
        '''
        nonlocal  cur,cur_len
        if cur:  #空块不封(防close()被连续调用)
            chunks.append(cur)
            lengths.append(cur_len)
            cur = []
            cur_len = 0
    for k,(i,s) in enumerate(pieces):
        L = len(s)
        if L > MAX:
            #超长单句 -> 先封掉当前块，再硬切长句(R3c/R5)
            close()
            hs = _hardcut(i,s)
            for p in hs:                      # 每片各自成块
                chunks.append([p])            # 包一层 list，与 cur 的类型保持一致
                lengths.append(len(p[1]))     # 片内同段，块长就是文本长度
            continue

        # 其余情况一律加句——注意 cur_len < MIN 时，即使突破MAX 也加，这是 > 550 块的唯一来源(模拟占比 5.3%，白名单化，不是bug)
        sep = 2 if cur and i != cur[-1][0] else 0
        # 已达下限且下一句放不进上限 -> 封块，本句开启新块，cur为空时跳过，新块的第一句再长(> MAX)也会被硬切
        if cur and cur_len >= MIN and cur_len + sep + L >MAX:
            close()
            sep = 0
        cur.append((i,s))
        cur_len += sep + L

        #句子是该段最后一句，且当前块已达下限 -> 段落末尾优选断点，封块。即使下一句其实放的进550 依然封句，段落语义完整性优化
        para_end = (k + 1 == len(pieces)) or (pieces[k + 1][0] != i)
        if para_end and cur_len >= MIN:
            close()


        # 收尾-全部句子处理完，cur若还有剩余就是尾块
    if cur:
        # 尾块 < 100 字：并入前块(前提：合并后不超650，且前块存在)。
        # 注意合并处的段落分隔：前块末句与尾块首句若跨段，同样要 +2
        if chunks and cur_len < config.CHUNK_TAIL_MERGE:
            sep = 2 if chunks[-1][-1][0] != cur[0][0] else 0
            if lengths[-1] + sep + cur_len <= config.CHUNK_MERGE_LIMIT:
                chunks[-1].extend(cur) # 列表拼接即完成合并
                lengths[-1] += sep + cur_len
                cur = []
        close()
    return chunks

# ======================================================================
# Step 3:重叠回填(R7)+ 块文本组装
#
# 这一层只解决一件事:让相邻两块"接得上"。
# 切片把长文切成若干块,接缝处最容易丢上下文——比如某概念的定义在上一块
# 末尾、例子在下一块开头,检索只命中下一块就会看到半句话。所以规定:
# 下一块的开头,重复上一块的最后 50~100 个字。
#
# 术语先对齐(下面注释里反复出现):
#   正文(body)  : 这一块【自己的】内容(_chunk_text 的产出),不含任何重叠
#   重叠(overlap): 从【上一块】尾部抄过来的一段,放在【本块】最前面
#   成品(text)  : 重叠 + 段落分隔 + 正文,这才是最终写进 JSONL 的字符串
# ======================================================================

# 句子结束符 —— 必须与 R1/_SENT_RE 一样用【全角】标点
# ★ 写成半角 '.!?' 的后果极其隐蔽:中文正文里几乎没有半角句号,
#   下面的 rfind 会永远返回 -1,于是重叠永远退化成"按 50 字硬截",
#   句对齐分支形同虚设 —— 不报错、不崩溃,只静默拉低检索质量。
_SENT_END = '。！？'

# 合法断点字符集 = 句子结束符 + 换行符
# 换行符也算断点,因为段落边界 \n\n 本身就是天然的语义断点。
# 单独定义在模块层(而不是每次进函数才现拼),省一点开销,也让下面的循环更好读。
_BREAK_CHARS = _SENT_END + '\n'

def _at_break(text:str,pos:int) ->bool:
    '''
    判断切点 pos 是否落在 ‘合法断点’上
    参数:
        text:str  待检查的文本(_chunk_text()的产出)
        pos:int   候选切点下标，恒等于 len(text) - CHUNK_OVERLAP,也就是"从末尾往前数50个字"的位置
    返回：
        True = 可以从这里开始，重叠正好50个字
        False = 此时正好切在句子中间，需要往回退
    判断方法：看切点（左边紧挨着的那个字符），即text[pos - 1]
        (1) 它是 。！？  ->切点正好落在句尾
        (2) 它是 \\n    -> 切点在段落分隔之后
        (3) pos == 0   -> 切点就是文本开头（整块内容很少，不是句中）
        (4) 以上三种除外，都算切在句子中间
    '''
    # 先单独处理 pos == 0 的情况，如果不单独判断，text[pos - 1] 会变成 text[-1]。而Python的负下标会绕到字符串末尾取字符
    if pos == 0:
        return True
    # 取切点左边紧挨着的字符
    char_before = text[pos - 1]
    # 判断是句子结束符
    if char_before in _SENT_END:
        return True
    # 判断是换行符
    if char_before == '\n':
        return True
    #以上都不是，则返回False，表示切在句子中间
    return False

def _pick_overlap(prev:str) -> str:
    '''
    R7 重叠决策:从【上一块的成品文本】里,取出要复制到下一块开头的那一段

    参数:
        prev  上一块的【成品文本】(即 texts[-1]);它自己也可能带着重叠前缀
    返回:
        重叠字符串(极端情况下是空串 ''),将作为下一块最前面的一部分

    三条分支(按尝试顺序):
        a) 切点正好在断点上                  -> 重叠 = 精确 50 字(最理想)
        b) 切在句中,但往回 100 字内有断点   -> 退到该断点之后,重叠 50~100 字
        c) 最近的断点在 100 字之外(长句)    -> 放弃对齐,接受精确 50 字

    图示(prev 是上一块,| 是候选切点,它右边 50 字是本来想抄的部分):

        a) ……内容内容。|内容内容内容
                       ↑ 左边刚才是句号 -> 直接抄最后 50 字

        b) ……内容。内容内|容内容内容
                       ↑ 落在句中 -> 往回退到最近的句号之后,多抄一点(<=100)

        c) ……(一整句 300 多字,中间一个句号都没有)……|最后50字
                       ↑ 退无可退 -> 就地截断,重叠就是 50 字
    '''
    # ---------第0步：防御性判断------------------
    # 如果上一块本身 <= 50，此时抄它的“最后”50个字，等于把整块复制一遍，美哦与意义；更严重的是此时cut会算成负数，而Python的负下标会绕到字符串末尾取字符。
    # 直接返回 空串，即不重叠。而正常块长450~550 此时基本不会命中，留着是为了防止负下标静默出错
    if len(prev) <= config.CHUNK_OVERLAP:
        return ''

    # ---------第1步：算出候选切点------------------
    # 目标位置：从末尾往前数50个字
    cut = len(prev) - config.CHUNK_OVERLAP

    # --------第2步：分支a->判断切点是否落在断点上-----------
    if _at_break(prev,cut):
        # 从cut开始取到末尾，长度恰好50个字
        return prev[cut:]

    # --------第3步：切点在句中，找离 cut 最近的那个断点-----------
    # 在 cut 左边(不含 cut 本身)的范围内,分别找 。！？\\n 最后一次出现的位置,
    # 取其中最大的那个 = 离 cut 最近的那个断点。
    #
    # rfind(c, 0, cut) 的含义:在 prev[0:cut] 这一段里从右往左找字符 c,
    #                        找到返回下标,找不到返回 -1
    nearest_break_pos = -1
    for break_char in _BREAK_CHARS:
        pos = prev.rfind(break_char, 0, cut)
        if pos > nearest_break_pos:
            # 下标越大 -> 越靠右 -> 离 cut 越近,所以保留最大值
            nearest_break_pos = pos
        # ★ 循环里只做“找断点”这一件事。绝不能在这里顺手判断“要不要抄” ——
        #   那样会在只考察了第一个断点字符(。)时就提前 return,
        #   后面的 ！？\\n 全被跳过，找出来的根本不是“最近的”那个断点。

    # --------第4步：分支b-> 断点够近就退回去-----------
    if nearest_break_pos != -1:
        # 断点字符本身不要，从它的【下一个字符】开始抄 —— 那才是新句/新段的首字
        # ★★ 必须是 prev[start:] 而不是 prev[:start]：
        #    我们要的是 prev 的【尾部】副本。写反就变成了 prev 的“开头”，
        #    长度还常常碰巧不等于 50，是本项目最难发现的一类 bug。
        start = nearest_break_pos + 1
        # 抄下来的长度 = 从 start 一直到末尾
        overlap_len = len(prev) - start
        # 只要不超过 100，就接受这次回退(超过就不划算了)
        if overlap_len <= config.CHUNK_OVERLAP_MAX:
            return prev[start:]

    # --------第5步：分支c-> 退无可退，接受精确 50 字-----------
    # 走到这里只有两种可能:
    #   ① 压根没找到断点(nearest_break_pos == -1)
    #   ② 找到了,但离 cut 太远,退过去要抄超过 100 字(不划算)
    # 两种情况都接受"精确 50 字",哪怕切在词语中间
    return prev[cut:]

def _assemble(chunks:list[list[tuple[int,str]]]) ->list[str]:
    '''
        组装:把 _pack 的块列表变成最终的文本列表(重叠 + 段落分隔一次做完)

        参数:
            chunks  _pack 的产物,三层嵌套:
                        chunks           -> 整篇文章的所有块
                        chunks[k]        -> 第 k 块:[(段落号, 句子), ...]
                        chunks[k][j]     -> 第 k 块第 j 个句子:(段落号, 句子)
        返回:
            每块一个字符串的列表。第 0 块没有重叠;
            第 k 块 = 重叠 + 段落分隔 + 本块正文。

        拼装顺序不能反:[重叠][分隔][正文]
          重叠是从上一块【尾部】抄来的,分隔是两块【之间】的段落边界。
        '''
    # 空文章防御，之前这已经保证每篇 >= 50个字，理论上不会走得这里
    if len(chunks) == 0:
        return []
    # 第一步：先把每块自己的正文拼出来
    # _chunk_text 只负责 块内部  拼接，同段句子直连，块内跨段补 \n\n
    bodies = []
    for block in chunks:
        body = _chunk_text(block)
        bodies.append(body)
    #第二步：第一块中没有‘上一块’，结果就是它自己的正文
    texts = [bodies[0]]

    #第三步：从第二块开始，逐个块加重叠
    for k in range(1,len(chunks)):
        #先取出本快和上一块，避免下面都是chunk[k-1]这种下标
        curr_block = chunks[k]
        prev_block = chunks[k - 1]

        # 重点： 整个模块最隐蔽的坑,就在下面这四行
        # 如果块边界恰好落在段落边界上(这正是 R3b"段落末尾优选断点"的结果),
        # 那个 \n\n 就【不属于任何一块的内部】:
        #   · 上一块的 _chunk_text 拼到它的末句就结束,看不见后面的 \n\n
        #   · 本块的 _chunk_text 从它的首句开始,也看不见前面的 \n\n
        # 所以必须在这里显式补回来,依据是"交界处段落号变没变"。
        curr_first_sent = curr_block[0]     # 本块第一句：（段落号, 句子）
        prev_last_sent = prev_block[-1]     # 上一块的最后一句
        curr_first_para = curr_first_sent[0]#本块首句的段落号
        prev_last_para = prev_last_sent[0]  # 上一块末句的段落号

        if curr_first_para != prev_last_para:
            sep = '\n\n'                # 跨段 -> 补回段落分隔
        else:
            sep = ''                    # 同一个段内 -> 什么都不加

        # 不补会怎样:重建原文时上一块末尾的"生平"会和下一块开头的"恐龙"粘成"生平恐龙",而且 (无损性:重建文本 == 原文)必挂。
        # 为什么这样判断就够了:推演过, 封块后下一句必定还在同段(sep 为空),
        # 封块后下一句必定在新段(sep 为 \n\n),两种情况互斥且完备。

        # ---- 取重叠:从【上一块的成品文本】尾部抄一段 -------------------
        # 注意传的是 texts[-1](上一块组装后的成品),不是 bodies[-1](上一块正文)。
        # 两者尾部 50 字其实一样(块长 >= 450 时)
        # "上一块的尾部 == 下一块的头部"必须字符串精确相等,所以用 texts[-1]。
        overlap = _pick_overlap(texts[-1])

        # 拼装：重叠 + 分隔 + 正文
        new_text = overlap + sep + bodies[k]
        texts.append(new_text)
    return texts

# ======================================================================
# Step 4:组装层 —— sliding_window(整篇 -> 块文本) + chunk_article(块文本 -> 块字典)
#
# 这一层是"算法层"和"文件层"之间的桥:
#   _stream / _pack / _assemble  只认字符串和元组,不认识文件、不认识 id;
#   chunk_all                    只认文件行和字典,不关心块是怎么切出来的。
# 所以 Step 4 的职责就两件事:把三步算法串成一根管子 + 给每块发身份证(chunk_id)。
# ======================================================================
def sliding_window(paragraphs:str | list[str]) ->list[str]:
    '''
    把一篇文章切成若干"块文本"(已带好相邻块之间的重叠)

    参数:
        paragraphs  两种写法都收:
                    · 一整篇正文(字符串)        —— chunk_article 就是这样调的
                    · 已经切好的段落列表        —— 手搓测试用例时方便
    返回:
        字符串的列表,每个元素是一块的【最终文本】:
            第 1 块 = 它自己的内容(前面没有上一块,所以没有重叠)
            第 2 块及以后 = 【从上一块尾部抄来的一小段】+【段落之间的空行(可能没有)】+【本块自己的内容】

    内部按顺序走三步(顺序不能换):
        第一步 _stream :把正文切成段落,再把每段切成句子
        第二步 _pack   :按字数把句子装进一个个块里,决定每块到哪里结束
        第三步 _assemble:给每块前面加上从上一块抄来的重叠,并把跨段时丢掉的空行补回来

    为什么顺序不能换:
        第二步要靠第一步给出的"这句话属于第几段"来判断"能不能在这儿收块";
        第三步要靠第二步给出的块结构,才知道哪里该补空行。
        没有段落号,"跨段要保留空行"和"正好说完一段就收块"这两件事都做不了。

     本函数不做任何清洗:不删短段、不去空格、不改任何一个字符。
      输入已经是上一步处理好的正文,这里只负责"搬和拼",
      因为后面有一项验收要把各块去掉重叠后拼回去、跟原文逐字比对,动一个字符就会失败。
    '''
    #---------入参归一：段落列表先拼回整篇正文，在交给_stream统一处理
    if isinstance(paragraphs,str):
        text = paragraphs                   # 已经是整篇文章，直接用
    else:
        text = '\n\n'.join(paragraphs)      # 段落列表，整篇文章

    # 1 切段 + 切句 -> 二维句子流 [[句，句，。。。],[句，句，，，，]]
    # 外层下标 = 段落号，这是后面所有机制的唯一依据
    stream = _stream(text)

    # 2 滑窗封块 -> [[（段落号，句子）,....],....]
    # 到这里每一步还是只有正文，没有重叠，没有chunk_id
    blocks = _pack(stream)

    # 3 叠加重叠 + 补回块间段落分隔 -> [‘成品文本’,....]
    texts = _assemble(blocks)
    return texts

def chunk_article(doc:dict) ->list[dict]:
    '''
        把一篇文章变成若干"块字典"(这是本模块最终交付的东西,下一步做向量化时直接吃它)

        参数:
            doc  从输入文件里读出来的一行,是一篇文章,里面有几个字段:
                    id    —— 这篇文章的编号(注意:输入里叫 id,输出里改叫 doc_id)
                    title —— 文章标题
                    text  —— 正文(真正要被切的内容)
                    (还有 url、n_chars 等,这里用不到,不进块;
                     需要溯源时,用 doc_id 回原文件查就行)
        返回:
            字典的列表,每个字典固定 5 个字段、顺序也是固定的:
                chunk_id     这块在全库唯一的编号,形如 wiki12345_c000
                doc_id       这块属于哪篇文章,检索命中后靠它回原文
                title        文章标题(给人看、给模型引用用)
                chunk_index  这块在本篇里的序号,从 0 开始、连续递增
                text         这块的最终文本(含重叠),这才是最拿去做向量化的内容

            补充说明:
                · 很短的文章(比如只有 60 字)只会切出 1 块,内容就是全文;
                · 正常不会返回空列表(上一步已保证每篇正文至少 50 字)。

        关于编号里的 000:
            把序号写成固定 3 位(000、001、002……999)。
            好处是:按编号排序就等于按原文顺序排序,排查问题时一眼能看出前后。
            序号超过 999(超长文)时会自动变成 4 位,不会截断、不会报错。
        '''
    # 取文章编号，注意输入字段叫id,输出字段叫doc_id
    doc_id = doc.get('id')

    # 如果编号为空，报错
    if not doc_id:
        raise ValueError(
            f"文章缺少id字段，无法生产成本块编号:title={doc.get('title')!r}"
        )
    # 标题和正文有可能为空（None）,用or '' 兜成空字符串
    title = doc.get('title') or ''
    text = doc.get('text') or ''

    #切块
    texts = sliding_window(text)

    #给每一块定义编号->唯一标识
    chunks: list[dict]=[]
    for chunk_index,chunk_text in enumerate(texts):
        chunks.append({
            'chunk_id':f'wiki{doc_id}_c{chunk_index:03d}',    # 全库唯一编号
            'doc_id':doc_id,                                  # 回原文用的钩子
            'title':title,                                    # 溯源时展示
            'chunk_index':chunk_index,                        # 本篇内存号，从0连续
            'text':chunk_text                                 # 最终文本（含重叠）
        })
    return chunks

# ======================================================================
# 主循环:把上百万篇文章切成上百万块,边读边写进 data/chunks/
#
# 三条必须守住的事(都是上一个模块踩过的坑):
#   1. 必须一篇一篇读、一块一块写,不能把整个文件读进内存(文件有两个多 G);
#   2. 写文件时强制用 \n 换行:Windows 默认会写成 \r\n,产物后面还要进索引流程,不能混;
#   3. 这里不做任何清洗,输入已经是成品,一个字符都不许改。
# ======================================================================
_WRITE_BATCH = 10000
'''
攒够 1 万行才真正写一次硬盘。

为什么不写一行存一行:全量大约 250 万块,一行一写就是 250 万次函数调用;
攒批之后只写 250 次左右,快很多(上一个模块实测:19 秒合并完 2.3GB)。

注意它和下面的 _HEARTBEAT 不是一回事:
    这个是"写盘节奏",那个是"打日志的节奏"。
另外:这个数字是工程上的习惯值,不是算法参数;
      凡是跟切块算法有关的数字(450、550、50、100)都统一放在 config.py 里,
      只有这种跟算法无关的工程常量才写在这里。
'''
_HEARTBEAT = 50000
'''
每处理满 5 万【篇】文章,打一条进度日志。

为什么要打:全量预计要跑 15~40 分钟,屏幕上一直没动静的话,
你分不清是"还在跑"还是"卡死了"。全量下来大约打 27 条,不至于刷屏。
'''

def chunk_all(in_path=config.ARTICLES_FILE,
              out_path=config.CHUNKS_FILE,
              max_docs:int | None = None) ->dict:
    '''
        切片主循环:输入文件一行一篇文章,输出文件一行一块

        参数:
            in_path   输入文件(默认 data/articles/wiki_zh.jsonl,上一步的产物)
            out_path  输出文件(默认 data/chunks/wiki_chunks.jsonl)
            max_docs  最多处理多少篇:
                        None  = 不限,全部处理
                        整数  = 处理够这个数就停,后面的行不再读
                      下面那个"只切一小部分"的功能,就是靠这个参数实现的,
                      这样全量和子集共用同一个循环,不用写两套代码。
        返回:
            一个统计字典,共 9 项:
                docs            实际处理了多少篇文章
                chunks          一共切出多少块
                max_len         最长的一块有多少字(含重叠)
                over_max        超过 550 字的块有多少(含重叠)
                elapsed_sec     总共花了多少秒
                chunks_per_doc  平均每篇切出几块(量级大约是 1.9)
                in              输入文件路径
                out             输出文件路径
                out_bytes       产物文件有多少字节
        '''
    #先检查，再打开文件 -> 顺序很重要。
    if not in_path.is_file():
        raise FileNotFoundError(
            f'切片输入不存在:{in_path} \n 请先运行文件清洗模块:parse.py'
        )
    if max_docs is not None and max_docs <= 0:
        raise ValueError(f'max_docs 必须是正整数或None,收到{max_docs}')

    #data/chunks 这个目录可能不存在，先创建 parents = True，表示父目录也一并创建
    out_path.parent.mkdir(parents=True,exist_ok=True)

    #计数器
    t0 = time.perf_counter()  # 计时用,这个时钟不受系统时间调整影响
    n_docs = 0  # 已处理文章数
    n_chunks = 0  # 已产出的块数
    max_len = 0  # 最长块的字数
    over_max = 0  # 超过 550 字的块数
    buf: list[str] = []  # 攒批用的缓冲区(见 _WRITE_BATCH)

    # ---- 主循环 ----
    # iter_jsonl 是"生成器":调用它只是拿到一个读取器,此时一行都还没读;
    # 真正读是在下面 for 循环里,读一行处理一行,内存里始终只有当前这一篇。
    gen = iter_jsonl(in_path)
    try:
        # newline = '\n':关掉Windows 自动把\n 变成 \r\n,保证产物换行统一
        with out_path.open('w',encoding='utf-8',newline='\n') as fout:
            for doc in gen:
                #一篇文章 ->若干块字典
                blocks = chunk_article(doc)
                for block in blocks:
                    # 量长度量的是"最终写进文件的文本"(它是带重叠的)。
                    # 提醒:这个数字会比算法阶段统计的"正文长度"大 50~100 字,
                    # 因为每块前面都抄了上一块的尾巴。两者别混着比。
                    L = len(block['text'])
                    if L > max_len:
                        max_len = L
                    if L > config.CHUNK_MAX_CHARS:
                        over_max += 1
                    buf.append(json.dumps(block,ensure_ascii=False))
                    n_chunks += 1
                n_docs += 1

                # 攒够一批就写入硬盘。join 只在行与行之间补换行,
                # 所以最后要额外补一个换行,否则下一批的第一行会粘上来。
                # ★★ 必须是反斜杠 '\n'(换行符)。写成 '/n'(正斜杠)它只是一个普通字符串,
                #    整批内容会被粘成一行、JSON 全部非法 —— 而且只在攒满一批时才会暴露,
                #    块数不足 10000 时走的是下面"收尾"那行,小样本根本测不出来。
                if len(buf) >= _WRITE_BATCH:
                    fout.write('\n'.join(buf))
                    fout.write('\n')
                    buf.clear()            # 清空但复用同一个列表,节省一次重新分配
                #进度日志
                if n_docs % _HEARTBEAT == 0:
                    logger.info(
                        f'      已处理 {n_docs} 篇 / {n_chunks} 块 | '
                        f'最长 {max_len} 字 | 耗时 {time.perf_counter() - t0:.0f}s'
                    )
                # 限量模式:处理够指定篇数就收工
                if max_docs is not None and n_docs >= max_docs:
                    logger.info(f'      已达上限 {max_docs} 篇,提前结束(子集模式)')
                    break
            #收尾：把最后不足一批的尾巴写掉(漏了就会丢掉最多 9999 块)
            if buf:
                fout.write('\n'.join(buf))
                fout.write('\n')
    finally:
        # 主动关掉读取器:它内部打开着输入文件。
        # 上面如果因为达到上限而提前 break,它还没来得及自己收尾,
        # 显式 close() 才会触发它内部的清理、把文件句柄放掉。
        # (不关的话垃圾回收最终也会关,但那属于"碰运气",不该依赖。)
        gen.close()
    # 汇总
    elapsed = time.perf_counter() - t0
    stats = {
        'docs':n_docs,
        'chunks':n_chunks,
        'max_len':max_len,
        'over_max':over_max,
        'elapsed_sec':round(elapsed,1),
        'chunks_per_doc':round(n_chunks / n_docs,2) if n_docs else 0.0,
        'in':str(in_path),
        'out':str(out_path),
        'out_bytes':out_path.stat().st_size,
    }
    logger.info(
        f'      切片完成: {n_docs} 篇 -> {n_chunks} 块 | '
        f'最长 {max_len} 字 | >{config.CHUNK_MAX_CHARS} 字 {over_max} 块 | '
        f'{stats["out_bytes"] / 1024 / 1024:.1f} MiB | 耗时 {elapsed:.1f}s'
    )
    return stats

def build_subset(n_docs:int = 100_000) ->dict:
    '''
        只切前 N 篇,写成一个小的调试用文件 data/chunks/wiki_chunks_subset.jsonl

        参数:
            n_docs  取前多少篇(默认 10 万)

        为什么需要它:
            做后面几步时,每调一次都要跑一遍完整流程。
            用它省下的其实不是"切块的时间"(全量也就十几到几十分钟),
            而是省掉后面那些动辄几小时的重活。
            它每次取的都是"前 N 篇",不随机抽样,所以结果稳定,
            在小文件上试好的参数,换成全量文件照样成立,不会跑偏。

        实现方式:
            就是给主循环传一个"最多处理多少篇"的上限,不另写一套循环。
            写成两套循环的后果是:改好了全量的 bug,忘了改子集那份,而且这种错极难发现。

        返回:主循环那个统计字典,原样返回(不是返回空,方便接着看日志或做断言)。
        '''
    return chunk_all(
        in_path=config.ARTICLES_FILE,    #输入和全量用的是同一个文件
        out_path=config.CHUNKS_SUBSET_FILE,  # 只有输出不同，不会修改全量的产物
        max_docs=n_docs,
    )
def _read_subset_n(argv) -> int | None:
    '''
    从命令行参数里读出"这次最多处理多少篇"

    四种用法:
        不带 --subset      -> 返回 None,表示全量,不限制篇数
        --subset           -> 返回 100000(默认取前 10 万篇)
        --subset 5000      -> 返回 5000(只取前 5000 篇)
        --subset=5000      -> 同上,等号写法

    参数:
        argv  命令行参数列表,就是 sys.argv(第 0 个是脚本名,后面才是我们传的参数)
    '''
    # 先处理等号写法(--subset=5000)。
    # ★ 为什么必须先判等号:下面的 '--subset' not in argv 用的是【完全相等】判断,
    #   '--subset=5000' 并不等于 '--subset',漏了这一步就会被当成"没写 --subset"
    #   而返回 None —— 你以为只切 5000 篇,实际跑起了 40 分钟的全量。
    for arg in argv:
        if arg.startswith('--subset='):
            tail = arg.split('=', 1)[1]      # 取等号右边的部分
            return int(tail) if tail.isdigit() else 100_000

    # 命令行里压根没写 --subset -> 全量跑
    if '--subset' not in argv:
        return None

    # 找到 --subset 在参数列表里的位置
    pos = argv.index('--subset')

    # 看看它后面还跟着一个参数、而且那个参数是纯数字 -> 就用这个数字当上限
    # 例:['offline.chunk', '--subset', '5000'] -> 取到 '5000'
    if pos + 1 < len(argv) and argv[pos + 1].isdigit():
        return int(argv[pos + 1])

    # 只写了 --subset 没写数字 -> 用默认的 10 万篇
    return 100_000
# ======================================================================
# 命令行入口:直接运行本文件时才执行
# ======================================================================
if __name__ == '__main__':
    # 日志配置只放在这里(原因见文件顶部 logger 的注释)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 先问命令行:这次要不要限量
    n = _read_subset_n(sys.argv)

    if n is None:
        # 没写 --subset -> 全量,输出写到正式产物文件
        logger.info('[full] 全量切片 -> wiki_chunks.jsonl')
        stats = chunk_all()
    else:
        # 写了 --subset(可带数字)-> 只切前 n 篇,输出写到单独的调试文件,
        # 不会碰全量的产物文件
        logger.info(f'[subset] 只切前 {n} 篇 -> wiki_chunks_subset.jsonl')
        stats = build_subset(n_docs=n)

    # 把结果打到屏幕上
    print('=' * 64)
    print(f"输入    : {stats['in']}")
    print(f"输出    : {stats['out']}  ({stats['out_bytes'] / 1024 / 1024:.1f} MiB)")
    print(f"文章/块 : {stats['docs']} 篇 -> {stats['chunks']} 块 "
          f"(平均 {stats['chunks_per_doc']} 块/篇,期望量级约 1.9)")
    print(f"最长块  : {stats['max_len']} 字  (含重叠)")
    if stats['chunks']:
        ratio = stats['over_max'] / stats['chunks'] * 100
        print(f"超 550 字: {stats['over_max']} 块  (占 {ratio:.1f}%,含重叠)")
        print("  ↑ 这个比例会比算法阶段的统计明显偏高,是因为每块都抄了上一块的尾巴;")
        print("    想跟算法阶段对数字,要先在验收脚本里把重叠那一段扣掉再量。")
    else:
        print("超 550 字: 0 块(没有产出)")
    print(f"耗时    : {stats['elapsed_sec']} 秒")
    print('=' * 64)




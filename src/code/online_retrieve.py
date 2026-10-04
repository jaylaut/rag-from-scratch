# -*- coding: utf-8 -*-
"""D6 - retrieve.py:向量检索(在线流水线第②步)—— 完整版【参考副本】

⚠️ 本文件是 src/code/ 下的参考副本:import 按 src/ 目录结构书写,直接运行会 ImportError;
   要运行请放回 src/online/retrieve.py(见 src/code/README.md 对照表)。

D6 相对 D5 的三个升级点(签名全部不变,D5 指引 §12 E3 的裁决兑现):
    1. meta 加载拆成独立单例 _get_meta() —— 索引与 meta 分开缓存、分开成对;
    2. search 返回的 dict 收紧为固定四键 {score, chunk_id, title, text}(顺序固定);
    3. 空 hits 一律返回 [](下游 rerank / build_prompt 只判 falsy,不判 None)。
(D5 最小版差异:meta 与索引一起缓存、返回 **meta 展开、无 _norm_full 归一小函数。)

索引选择策略沿用 D5 §5.2 口径(默认 subset,参数/环境变量 RAG_INDEX_FULL 切全量),D6 不另立规矩。

对应文档:D6在线链路开发指引.md §5.1 / §5.2 | 系统架构说明.md §4 步骤②
"""
import os          # 标准库:读环境变量 RAG_INDEX_FULL(切全量索引的开关)
import time        # 标准库:计时,打"加载耗时"日志用
import json        # 标准库:JSON 解析(_get_meta 里逐行 json.loads 用)
import logging     # 标准库:日志;不用 print,因为跑批/服务时日志级别可控

import numpy as np  # 第三方:查询向量的 dtype(float32)与形状(1, dim)整备
import faiss        # 第三方:Facebook 向量检索库;提供 read_index(读索引文件)与 Index.search(检索)

from common import config  # 项目内:路径与参数的单一事实来源,本模块不写死任何路径

# 模块级 logger:命名继承本模块路径(如 online.retrieve),方便日志过滤
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 模块级单例缓存:键 = full 布尔值(False=调试子集,True=全量)
# 为什么用 dict 而不是单个变量:两套产物必须能并存切换,且 INDEX_* 与 META_*
# 必须"成对切换"(D7 §7.6:错配 = 回填全错,faiss 还不报错 —— 静默错配最危险)
# ----------------------------------------------------------------------------
_INDEX_CACHE: dict = {}  # {full: faiss.Index 实例} —— 索引对象,加载一次全程复用
_META_CACHE: dict = {}   # {full: list[dict]}      —— meta 全部行,list 下标 == FAISS 行号


def _norm_full(full: bool | None) -> bool:
    """把三态参数(full=True/False/None)归一成确定的布尔值。

    功能实现原理:
        full=None 表示"调用方没表态",此时看环境变量 RAG_INDEX_FULL;
        提取成独立小函数,让 load_index 与 search 共用同一套判断,
        保证"索引与 meta 用的是同一个模式"(成对切换,防静默错配)。

    参数:
        full: bool | None —— True=强制全量;False=强制子集;None=读环境变量
    返回:bool —— 归一后的模式值
    """
    if full is not None:          # 调用方显式给了值,直接采纳
        return full
    # os.environ.get:环境变量不存在返回 "",.lower() 统一小写后与三个真值比较
    return os.environ.get("RAG_INDEX_FULL", "").lower() in ("1", "true", "yes")


def load_index(full: bool | None = None):
    """加载 FAISS 索引并缓存(进程内单例)。

    功能实现原理:
        faiss.read_index 每次都从磁盘反序列化整个索引(20 万块约 1~2 秒),
        而索引对象可以无限次复用 —— 所以"首次加载 + 模块级 dict 缓存",
        之后同模式调用直接命中缓存返回同一对象(单测 T2 靠"同一对象"断言)。
        写法与 embed.py 的 _get_session(懒加载单例)同一套路,项目内统一叫"懒加载"。

    调用的外部方法:
        faiss.read_index(path_str):
            把 faiss.write_index 写出的二进制文件反序列化成内存中的索引对象;
            参数必须是字符串路径(所以下面要 str(index_path))。

    参数:
        full: bool | None —— True=强制全量索引;False=强制子集;None=看环境变量
    返回:faiss.Index —— 可直接调 .search() 的索引对象(缓存命中时与上次是同一个对象)

    明确不做:
        不校验"索引行数 == meta 行数" —— 那是 D4 sanity_check 的职责(D6 §5.1),
        在线阶段信任离线产物;"自检未通过,索引不可信,不要往下走"(D4 指引原话)。
    """
    full = _norm_full(full)                 # 先归一,后续缓存键/路径选择都用这个值
    if full in _INDEX_CACHE:                # 缓存命中:第二次起耗时 0
        return _INDEX_CACHE[full]

    # ---- 按模式选路径(成对原则:索引与 meta 必须同一档)----
    index_path = config.INDEX_FILE if full else config.INDEX_SUBSET_FILE
    # config.INDEX_FILE        = index/wiki.faiss(全量,D4 产物)
    # config.INDEX_SUBSET_FILE = index/wiki_subset.faiss(调试子集,D4 产物)

    if not index_path.is_file():            # 文件缺失立即报错,信息里带修复命令
        # Path.is_file():判断路径存在且是文件;FileNotFoundError 信息里写清补救路径
        raise FileNotFoundError(
            f"FAISS 索引不存在:{index_path}。"
            f"请先运行 `python -m offline.index_build` 建索引(全量加 --full)。"
        )
    t0 = time.time()                             # 记录起始时间戳(打"加载耗时"日志)
    index = faiss.read_index(str(index_path))    # 反序列化;参数要求 str,不能直接传 Path
    logger.info("已加载 FAISS 索引:%s(ntotal=%d,耗时 %.2fs)",
                index_path, index.ntotal, time.time() - t0)  # ntotal=索引里向量总数
    _INDEX_CACHE[full] = index              # 写缓存,下次同模式调用直接复用
    return index


def _get_meta(full: bool) -> list[dict]:
    """加载 meta 文件为行列表(独立单例,D6 升级点 1)。

    功能实现原理:
        meta 是独立 JSONL(D4 产物,"第 i 行 ↔ FAISS 第 i 行"是契约);
        整文件一次读入 list 后,按 faiss_row 回填就是 O(1) 下标访问。
        M 档(1~5 万行)整文件几十 MB,可接受;L 档 20 万行如内存紧张
        再改按行 mmap(留注释即可,现在不过度设计 —— D6 §5.1 原话)。

    调用的外部方法:
        open(path, encoding="utf-8") 逐行 for:流式读,内存里只留解析结果;
        json.loads(line):一行 JSON → dict;坏行抛 JSONDecodeError 刻意不捕获
        (离线产物坏了应该"炸得早、炸得响",离线铁律哲学)。

    参数:
        full: bool —— 必须与 load_index 用同一个归一后的值,保证成对
    返回:list[dict] —— 全部行,顺序即 FAISS 行号(不许排序/去重/跳行)
    """
    if full in _META_CACHE:                 # 独立缓存:meta 与索引分开管
        return _META_CACHE[full]
    meta_path = config.META_FILE if full else config.META_SUBSET_FILE
    if not meta_path.is_file():             # 与索引必须同时存在,缺一个=离线产物不完整
        raise FileNotFoundError(f"元数据文件不存在:{meta_path},请检查 D4 是否跑完整。")
    # 列表推导:逐行解析非空行;list 下标天然 == 文件行号(顺序绝不能动,这是契约)
    rows = [json.loads(line) for line in open(meta_path, encoding="utf-8") if line.strip()]
    logger.info("已加载元数据:%s(共 %d 行)", meta_path, len(rows))
    _META_CACHE[full] = rows
    return rows


def search(query_vec: np.ndarray, k: int = config.RECALL_K) -> list[dict]:
    """top-k 检索 + 元数据回填 → [{score, chunk_id, title, text}, ...](固定四键)。

    功能实现原理:
        归一化向量 + IndexFlatIP ⇒ 检索分数直接就是余弦相似度
        (D3 契约:全库向量 L2 归一化后,内积 == 余弦);
        按 faiss_row(行号)回填,不按返回顺序 zip(faiss 不保证按 id 排序返回);
        score 统一 4 位小数 —— sources 展示稳定,D8 对比表不出现浮点尾差(D6 §5.2)。

    三大坑(全在 D5 踩过,回归测试 T1 兜底):
        ① 一维查询必须 reshape(1, -1),否则 faiss 报维度错;
        ② dtype 必须 float32,float64 会静默或报错(版本相关);
        ③ k > ntotal 时 I 数组用 -1 填充,不过滤就会拿 -1 当行号取 meta
           (轻则 Python 负下标取到最后一行,重则回填全错且不报错)。

    调用的外部方法:
        np.ascontiguousarray(x, dtype="float32"):
            ① dtype=float32 —— faiss 只认 float32;② C 连续内存布局 —— faiss 直接读缓冲区;
            已满足条件时返回原对象(零拷贝)。
        .reshape(1, -1):一维 (1024,) → 二维 (1,1024);-1 表示"该维自动推算",二维入参也安全。
        index.search(q, k):返回 (D, I) 两个 (1,k) 数组 —— D=相似度分,I=命中行号;
            ★ k > ntotal 时 I 用 -1 填充(三大坑之三)。
        zip(scores[0], ids[0]):只查了 1 条查询,取第 0 行把分数与行号一一配对。

    参数:
        query_vec: np.ndarray —— 查询向量;一维 (1024,) 或二维 (1,1024) 都能接
                  (通常来自 offline.embed.embed_query,它返回的就是 (1,1024))
        k: int —— 召回宽口径,默认 config.RECALL_K=50(与 top_m=5 的两层口径见 D6 §1)
    返回:list[dict] —— [{score, chunk_id, title, text}, ...],按相似度降序;
            检索不到返回 [](D6 升级点 3:返回空列表而非 None)

    明确不做:不做 min_score 阈值过滤(D9 工具层);不去重相邻块
              (50~100 字重叠是切片设计的一部分,去重会破坏 D2 的语义,§5.2 明令)。
    """
    full = _norm_full(None)                       # search 无参可传,统一走环境变量口径
    index = load_index(full)                      # 单例加载(命中缓存则零耗时)
    q = np.ascontiguousarray(query_vec, dtype="float32").reshape(1, -1)  # ①②坑一次解决
    scores, ids = index.search(q, k)              # D=分数 (1,k) / I=行号 (1,k)
    meta_rows = _get_meta(full)                   # 独立单例拿 meta 行
    results: list[dict] = []                      # 最终输出容器
    for score, faiss_row in zip(scores[0], ids[0]):   # 第 0 行 = 我们唯一的那条查询
        if faiss_row == -1:                       # ③坑:过滤 faiss 的"没凑够 k"填充值
            continue                              # 直接跳过,绝不拿 -1 当行号去取 meta
        meta = meta_rows[faiss_row]               # ★ 按"行号"回填(唯一可信关联)
        results.append({                          # 固定四键,顺序即契约(下游按此消费)
            "score": round(float(score), 4),      # numpy 标量 → Python float + 4 位小数
            "chunk_id": meta.get("chunk_id", ""), # 主键:贯穿全系统的 chunk_id(架构 §5)
            "title": meta.get("title", ""),       # 标题:prompt 的 [来源N] 头 + D8 判定用
            "text": meta.get("text", ""),         # 正文:直接进 prompt,一个字符都不许改
        })
    logger.debug("检索完成:k=%d,命中=%d,最高分=%s", k, len(results),
                 results[0]["score"] if results else "N/A")   # 调试日志(§5.2 日志点)
    return results

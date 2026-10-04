# -*- coding: utf-8 -*-
"""W5 - agent/tools.py:检索工具层(D9 设计骨架)【⚠️ 红线:本轮不落盘到 src/agent/】

★★ 本文件是 D9「只定接口,不写实现」的设计骨架,存放于 src/code/ 仅供阅读与日后复制。
   按项目纪律(D9Agent接入设计指引.md §10 第 5 条红线):
       ❌ 禁止把本文件直接拷到 src/agent/ 下 —— src/agent/ 的创建与实现归 W5 进阶;
       ✅ 落地时按本文件"实现步骤"6 步展开,实现与 §5 的偏差写进彼时新增的勘误节;
       ✅ 对 D6 的反向承诺:工具层落地时,底层 retrieve.search 的签名与抛错语义
          一个字符不动(工具层包装底层,不修改底层)。

与底层的最大接口差异(牢记):
    底层 retrieve.search:向量进 → list 出 → 异常直接抛(守数据一致性,离线哲学);
    工具层本文件:  文本进 → dict 出 → 【可预期失败转结构化结果】(守交互可用性)。
    两者不矛盾 —— 边界不同:离线批处理错一次要重跑几小时,在线交互工具失败是常态。

错误码表(D9 指引 §5.3;NO_HITS 不是错误 —— 检索到 0 条是正常业务结果,
        LLM 拿到 hits=[] 应走"资料不足"路径,把它错当 error 会让 LLM 去"修复"一个没故障的系统):
    INDEX_NOT_LOADED     索引文件缺失/未加载   hint:"索引不可用,请提示用户先建库"
    OLLAMA_UNAVAILABLE   embed 服务不可达       hint:"向量服务离线,请稍后重试"
    EMPTY_QUERY          query 为空             hint:"请给出具体检索词"
    (不是错误) NO_HITS  检索结果为空 → ok=True + hits=[],与 error 明确区分

三种「借 RAG」形态(§3,决定为什么工具层先于 Agent 循环):
    (a) RAG 作为工具   LLM 经 function calling 按需调 search()   → ✅ 起步形态(本骨架)
    (b) Agentic RAG    query 改写→检索→判断充分性→再检索      → 第二步(先定最大迭代数)
    (c) RAG 作为记忆   检索结果写入对话记忆跨轮复用           → 最后(只做最简版)

对应文档:D9Agent接入设计指引.md §5 | 开发路线与学习计划.md D9 节(D9.0~D9.6)
"""
from common import config  # 项目内:RECALL_K 等常量(SEARCH_DEFAULT_K 落地时再决策进不进 config)

# ---- 预留提案常量(D9 §2:本轮不改 config.py,落地时再决策进不进)----
AGENT_MAX_TOOL_ROUNDS = 3   # 形态 (b) 多跳循环的死循环保险丝(只有做形态 b 才需要;
                            # 没有它的多跳循环会把 API 账单和延迟一起拖爆,§10.2)
SEARCH_DEFAULT_K = 5        # 工具层默认返回条数(LLM 视角"给 5 条差不多",
                            # 区别于 RECALL_K=50 的召回语义;落地时确认与 top_m 的关系)
SEARCH_EXPAND_FACTOR = 4    # filters 后置过滤不足 k 时的扩召回倍数(按命中率实测调)

# ---- 错误码常量(错误码表的代码化;value 即文档里的 code 字符串)----
ERR_INDEX_NOT_LOADED = "INDEX_NOT_LOADED"       # 索引文件缺失/未加载
ERR_OLLAMA_UNAVAILABLE = "OLLAMA_UNAVAILABLE"   # embed 服务不可达
ERR_EMPTY_QUERY = "EMPTY_QUERY"                 # query 为空

# 各错误码建议给 LLM 的 hint(工具返回结构化错误时一并带给 LLM,让 LLM 会" human 地"应对)
_ERROR_HINTS = {
    ERR_INDEX_NOT_LOADED: "索引不可用,请提示用户先建库",
    ERR_OLLAMA_UNAVAILABLE: "向量服务离线,请稍后重试",
    ERR_EMPTY_QUERY: "请给出具体检索词",
}


def _error(code: str, message: str) -> dict:
    """构造统一格式的失败结果(内部小工具,保证错误结构一致)。

    功能实现原理:
        工具层的错误哲学 = "可预期失败转结构化结果,不抛异常";
        所有失败都经这个小函数打包成 {"ok": False, "error": {...}},
        LLM 拿到后能读 code(什么错)、message(细节)、hint(建议动作)。

    参数:
        code: str —— 错误码(取 _ERROR_HINTS 的键)
        message: str —— 人可读的错误细节(排查用)
    返回:dict —— {"ok": False, "error": {"code", "message", "hint"}}
    """
    return {"ok": False,
            "error": {"code": code,            # 机器可读的错误码
                      "message": message,      # 人可读的细节
                      "hint": _ERROR_HINTS.get(code, "")}}  # 给 LLM 的建议动作


def search(query: str, k: int = SEARCH_DEFAULT_K, *,
           filters: dict | None = None,
           min_score: float | None = None,
           max_chars: int | None = None) -> dict:
    """检索工具:给 LLM 的 function calling 用(文本进 → 结构化 dict 出 → 不抛异常)。

    成功:{"ok": True,
          "hits": [{chunk_id, doc_id, title, text, score}, ...],
          "meta": {"recall_k": 50, "returned": 5, "filtered": 2}}
        —— meta 三数让调用方(LLM 或人)看出"召回多少/过滤前多少/最终多少",评测调试靠它。
    失败:{"ok": False,
          "error": {"code": "...", "message": "...", "hint": "..."}}

    参数:
        query: str —— 检索文本(★ 收文本不收向量:内部先 embed_query 再调底层,与 D6 解耦)
        k: int —— 最终返回条数,默认 SEARCH_DEFAULT_K=5
        filters: dict | None —— 元数据后置过滤;★ faiss 不支持元数据过滤,
            固定位置 =「回填 meta 之后、返回之前」—— 提前过滤是给 faiss 提要求(做不到),
            推后过滤浪费召回名额;过滤后不足 k → 按 k x SEARCH_EXPAND_FACTOR 扩召回
            再过滤【最多一次,不递归】,仍不足按实际条数返回,不报错
            (结果少不是错误,对照错误码表的 NO_HITS)
            filters = {"doc_id_in":      ["wiki123", ...],  # 文档白名单
                       "title_contains": "机器学习",       # 标题子串
                       "chunk_index_lt": 10,               # 只取文首块}
            组合过滤 = 交集(各键同时满足);底层签名不动。
        min_score: float | None —— 相似度阈值,低于则丢弃
        max_chars: int | None —— 单块 text 截断(控 token;LLM 消费不需要全文)
    返回:dict —— 纯数据(json.dumps 可序列化,async-ready,将来单测 T7 断言)

    async-ready 约定(§5.4):
        接口只收发纯数据(不返回 generator/文件句柄);工具层内部调 pipeline 的
        非流式模式(stream=False);将来并发用 asyncio.to_thread 包一层,底层零改动。
        ★ 与 D6 generate(stream=True) 的张力辨析:流式是 pipeline 表现层特性(面向人);
          工具层面向 LLM 消费,LLM 要一次拿全的结构化结果 —— 边界不同,不矛盾。

    实现步骤(D9 §5.1 伪代码,落地时逐步展开 —— 本轮【不写实现】,只保留骨架):
        1. if not query.strip(): return _error(ERR_EMPTY_QUERY, "query 为空")   # 不抛!
        2. try: q_vec = embed_query(query)
           except 连接类异常: return _error(ERR_OLLAMA_UNAVAILABLE, str(e))     # 不抛!
        3. try: hits = retrieve.search(q_vec, k=recall_k)   # 默认 recall_k = k;
           except FileNotFoundError:                        #   有 filters 时直接取 kxfactor
               return _error(ERR_INDEX_NOT_LOADED, str(e))  # 不抛!
        4. 若 filters:对 hits 逐条做三键交集过滤(doc_id_in / title_contains /
           chunk_index_lt)→ filtered 计数;
           不足 k 且未扩过召回 → recall_k 扩大一次重查(不递归);
           仍不足按实际条数返回,不报错(NO_HITS 语义)
        5. 若 min_score:丢弃低分;若 max_chars:截断 text
        6. return {"ok": True, "hits": hits[:k],
                   "meta": {"recall_k": recall_k, "returned": len(结果), "filtered": 过滤数}}
           ★ hits=[] 也是 ok=True(NO_HITS 不是错误!)
    """
    raise NotImplementedError  # ★ 本轮红线:只定接口;W5 落地时按上面 6 步展开


# ============================================================================
# Agent 循环(伪代码,D9 §5.5,不含实现;复用现有模块,不重写)
# ============================================================================
#   复用关系(D9.5 原表):
#       retrieve(经工具层 search 包装)= 检索工具本体;
#       qwen = 生成器(function calling 的消息循环由落地时实现);
#       pipeline.ask = fallback 工具(LLM 想"直接要答案"而不只是"要资料"时可用)。
#
#   Agent 循环(形态 a 起步):
#       messages = [system, user]
#       while 未达停止条件(轮数 < AGENT_MAX_TOOL_ROUNDS):
#           决策 = LLM(messages, tools=[search])          # function calling
#           if 决策是调用工具:
#               result = search(**决策.args)              # 结构化结果,不抛错
#               messages += [工具调用记录, result]
#           else:
#               break                                     # LLM 决定直接作答
#       return 决策.内容
#
# ⚠️ 一句话说清 Agent 的边界(§10.1):Agent 不改善召回上限,只改善「何时检索/检索几次」
#    —— 检索质量仍受 D2(切片)/D3(向量化)/D4(索引)支配;
#    真正的召回改善来自 rerank(D6)与混合检索(W5)。
#    期待"上了 Agent 检索就准了"是方向性误解。
# ============================================================================

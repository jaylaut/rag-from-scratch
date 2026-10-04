# -*- coding: utf-8 -*-
"""D6 - pipeline.py:端到端编排(系统的门面)—— 完整版【参考副本】

⚠️ 本文件是 src/code/ 下的参考副本:import 按 src/ 目录结构书写,直接运行会 ImportError;
   要运行请放回 src/pipeline.py(见 src/code/README.md 对照表)。

调用链(D6 完整四步,③ rerank 接通;D5 极简版只有 ①②④,③ 直通原序截断):
    ask(question)
      ├─① offline.embed.embed_query(question)      问题 → 1024 维向量(~10ms)
      ├─② online.retrieve.search(q_vec, k=50)      召回宽口径 top-50 + 回填
      ├─③ online.rerank.rerank(question, hits, 5)  交叉编码器精排(失败自动降级原序截断)
      ├─④ online.qwen.build_prompt + generate      窄口径 top-5 进 prompt → qwen-plus
      └─ 返回 {"answer": str, "sources": [{title, chunk_id, score}, ...]}

k 与 top_m 的两层口径(铁律,任何文档/代码不得混用,D6 指引 §1):
    k = RECALL_K = 50 是"召回宽口径"(向量检索的候选池,只在 ② 消费);
    top_m = TOP_M = 5 是"精读窄口径"(真正进 prompt 的来源数,只在 ③④ 消费)。
    trade-off:进 prompt 的越多越贵、越慢、越可能跑题。

rerank=False 旁路(★公共接口,不是内部参数):
    完全跳过 ③(直接 hits[:top_m])—— 这是 D8 第一组 A/B(rerank 开/关)的物理开关,
    必须是干净的分岔,不是"调 rerank 但传个空参数"(D6 §5.9)。

三条铁律(D5 指引 §1,D6 继承):
    1. 拒答是特性:语料外的问题必须"老实说资料中没有"(由 QWEN_SYSTEM_PROMPT 兑现);
    2. 签名一次做对:k 默认 RECALL_K=50(召回语义,E2 裁决以占位签名为准);
    3. meta 的 text 一个字符都不改:回填的 text 直接进 prompt,保证"文本→向量"可比性。

对应文档:D6在线链路开发指引.md §5.9 | 系统架构说明.md §4
用法(放回原位后,venv 激活,cd src):
    python -c "from pipeline import ask; print(ask('京剧的起源是什么?'))"
    python pipeline.py             # 交互式演示
    python pipeline.py --no-rerank # 旁路精排(D6 §9.2 实验的命令行形态)
"""
import time        # 标准库:分段计时(检索/重排/生成耗时,D6 §5.9 日志点)
import logging     # 标准库:日志

from common import config                 # 项目内:RECALL_K / TOP_M
from offline.embed import embed_query     # 项目内:① 问题向量化(D3 第 2 层,前置依赖)
from online import retrieve               # 项目内:② 检索 + 回填
from online import rerank as rerank_mod   # 项目内:③ 重排;别名避免与参数名 rerank 撞名
from online import qwen                   # 项目内:④ prompt 组装 + 生成

logger = logging.getLogger(__name__)      # 模块级 logger


def ask(question: str, k: int = config.RECALL_K, top_m: int = config.TOP_M,
        rerank: bool = True, stream: bool = True) -> dict:
    """端到端问答:问题 → 带引用来源的答案(D6 完整编排 ①②③④)。

    功能实现原理(编排层 = 纯调度 + 分段计时,不含业务逻辑):
        ① embed → ② search(k) → ③ rerank(可旁路)→ ④ prompt + 生成;
        返回的 sources 从【最终进 prompt 的那批 hits】抽取 ——
        rerank 开时是重排后顺序、关时是原序,这样答案里的 [来源N] 与 sources 下标严格对齐。

    参数:
        question: str  —— 用户问题,必填
        k: int         —— 召回宽口径,默认 config.RECALL_K=50(只在 ② 消费)
        top_m: int     —— 精读窄口径,默认 config.TOP_M=5(只在 ③④ 消费)
        rerank: bool   —— True=走交叉编码器精排(内部失败自动降级);
                          False=完全跳过 ③,直接原序截断(D8 A/B 的开关)
        stream: bool   —— True=generate 返回增量迭代器,ask 内部拼成完整串再返回
    返回:dict —— {"answer": str, "sources": [{title, chunk_id, score}, ...]}
    明确不做:embed 失败/索引缺失这类【不可恢复】错误照样抛 ——
              编排层不是降级壳,工具层才转结果(D9 §5.3 的分层原则);
              不做多 query 改写(D9 之后);不缓存 query 向量(W5 的 LRU)。
    """
    # ---- ① 问题向量化:文本 → (1, 1024) 归一化向量(Ollama bge-m3,~10ms)----
    q_vec = embed_query(question)

    # ---- ② 召回宽口径:k 只在这里消费 ----
    t0 = time.time()                          # 分段计时:延迟数字的原始来源(D6 §5.9)
    hits = retrieve.search(q_vec, k=k)        # [{score, chunk_id, title, text}, ...] x ~k 条
    t_retrieval = time.time() - t0            # 检索耗时(验收线 < 1s)
    logger.info("检索命中 %d 条,耗时 %.3fs", len(hits), t_retrieval)

    # ---- ③ 精排或旁路:干净的分岔(A/B 开关的物理实现)----
    if rerank:                                # 开关为真:走交叉编码器
        t0 = time.time()
        # rerank 内部自带降级壳:失败时它自己降级为原序截断,编排层无感知
        picked = rerank_mod.rerank(question, hits, top_m=top_m)
        logger.info("重排完成,耗时 %.3fs,取 top-%d", time.time() - t0, len(picked))
    else:                                     # 开关为假:完全跳过 ③(单测 T9 靠此断言)
        picked = hits[:top_m]                 # 原序截断 = rerank 的降级形态,行为可对照
        logger.info("rerank=False 旁路,原序截断 top-%d", top_m)

    # ---- ④ prompt 组装 + 生成 ----
    prompt = qwen.build_prompt(question, picked)   # 编号 [来源1..N],N 与 picked 条数一致
    logger.info("prompt %d 字符(来源 %d 条)", len(prompt), len(picked))
    t0 = time.time()
    if stream:
        # stream=True:generate 返回增量迭代器 —— ask 内部消费成完整字符串后返回;
        # (若将来要做"边生成边打印",把 for 循环体里加 print 即可,契约不变)
        chunks = []
        for delta in qwen.generate(prompt, stream=True):   # 逐段接收增量
            chunks.append(delta)                           # 收集片段
            print(delta, end="", flush=True)               # 顺手实时打印(感知延迟的来源)
        print()                                            # 收尾换行
        answer = "".join(chunks)                           # ★ 增量拼接才是完整全文
    else:
        answer = qwen.generate(prompt, stream=False)       # 非流式:直接拿完整串
    logger.info("生成耗时 %.2fs(检索 %.3fs)", time.time() - t0, t_retrieval)

    # ---- 组装 sources:从 picked(最终进 prompt 的那批)抽取,[来源N] 与下标严格对齐 ----
    sources = [
        {
            "title": h.get("title") or "",      # 逐键兜底,防上游缺字段
            "chunk_id": h.get("chunk_id") or "",# 全局唯一主键,溯源用
            "score": h.get("score", 0.0),       # 已在 retrieve round 到 4 位小数
        }
        for h in picked                          # 顺序 = 重排后顺序(或旁路原序)
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # ---- 演示入口:--no-rerank 可体验 A/B 差异(D6 §9.2 实验的命令行形态)----
    import sys                                    # 读命令行参数
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_rerank = "--no-rerank" not in sys.argv    # 带 --no-rerank 就旁路精排
    print(f"RAG 演示(D6 完整版,rerank={'开' if use_rerank else '关'})。输入 q 退出。")
    while True:
        question = input("\n你的问题: ").strip()  # 读一行用户输入并去首尾空白
        if question.lower() in ("q", "quit", "exit"):
            break                                 # 退出循环
        if not question:
            continue                              # 空输入不处理,重新问
        result = ask(question, rerank=use_rerank) # 调编排入口(流式打印在 ask 内部完成)
        print("\n\n引用来源:")                    # 答案已流式打出,这里补来源清单
        for i, s in enumerate(result["sources"], 1):
            print(f"  [来源{i}] {s['title']}  (chunk_id={s['chunk_id']}, score={s['score']})")

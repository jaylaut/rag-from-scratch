# -*- coding: utf-8 -*-
"""D6 - rerank.py:重排(在线流水线第③步)—— 全新落地【参考副本】

⚠️ 本文件是 src/code/ 下的参考副本:import 按 src/ 目录结构书写,直接运行会 ImportError;
   要运行请放回 src/online/rerank.py(见 src/code/README.md 对照表)。

原理(架构 §6-R4 / D6 指引 §3.1):
    D3 的 bge-m3 是"双编码器":query 和 doc 各自独立编码成向量再比余弦 —— 快,可离线建库;
    bge-reranker-base 是"交叉编码器":把 (query, doc) 拼成一对喂进模型,
    注意力在两者之间充分交互后直接输出相关性分 —— 准得多,但每对都要现算。
    生产标配组合 = 向量召回(快、粗,k=50)→ 交叉编码器精排(慢、准,top_m=5)。

关键约束(§6-R7):
    Ollama 不支持 cross-encoder(R1 已知代价),本模块走 onnxruntime(纯 C++ 推理,
    不引入 torch);依赖豁免范围严格限定在本模块。

★★ 本模块唯一的铁律(D6 指引 §1):
    rerank 任何失败(模型缺失/推理异常/输入畸形)一律【降级为原序截断 top_m】,绝不抛异常
    打断问答;降级必须打 WARNING(带原始异常栈),与成功的 INFO 日志可区分 ——
    静默降级 = rerank 坏了三个月都没人发现。
    例外:hit 缺 text 键属上游契约破坏,让它穿透降级壳向外抛(降级壳只兜自身失败)。

对应文档:D6在线链路开发指引.md §5.3~§5.5 | 架构 §4 步骤③、§6-R4/R7
"""
import time        # 标准库:计时(模型加载耗时/推理耗时日志)
from pathlib import Path   # 标准库:模型文件路径处理
import logging     # 标准库:日志

import numpy as np   # 第三方:分数数组与排序

from common import config  # 项目内:RERANK_MODEL_NAME / RERANK_MAX_LENGTH / TOP_M

logger = logging.getLogger(__name__)

# ---- 工程常量(留模块级,不进 config —— D6 §2 约定)----
_RERANK_BATCH = 16       # 每批喂给 onnx 的 (query, doc) 对数;50 条一次全喂内存峰值过高
_RERANK_SESSION = None   # onnxruntime InferenceSession 单例(懒加载,同 embed._get_session 套路)
_TOKENIZER = None        # tokenizer 单例(与 session 一起加载,一起复用)
_MODELS_DIR = Path("models")  # onnx 权重与 tokenizer 的存放目录(gitignore;D6 §2 依赖清单)


def _load_model():
    """首次调用时加载 tokenizer + onnx session(延迟单例,失败裸抛交给降级壳)。

    功能实现原理:
        与 embed._get_session 同款"None + 懒加载"单例:模型加载 1~2 秒,
        只应在第一次 rerank 时发生,之后全程复用。
        ★ 本函数自己【不做降级】—— 它只管加载,加载失败原样抛出,
          由上层 rerank() 的降级壳统一捕获(职责分离,D6 §5.3)。

    调用的外部方法:
        tokenizers.Tokenizer.from_file(path):从本地 JSON 加载分词器
            (轻量库 tokenizers,不装 transformers;tokenizer.json 是 HuggingFace 导出件);
        onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"]):
            加载 onnx 计算图;providers 指定 CPU 执行
            (CPU 跑 50 对 x 512 token 几百 ms,够用;明确不做 GPU —— D6 §5.3)。

    参数:无
    返回:None(结果写入模块级 _TOKENIZER / _RERANK_SESSION)

    明确不做:不做 GPU;不校验模型文件哈希(下载完整性是交付时的活)。
    """
    global _TOKENIZER, _RERANK_SESSION      # 声明改的是模块级变量(不是局部变量)
    if _RERANK_SESSION is not None:         # 已加载过:直接返回(单例,第二次起零开销)
        return
    from tokenizers import Tokenizer        # 函数内导入:失败时 ImportError 也归降级壳管
    import onnxruntime as ort               # 同上
    t0 = time.time()                        # 计时起点:首次加载 1~2 秒,必须打日志
    # tokenizer 目录约定:models/bge-reranker-base/tokenizer.json(HuggingFace 导出件)
    tok_path = _MODELS_DIR / config.RERANK_MODEL_NAME / "tokenizer.json"
    if not tok_path.is_file():              # 权重/分词器缺失 → FileNotFoundError 向上抛
        raise FileNotFoundError(
            f"rerank 模型文件缺失:{tok_path}。"
            f"请按 D6 指引 §2 下载 bge-reranker-base 的 onnx 导出件到 models/ 目录。"
        )
    _TOKENIZER = Tokenizer.from_file(str(tok_path))                  # 加载分词器(参数要 str)
    # ★ 首次加载时打印输入/输出张量名(D6 §10.1 第 1 坑:onnx 张量名因导出方式而异,
    #   必须按名字取,不能按位置猜 —— 打出来一次,后面排查省一半时间)
    _RERANK_SESSION = ort.InferenceSession(                          # 加载 onnx 计算图
        str(_MODELS_DIR / config.RERANK_MODEL_NAME / "model.onnx"),
        providers=["CPUExecutionProvider"],                          # 明确用 CPU 后端
    )
    logger.info("rerank 模型加载完成(耗时 %.2fs),输入张量:%s,输出张量:%s",
                time.time() - t0,
                [i.name for i in _RERANK_SESSION.get_inputs()],      # 输入张量名列表
                [o.name for o in _RERANK_SESSION.get_outputs()])     # 输出张量名列表


def _score_pairs(query: str, hits: list[dict]) -> np.ndarray:
    """对每条 hit 计算 (query, doc) 对的相关性分(原始 logits → sigmoid → [0,1])。

    功能实现原理(D6 指引 §3.7 流程):
        tokenizer.encode_batch(batch_pairs) → session.run(输入名, feed)
            → 原始 logits(未过激活)→ sigmoid 压到 [0,1] → 与 hits 一一对应的分数组。

    三个必踩点(D6 §3.7):
        ① 张量名按 _load_model 打印的名字取,不硬编码;
        ② attention_mask 必须一起喂 —— 否则 padding 部分参与计算,分数系统性偏差;
        ③ 输出是 logits,必须过 sigmoid 才是"分数"
           (不过激活直接排序结果一样 —— 单调变换,但日志里的 ±10 级分数会吓人)。

    调用的外部方法:
        Tokenizer.encode_batch(list_of_pairs):批量编码句对,返回 Encoding 对象列表;
            Encoding.ids=词 id 序列 / .attention_mask=真词位置 / .type_ids=段 id。
        InferenceSession.run(output_names, feed_dict):执行推理;None=取全部输出。
        np.exp(-logits) + 1 的倒数 = 手写 sigmoid:1/(1+e^-x),把 logits 压到 (0,1)。

    参数:
        query: str —— 用户问题;hits: list[dict] —— 至少含 text 键
    返回:np.ndarray —— 形状 (len(hits),) 的 float32 分数,顺序与 hits 一致

    明确不做:不捕获异常(裸抛,让降级壳统一处理);不处理空 hits(调用方保证)。
    """
    _load_model()              # 懒加载单例(已加载则零开销)
    # ---- 构造 (query, doc) 对的批量编码:句对是交叉编码器的输入形态(不是单句!)----
    enc = None                 # 占位声明(实际编码在下一行完成,拆开写便于阅读)
    enc = [_TOKENIZER.encode(query, h["text"]) for h in hits]  # 逐条编码 (query, doc) 句对
    # 注意:h["text"] 缺键时 KeyError 在这里抛 —— 上游契约错误,穿透降级壳(§5.5 约定)
    scores = np.zeros(len(hits), dtype=np.float32)    # 结果容器,顺序与 hits 对齐
    for start in range(0, len(hits), _RERANK_BATCH):  # 分批推理,控内存峰值(§5.4)
        batch = enc[start:start + _RERANK_BATCH]      # 本批的编码结果
        # 三组输入张量:input_ids=词 id;attention_mask=真词位置;padding 部分必须被它屏蔽
        feed = {
            "input_ids": np.array([b.ids for b in batch], dtype=np.int64),
            "attention_mask": np.array([b.attention_mask for b in batch], dtype=np.int64),
            "token_type_ids": np.array([b.type_ids for b in batch], dtype=np.int64),
        }
        # run(None=取全部输出)[0]=第一个输出张量;ravel() 拉平成一维,与本批对齐
        logits = _RERANK_SESSION.run(None, feed)[0].ravel()
        scores[start:start + len(batch)] = 1.0 / (1.0 + np.exp(-logits))  # ★ 手写 sigmoid
    return scores


def rerank(query: str, hits: list[dict], top_m: int = config.TOP_M) -> list[dict]:
    """精排入口:宽口径 hits(约 50 条)→ 按相关性重排 → 窄口径 top_m 条。

    功能实现原理(降级壳,本模块核心):
        候选不足 top_m 时重排无意义,直接原样返回;
        正常路径:_score_pairs 打分 → 按分数降序重排 → 截 top_m;
        任何异常:logger.warning(..., exc_info=True) 留栈后【原序截断 top_m】返回 ——
        绝不向上抛异常打断问答(§1 铁律);降级不重试(重试是编排层/D9 的事)。

    调用的外部方法:
        np.argsort(-scores, kind="stable"):分数取负后升序 = 原分数降序;
            stable 保证同分时保持原序(结果可复现,P4 原则)。

    参数:
        query: str —— 用户问题(交叉编码器需要原文,不是向量)
        hits: list[dict] —— retrieve.search 的宽口径结果,每条含 score/chunk_id/title/text
        top_m: int —— 最终保留条数,默认 config.TOP_M=5
    返回:list[dict] —— 条数 == min(top_m, len(hits)),顺序 = 精排顺序(成功)或原序(降级)

    明确不做:不重试;不做分数归一化以外的后处理;不缓存分数(同一 query 不会短时重排两次)。
    """
    if not hits:                    # 空输入:原样返回空列表(不触发模型加载,省 1~2 秒)
        return []
    if len(hits) <= top_m:          # 候选不足,重排无意义(顺序没得变)
        return list(hits)           # 拷贝一份返回,避免调用方原地修改影响缓存
    try:
        scores = _score_pairs(query, hits)            # 正常路径:交叉编码器打分
    except Exception:                                 # ★ 兜一切失败:模型缺失/推理出错/形状不对
        logger.warning("rerank 失败,降级为原序截断 top_m=%d", top_m,
                       exc_info=True)                 # ★ 带原始异常栈:降级必须可观测(§1 配套)
        return hits[:top_m]                           # ★ 契约:原序截断,绝不中断链路
    order = np.argsort(-scores, kind="stable")        # 分数降序的索引序列(稳定排序)
    ranked = [hits[int(i)] for i in order]            # 按新顺序重排(int() 把 numpy 索引转 int)
    logger.info("rerank 完成:before=%s(%.4f) after=%s(%.4f),重排 %d 条",
                hits[0].get("chunk_id"), hits[0].get("score", 0.0),      # 重排前首条
                ranked[0].get("chunk_id"), float(scores[order[0]]),      # 重排后首条及其新分
                len(hits))
    # ↑ 这条日志是 D8 攒素材的原始来源之一(D6 §5.5):同题 rerank 前后首条变化一目了然
    return ranked[:top_m]                             # 精排顺序截取窄口径

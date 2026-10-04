# -*- coding: utf-8 -*-
"""D6 - qwen.py:prompt 组装 + Qwen 生成(在线流水线第④步)—— 完整版【参考副本】

⚠️ 本文件是 src/code/ 下的参考副本:import 按 src/ 目录结构书写,直接运行会 ImportError;
   要运行请放回 src/online/qwen.py(见 src/code/README.md 对照表)。

D6 相对 D5 的三个升级点(D5 指引 §12 E4 的兑现):
    1. count_tokens:tiktoken cl100k 估算 token 数(中文是近似值,只做预算比较,不做计费);
    2. build_prompt 加"超预算削减":超过 QWEN_PROMPT_BUDGET_TOKENS 时按固定顺序削减
       —— ① 先砍单来源截断长度(可多轮减半)→ ② 再减来源个数(5→3→2)→ ③ 绝不动问题;
    3. generate 加超时与重试(指数退避,最多 2 次,只针对网络/超时类错误)。
(D5 最小版差异:无 count_tokens、无削减循环、generate 无论 stream 与否都返回完整 str。)

Prompt 工程三要素(对 RAG,D6 指引 §3.4,已固化在 config.QWEN_SYSTEM_PROMPT):
    角色限定(防自由发挥)+ 依据限定(防幻觉:"仅根据参考资料")+ 格式限定([来源N] 标注)。

对应文档:D6在线链路开发指引.md §5.6~§5.8 | 系统架构说明.md §4 步骤④
"""
import os          # 标准库:检查 DASHSCOPE_API_KEY 是否已设置
import time        # 标准库:统计生成耗时/重试退避
import logging     # 标准库:日志

import dashscope   # 第三方:阿里云百炼(DashScope)官方 SDK,pip install dashscope

from common import config  # 项目内:QWEN_MODEL / QWEN_SYSTEM_PROMPT / QWEN_PROMPT_BUDGET_TOKENS

logger = logging.getLogger(__name__)

# ---- 工程常量(留模块级,不进 config —— D6 §2 约定)----
_SRC_SEP = "\n\n"            # 来源块之间的分隔符(两个换行 = 空行,模型读起来分段清晰)
_NO_SOURCE_NOTICE = (        # 空 hits 时的兜底文案(触发拒答,D5 已验证)
    "参考资料为空。资料中没有与该问题相关的内容,请明确说明无法根据资料回答。"
)
_BUDGET_MARGIN = 0.9         # 削减循环留 10% 余量:tiktoken 对中文是近似值,
                             # "估着没超、实际超了"比多砍一点更糟(D6 §5.6 踩坑清单第 4 条)
_MIN_SRC_CHARS = 100         # 单来源截断长度的下限:砍到 100 字以下就停止"减半"这一步


def count_tokens(text: str) -> int:
    """用 tiktoken cl100k 估算文本的 token 数(纯函数,无状态)。

    功能实现原理:
        tiktoken 是 OpenAI 的分词器;cl100k 编码表对中文约 1 字 ≈ 1.5~2 token,
        与 qwen 自己的 tokenizer 有偏差 —— 但预算比较只需要"量级正确",不需要精确值。
        ★ 用途红线:只用于预算比较(本文件 build_prompt),绝不用于计费预估(D8 §10.4 同款提醒)。

    调用的外部方法:
        tiktoken.get_encoding("cl100k_base"):取编码表对象;
            ★ 首次调用会联网下载编码表并缓存(离线机器要预热一次,或提前 pin 缓存);
        encoding.encode(text) / len(...):文本 → token id 列表 → 数个数。

    参数:text: str —— 待估算文本
    返回:int —— 估算 token 数(近似值)
    明确不做:不缓存结果(纯函数);不做按模型精确分词(qwen 有自己的 tokenizer,不值得为此引依赖)。
    """
    import tiktoken                       # 函数内导入:未安装时错误信息更聚焦
    enc = tiktoken.get_encoding("cl100k_base")  # 编码表(首次调用触发下载/读本地缓存)
    return len(enc.encode(text))          # encode 得到 token id 列表,数长度即 token 数


def build_prompt(question: str, hits: list[dict],
                 max_src: int = config.TOP_M, src_max_chars: int = 800) -> str:
    """组装带 [来源N] 编号的 prompt,超预算时按固定顺序削减。

    功能实现原理:
        基础拼装同 D5(指令 + [来源N] 资料 + 问题);
        D6 新增削减循环(D6 指引 §3.5 固定顺序,不许临场发挥):
            while 估算 token > 预算:
                第 1 优先级:src_max_chars 减半(损失"信息密度":只丢尾部);
                第 2 优先级:max_src 递减 5→3→2(损失"覆盖面");
                绝不动 question(用户意图排在一切之上)。
            减无可减(src_chars < 100 或 src 数 ≤ 2)就退出,带着现状去请求。

    [来源N] 编号契约:编号 = hits 列表序号 + 1,与 pipeline 返回的 sources 下标
    【严格一致】—— 否则答案里的 [来源2] 对不上 sources[1],引用溯源就断了。

    参数:
        question: str —— 用户问题(任何削减都不动它)
        hits: list[dict] —— retrieve/rerank 的输出,每条含 score/chunk_id/title/text
        max_src: int —— 最多几条来源,默认 config.TOP_M=5
        src_max_chars: int —— 单来源截断长度(字符),默认 800;超预算时会被减半
    返回:str —— 拼好的完整 prompt
    明确不做:不往 prompt 里塞检索分数(徒增 token,§5.6);不做流式相关的事(generate 的活)。
    """
    # ---- 空 hits:拒答路径(不抛错,D5 T3 契约)----
    if not hits:
        return f"{config.QWEN_SYSTEM_PROMPT}\n\n{_NO_SOURCE_NOTICE}\n\n问题:{question}\n回答:"

    # ---- 削减循环的工作副本:循环内反复改这两个局部变量,不改调用方入参 ----
    cur_src = max_src                 # 当前来源个数
    cur_chars = src_max_chars         # 当前单来源截断长度
    budget = int(config.QWEN_PROMPT_BUDGET_TOKENS * _BUDGET_MARGIN)  # 预算打 9 折(留近似误差余量)

    def _assemble(n_src: int, n_chars: int) -> str:   # 内部小函数:按给定参数拼 prompt
        parts = []
        # enumerate(..., 1):编号从 1 开始;切片 [:n_src] 只取前 n_src 条
        for i, hit in enumerate(hits[:n_src], 1):
            text = (hit.get("text") or "")[:n_chars]  # 缺字段兜底成空串 + 按长度截断
            title = hit.get("title") or ""            # 标题同样兜底
            parts.append(f"[来源{i}] {title}\n{text}")  # 一条来源:[来源N] + 标题 + 正文
        context = _SRC_SEP.join(parts)                # 所有来源段用空行连接
        return (f"{config.QWEN_SYSTEM_PROMPT}\n\n"    # 系统指令(角色/依据/格式三重限定)
                f"参考资料:\n{context}\n\n问题:{question}\n回答:")  # 资料 → 问题 → 回答:

    prompt = _assemble(cur_src, cur_chars)            # 第一版:按默认参数拼
    # ---- 削减主循环:超预算才进入;每一轮做一次"动作",重新拼、重新估 ----
    while count_tokens(prompt) > budget:
        if cur_chars > _MIN_SRC_CHARS:                # 优先级 1:还能减半就先减半
            cur_chars = max(cur_chars // 2, _MIN_SRC_CHARS)  # 整除减半,但不低于下限
            logger.info("prompt 超预算,单来源截断降到 %d 字", cur_chars)
        elif cur_src > 2:                             # 优先级 2:截不动了再减来源个数
            cur_src = max(cur_src - 2, 2)             # 5→3→2(每次 -2,快速收敛)
            logger.info("prompt 仍超预算,来源个数降到 %d", cur_src)
        else:                                         # 减无可减:带着现状退出,让 API 侧兜底
            logger.warning("prompt 削减到下限仍超预算(src=%d, chars=%d),按现状请求",
                           cur_src, cur_chars)
            break
        prompt = _assemble(cur_src, cur_chars)        # 按新参数重拼,回到循环头重新估
    return prompt


def generate(prompt: str, stream: bool = True):
    """调用 qwen-plus 生成;stream=True 返回增量迭代器(D6 定稿形态)。

    功能实现原理:
        stream=True:返回 Iterator[str] —— 每次迭代给一个增量片段(★ 不是累计全文),
            由调用方(pipeline.ask)拼成完整字符串;流式改变的是感知延迟,不是答案质量;
        stream=False:返回完整 str —— 评测场景更好断言(D8 走这条路)。
        ★ 两种模式返回类型不同是刻意的(D6 §5.8):调用方按自己传的 stream 知道拿到什么。

    重试策略(D6 §5.8):指数退避(2s/4s),最多重试 2 次;只针对网络/超时类异常
        (requests 系);API 参数错误/Key 错误属于确定性失败,重试没有意义,直接抛。

    调用的外部方法:
        dashscope.Generation.call(model, result_format, messages, stream, incremental_output):
            result_format="message" —— 返回 OpenAI 风格消息结构(旧版 "text" 已弃用);
            stream=True + incremental_output=True —— 每轮给增量片段(不是累计全文);
            response.status_code != 200 —— 业务失败(Key 错/参数错/限流),必须当场检查。

    参数:
        prompt: str —— build_prompt 的产物
        stream: bool —— True=返回迭代器;False=返回完整字符串
    返回:str | Iterator[str] —— 由 stream 参数决定
    明确不做:不做本地模型路由(W5 才对比 qwen2.5:7b);不缓存。
    """
    # ---- 入口检查:Key 必须在(D5 §10.1,报错要可读)----
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY。请检查项目根 .env 与 shell 环境。")
    messages = [{"role": "user", "content": prompt}]  # 单条 user 消息(指令已在 prompt 内)

    if not stream:
        # ---- 非流式:带重试的一次性调用 ----
        last = None                                   # 保存最后一次异常供报错
        for attempt in range(3):                      # 1 次正常 + 2 次重试
            try:
                t0 = time.time()
                resp = dashscope.Generation.call(     # 同步调用,内部等待完整结果
                    model=config.QWEN_MODEL,          # 模型名,来自 config(单一事实来源)
                    result_format="message",          # 消息格式(见函数 docstring 说明)
                    messages=messages,                # 对话内容
                    stream=False,                     # 一次性等待完整结果
                )
                if resp.status_code != 200:           # 业务失败(Key/参数/限流)不重试
                    raise RuntimeError(f"qwen-plus 失败:code={resp.code}, message={resp.message}")
                answer = resp.output.choices[0].message.content  # 完整答案直接取
                logger.info("qwen 生成完成:耗时 %.2fs,%d 字符", time.time() - t0, len(answer))
                return answer                         # 成功即返回完整串
            except RuntimeError:                      # 上面主动抛的 RuntimeError = 确定性失败
                raise                                 # 不重试,原样上抛
            except Exception as e:                    # 其余(网络/超时)= 可重试
                last = e
                if attempt < 2:                       # 还有重试机会
                    time.sleep(2 ** attempt)          # 指数退避:2s / 4s
                    logger.warning("qwen 第 %d 次失败:%s,退避后重试", attempt + 1, e)
        raise RuntimeError(f"qwen 调用失败(已重试 2 次):{last}")  # 机会用完,如实上抛

    # ---- 流式:返回增量迭代器(生成器函数写法,yield 逐段产出)----
    def _iter():
        resp_iter = dashscope.Generation.call(        # SDK 返回可迭代对象
            model=config.QWEN_MODEL,
            result_format="message",
            messages=messages,
            stream=True,                              # 流式开关
            incremental_output=True,                  # ★ 每轮给增量片段(不是累计全文)
        )
        for resp in resp_iter:                        # 逐轮接收
            if resp.status_code != 200:               # 任一轮失败都要当场炸,不能拼出残缺答案
                raise RuntimeError(f"qwen 流式失败:code={resp.code}, message={resp.message}")
            delta = resp.output.choices[0].message.content  # 本轮增量文本(可能为空)
            if delta:                                 # 空片段(None/"")跳过
                yield delta                           # yield:把增量交给调用方
    return _iter()                                    # 返回迭代器本身(注意:不是执行它)

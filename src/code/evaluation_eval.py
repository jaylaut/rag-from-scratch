# -*- coding: utf-8 -*-
"""D8 - eval.py:评估闭环(检索质量的唯一裁判)—— 完整版【参考副本】

⚠️ 本文件是 src/code/ 下的参考副本:import 按 src/ 目录结构书写,直接运行会 ImportError;
   要运行请放回 src/evaluation/eval.py,并在 src/common/config.py 追加两个常量:
       EVAL_SET_FILE = DATA_EVAL / "eval_set.jsonl"
       EVAL_N = 60

职责(三主函数 + 四辅助):
    1. build_eval_set:固定种子抽 n 篇 → qwen-plus 每篇出 1 题 → 落盘草稿 → 人工筛
       (★ 铁律 1:出题只是草稿,60 题每题必须人工过目 —— 垃圾进垃圾出);
    2. evaluate:逐题检索 → 判定 target 是否命中 → hit@k / MRR 三指标
       + 报告文件(含最差 10 case,改切片参数时盯它们);
    3. ab_test:单变量 A/B 对比框架(★ 铁律 2:每组对比只允许一个变量不同,否则结论作废)。

评测集行格式(D8 指引 §1 裁决):
    {"question": str, "target_title": str} 为主键格式;
    "target_chunk_id" 为可选字段(仅用于定位原文人工核对)。
    ★ 用 title 而不用 chunk_id 判定:title 对 chunk 重组稳健(重切后 chunk_id 全变,
      title 不变),第二组 A/B(切片参数)才能跨切片版本复用同一套题。

对应文档:D8评估闭环开发指引.md §5 | 系统架构说明.md §9
运行(放回原位后):python -m evaluation.eval [build|evaluate|ab]
"""
import random      # 标准库:固定种子抽样(可复现,P4 原则)
import json        # 标准库:评测集行序列化
import time        # 标准库:A/B 组间计时
import logging     # 标准库:日志
from datetime import datetime   # 标准库:报告文件名的时间戳
from pathlib import Path        # 标准库:路径处理

from common import config          # 项目内:EVAL_SET_FILE / EVAL_N / DATA_EVAL
from common.utils import iter_jsonl  # 项目内:JSONL 流式逐行读(内存安全)
from offline.embed import embed_query    # 项目内:问题向量化(①)
from online import retrieve              # 项目内:use_rerank=False 走的原序检索(②)
from online import qwen                  # 项目内:出题(草稿)用
from online import pipeline              # 项目内:use_rerank=True 走的完整链路(②③④)

logger = logging.getLogger(__name__)

_REPORT_DIR = config.DATA_EVAL / "reports"  # 评测报告落盘目录(展示层路径,不进 config,D8 §2)

# ---- 第一组 A/B 的标准变体(D8 §5.3 约定:描述性键名 + dict 值,禁止位置传参)----
VARIANTS_RERANK = {"rerank_off": {"use_rerank": False},
                   "rerank_on": {"use_rerank": True}}


# ============================================================================
#                     第 1 层:辅助纯函数(单测的靶子)
# ============================================================================
def _norm_title(title: str) -> str:
    """标题归一化:strip + 全角转半角 + 去空格 + 大小写折叠。

    功能实现原理:
        评测集与语料的 title 生成路径不同,空格/全半角/大小写三件事会造成
        "其实命中了却判为未命中"的假阴性 —— 裸字符串比较会平白多出错误结论,
        所以判定前必须先归一化到同一形态(D8 §10.1 第 1 坑,单测 T7 兜底)。

    调用的外部方法:
        str.strip():去首尾空白;str.translate(table):按映射表逐字符替换(全角→半角);
        str.replace(" ", ""):去全部空格;str.lower():英文统一小写(大小写折叠)。

    参数:title: str —— 任意一侧的标题
    返回:str —— 归一化后的标题(只用于比较,不回写任何数据)
    """
    t = title.strip()                       # 去首尾空白
    # 全角→半角映射表:{ord(全角字符): ord(半角字符)};str.translate 按 Unicode 码位替换
    table = {ord(f): ord(h) for f, h in (
        ("０", "0"), ("１", "1"), ("２", "2"), ("３", "3"), ("４", "4"),
        ("５", "5"), ("６", "6"), ("７", "7"), ("８", "8"), ("９", "9"),
        ("Ａ", "a"), ("Ｂ", "b"), ("Ｃ", "c"), ("Ｄ", "d"), ("Ｅ", "e"),
        ("Ｆ", "f"), ("Ｇ", "g"), ("Ｈ", "h"), ("Ｉ", "i"), ("Ｊ", "j"),
        ("ａ", "a"), ("ｂ", "b"), ("ｃ", "c"), ("ｄ", "d"), ("ｅ", "e"),
        ("（）", "()"), ("：", ":"), ("，", ","), ("　", " "),)}
    t = t.translate(table)                  # 逐字符套用全角→半角映射
    return t.replace(" ", "").lower()       # 去掉全部空格 + 英文统一小写


def _hit_at_k(ranked_titles: list[str], target_title: str, k: int) -> int:
    """判定 target 是否出现在 top-k(返回 0/1);同文多块命中按 title 去重算一次。

    功能实现原理:
        hit@k 回答"能不能检到":top-k 里出现目标文章即得 1 分;
        重叠切片下同一篇文章命中多块很正常 —— 不去重会虚高 hit@k,
        所以这里用 set 去重后判定(D8 §10.2 第 2 坑,单测 T8 兜底)。

    参数:
        ranked_titles: list[str] —— 检索结果按序排列的 title 列表(重排后顺序)
        target_title: str —— 该题应命中的文章标题
        k: int —— 判定窗口大小(如 5 或 10)
    返回:int —— 1(命中)或 0(未命中)
    """
    top_k = {_norm_title(t) for t in ranked_titles[:k]}  # 前 k 条 title 归一化并去重(set)
    return int(_norm_title(target_title) in top_k)       # in 判定,int() 把布尔转 0/1


def _mrr_single(ranked_titles: list[str], target_title: str) -> float:
    """单题 MRR 贡献:1 / target 的排名;未命中贡献 0。

    功能实现原理:
        MRR(Mean Reciprocal Rank)回答"排得靠不靠前",比 hit@k 多了排名信息:
        同样 hit@5 命中的两版检索,第 1 名命中(1/1=1.0)与第 5 名命中(1/5=0.2)差距悬殊。
        ★ 公式核心是【倒数】1/rank —— 直接用 rank 均值是常见错误(mutation 反证项 1)。

    参数:同 _hit_at_k(ranked_titles / target_title)
    返回:float —— 1/rank 或 0.0
    """
    target = _norm_title(target_title)                  # 归一化后比较
    for rank, title in enumerate(ranked_titles, 1):     # enumerate 从 1 数:rank 就是人类排名
        if _norm_title(title) == target:
            return 1.0 / rank                           # ★ 倒数:排第 1 得 1.0,第 10 得 0.1
    return 0.0                                          # 整个结果列表都没有 → 贡献 0


def _write_report(path: Path, metrics: dict, per_case: list[dict]) -> None:
    """把指标 + 逐题排名表落盘成 Markdown 报告。

    功能实现原理:
        报告 = 三指标汇总 + 逐题明细 + 最差 10 case(按 1/rank 升序,未命中排最前)。
        最差 case 是改切片参数时盯的对象(路线计划原话),必须可追溯每题的排名。

    调用的外部方法:
        Path.mkdir(parents=True, exist_ok=True):递归建目录,已存在不报错;
        Path.write_text(str, encoding):一次写全(报告很小,无需流式)。

    参数:
        path: Path —— 报告输出路径
        metrics: dict —— {"hit@5": x, "hit@10": y, "MRR": z, ...}
        per_case: list[dict] —— 每题 {question, target_title, rank, reciprocal}
    返回:None(写文件)
    """
    path.parent.mkdir(parents=True, exist_ok=True)      # 确保报告目录存在
    lines = [f"# 评测报告 {datetime.now().isoformat(timespec='seconds')}", ""]  # 带时间戳标题
    lines.append(f"- hit@5  = {metrics['hit@5']:.4f}")  # 三个指标各一行,4 位小数
    lines.append(f"- hit@10 = {metrics['hit@10']:.4f}")
    lines.append(f"- MRR    = {metrics['MRR']:.4f}")
    lines += ["", "| rank | question | target |", "|---|---|---|"]      # 逐题明细表头
    for c in per_case:                                   # 逐题一行:排名 + 问题摘要 + 目标
        lines.append(f"| {c['rank']} | {c['question'][:40]} | {c['target_title'][:20]} |")
    path.write_text("\n".join(lines), encoding="utf-8")  # join 后一次写盘
    logger.info("评测报告已写入:%s", path)


# ============================================================================
#                第 2 层:三主函数(build_eval_set / evaluate / ab_test)
# ============================================================================
def build_eval_set(n: int = config.EVAL_N,
                   out: Path = config.EVAL_SET_FILE,
                   corpus: Path = config.CHUNKS_SUBSET_FILE) -> None:
    """构造人工过目的评测集:固定种子抽样 → qwen-plus 出题 → 落盘草稿。

    功能实现原理(D8 指引 §3.2 四步):
        ① 固定种子随机抽 n 篇文章(可复现,P4 原则;种子打进日志);
        ② 每篇让 qwen-plus 出 1 个"应命中该文"的问题,要求:答案在文中、
           问题不抄标题原词(抄标题会把任务退化成字符串匹配,hit 判定就失真了);
        ③ 落盘草稿 JSONL;
        ④ 人工筛通道:直接编辑 jsonl 删行(最简单可靠),最终行数 == EVAL_N 才算定稿。

    调用的外部方法:
        random.seed(42) / random.sample(pop, n):固定种子 + 无放回抽样(两次运行同结果,T5);
        iter_jsonl(corpus):项目内工具,流式逐行读 JSONL(内存安全);
        qwen.generate(..., stream=False):出题用非流式 —— 好截取、好断言;
        qwen.build_prompt(ask_prompt, []):复用拼装逻辑(空 hits 走拒答文案,不影响出题)。

    参数:
        n: int —— 目标题数,默认 config.EVAL_N=60
        out: Path —— 评测集输出路径(占位签名裁决 E1:以 config 路径为准)
        corpus: Path —— 从哪个 chunks 文件抽样(默认调试子集)
    返回:None
    明确不做:不做自动难度分级;不让 LLM 出多题选一(可控性差);
              出题失败的文章跳过并记日志,不重试出题(质量宁可人筛补)。
    """
    random.seed(42)                                   # 固定种子:两次运行抽到同一批文章(T5)
    logger.info("评测集抽样种子=42,目标 %d 题,语料:%s", n, corpus)
    # 先把全部文章读进内存建立"标题 → 文本"索引(子集几万行,可接受;
    # 按文章去重:一个 title 只出一题,避免同文多块重复出题)
    articles: dict[str, str] = {}
    for obj in iter_jsonl(corpus):                    # 流式逐行读(内存安全)
        title = obj.get("title") or ""
        if title and title not in articles:           # 同名文章只保留第一次出现
            articles[title] = obj.get("text") or ""
    titles = list(articles.keys())                    # 去重后的标题清单
    sampled = random.sample(titles, min(n, len(titles)))  # 无放回抽样 n 个标题
    logger.info("语料共 %d 篇(去重后),本次抽 %d 篇出题", len(titles), len(sampled))

    out.parent.mkdir(parents=True, exist_ok=True)     # 确保 data/eval 存在
    with open(out, "w", encoding="utf-8") as f:       # "w" 模式:每次重建草稿
        for i, title in enumerate(sampled, 1):        # 逐篇出题并打日志(中断了能续)
            text = articles[title]
            # 出题 prompt:读文章 → 出一个不抄标题的问题(草稿生成,后续人工筛)
            ask_prompt = (f"请根据以下文章出一个检索测试问题。要求:答案必须能在文章中找到,"
                          f"问题不要出现标题原词。只输出问题本身。\n\n文章标题:{title}\n"
                          f"正文:{text[:1500]}")    # 正文截 1500 字,控制出题成本
            try:
                question = qwen.generate(             # 非流式调用 qwen-plus 出题
                    qwen.build_prompt(ask_prompt, []),
                    stream=False).strip()             # strip 去掉模型可能带的空白
            except Exception as e:                     # 出题失败:跳过该篇,不中断整批
                logger.warning("第 %d 篇(%s)出题失败,跳过:%s", i, title, e)
                continue
            f.write(json.dumps({"question": question,          # 落盘一行:问题 + 目标标题
                                "target_title": title},
                               ensure_ascii=False) + "\n")     # ensure_ascii=False 保中文可读
            logger.info("出题进度 %d/%d:%s", i, len(sampled), question[:30])
    logger.info("草稿已落盘 %s。请人工逐题过目后删行,直到剩余 %d 题为止。", out, n)


def evaluate(eval_set: Path, k_list: list[int] | None = None,
             use_rerank: bool = False) -> dict:
    """逐题检索 → 判定 → hit@k + MRR → 报告 + 最差 10 case。

    功能实现原理:
        ★ 关键裁决:use_rerank 必须参数化且默认 False —— 第一组 A/B 是"rerank 开/关",
          开关必须在评测器里,否则 A/B 根本无法实现(D8 §5.2)。
        检索口径两条路径,检索 k 相同(= max(k_list)),变量只剩 rerank(单变量铁律):
            use_rerank=True  → pipeline.ask(rerank=True, top_m=k_max),吃完整链路;
            use_rerank=False → retrieve.search(k=k_max) 原序,不进 prompt 不生成
            (★ 评测只测检索质量,不调 LLM 生成 —— 又快又省钱又确定)。
        hit 判定用 rank 直接判:1 <= rank <= k 即命中(与 _hit_at_k 语义等价)。

    调用的外部方法:
        retrieve.load_index():复用索引单例(60 题不重载,§10.3;调用即触发首次加载);
        embed_query(question):每题向量化;pipeline.ask / retrieve.search:两条检索路径;
        sorted(per_case, key=...):按 1/rank 升序取最差 10 case(未命中 reciprocal=0 排最前)。

    参数:
        eval_set: Path —— 评测集 JSONL(每行 {question, target_title})
        k_list: list[int] | None —— 判定窗口组;None 时补 [5,10]
                 (E2 裁决:默认值必须明确,None 的隐式默认是口径漂移温床)
        use_rerank: bool —— True=走 pipeline.ask(重排开);False=走 retrieve 原序
    返回:dict —— {"hit@5": float, "hit@10": float, "MRR": float, "n": int, "report": str}
    明确不做:不做置信区间(60 题没有统计意义);不做多 target;
              对数据质量零容忍(缺字段当场抛,评测器学离线铁律,不学工具层转结果)。
    """
    if k_list is None:                    # E2 裁决:默认 [5,10] 在这里补齐
        k_list = [5, 10]
    k_max = max(max(k_list), 10)          # 检索口径 = 最大 k(一次检索,多档判定);下限 10

    # ---- 读评测集 + 逐行校验(缺字段带行号报错,T9)----
    cases = []
    for lineno, obj in enumerate(iter_jsonl(eval_set), 1):  # enumerate 从 1 起记行号
        if not obj.get("question") or not obj.get("target_title"):  # 双字段都必填
            raise ValueError(f"{eval_set} 第 {lineno} 行缺 question/target_title 字段")
        cases.append({"question": obj["question"], "target_title": obj["target_title"]})
    logger.info("评测开始:%d 题,k_list=%s,use_rerank=%s", len(cases), k_list, use_rerank)

    per_case: list[dict] = []             # 逐题明细(rank / reciprocal)
    retrieve.load_index()                 # ★ 复用索引单例(60 题不重载,§10.3)
    for i, case in enumerate(cases, 1):
        q_vec = embed_query(case["question"])          # ① 问题向量化
        if use_rerank:                                 # 路径 A:完整链路(重排开)
            result = pipeline.ask(case["question"], k=k_max, top_m=k_max,
                                  rerank=True, stream=False)
            ranked = [s["title"] for s in result["sources"]]   # sources 已按精排序
        else:                                          # 路径 B:原序检索(变量只剩 rerank)
            hits = retrieve.search(q_vec, k=k_max)
            ranked = [h["title"] for h in hits]
        # ---- 判定:target 的排名(1 起;0=未命中);同文多块按 title 天然合并 ----
        case["rank"] = next((r for r, t in enumerate(ranked, 1)
                             if _norm_title(t) == _norm_title(case["target_title"])), 0)
        case["reciprocal"] = 1.0 / case["rank"] if case["rank"] else 0.0  # 倒数,未命中=0
        per_case.append(case)
        if i % 10 == 0 or i == len(cases):             # 进度日志(每 10 题一条)
            logger.info("评测进度 %d/%d", i, len(cases))

    # ---- 汇总三指标 ----
    n = len(per_case)
    metrics: dict = {}
    for k in k_list:                                    # 逐个 k 算命中率:hit@k = 命中数/总数
        hits_count = 0
        for c in per_case:
            hits_count += int(0 < c["rank"] <= k)       # rank 在 [1,k] 区间即命中
        metrics[f"hit@{k}"] = hits_count / n if n else 0.0
    metrics["MRR"] = sum(c["reciprocal"] for c in per_case) / n if n else 0.0  # 倒数均值

    # ---- 最差 10 case:按 1/rank 升序,未命中(rank=0,reciprocal=0)排最前 ----
    worst = sorted(per_case, key=lambda c: c["reciprocal"])[:10]
    report_path = _REPORT_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    _write_report(report_path, metrics, per_case)       # 全量明细落盘
    logger.info("评测完成:hit@5=%.4f hit@10=%.4f MRR=%.4f;最差 case:%s",
                metrics.get("hit@5", 0), metrics.get("hit@10", 0), metrics["MRR"],
                [(w["target_title"], w["rank"]) for w in worst])
    metrics["n"] = n                                    # 附带题数,方便 ab_test 展示
    metrics["report"] = str(report_path)                # 附带报告路径
    return metrics


def ab_test(eval_set: Path, variants: dict) -> None:
    """单变量 A/B 对比框架:同一评测集,多组配置各跑一遍 evaluate,一张表出结果。

    功能实现原理(D8 §5.3):
        variants = {"变体名": {"use_rerank": True/False, ...}} —— 描述性键名 + dict 值,
        禁止位置传参(对比表要打印变体名);逐个跑 evaluate(同一评测集文件),
        汇总成 markdown 对比表(hit@5 / hit@10 / MRR 各一行),可直接粘进文档回填。

    参数:
        eval_set: Path —— 评测集路径(所有变体必须同一份,框架内校验)
        variants: dict —— {变体名: evaluate 的 kwargs 字典}
    返回:None(结果打印 + 落盘)
    明确不做:不做显著性检验(60 题量级无意义);不做多变体网格
              (变量 >1 时自己拆成多组跑,单变量铁律)。
    """
    if len(variants) < 2:                                 # 至少两个变体才叫"对比"
        raise ValueError("ab_test 至少需要 2 个变体")
    results: dict[str, dict] = {}                         # {变体名: evaluate 返回的指标}
    for name, kwargs in variants.items():                 # 逐组评测,组间计时
        t0 = time.time()
        logger.info("A/B 组 [%s] 开始,参数:%s", name, kwargs)
        results[name] = evaluate(eval_set, **kwargs)      # kwargs 透传给 evaluate
        logger.info("A/B 组 [%s] 结束,耗时 %.1fs", name, time.time() - t0)
    # ---- 汇总对比表(markdown):第一列指标名,后面每列一个变体 ----
    lines = ["| 指标 | " + " | ".join(variants.keys()) + " |",
             "|---" * (len(variants) + 1) + "|"]
    for key in ("hit@5", "hit@10", "MRR"):                # 三行指标 x N 列变体
        row = [f"{results[v].get(key, 0):.4f}" for v in variants]
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    table = "\n".join(lines)
    print(table)                                          # 直接打印,可粘贴回填
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)        # 确保报告目录存在
    (_REPORT_DIR / "ab_test.md").write_text(table, encoding="utf-8")  # 同时落盘
    logger.info("A/B 对比表已写入 %s", _REPORT_DIR / "ab_test.md")


if __name__ == "__main__":
    # ---- 命令行入口:三步演示(建评测集 → 评测 → rerank A/B)----
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "evaluate"   # 子命令,默认 evaluate
    if mode == "build":                                       # 第 1 步:出题草稿
        build_eval_set()
    elif mode == "ab":                                        # 第 3 步:第一组 A/B(rerank 开/关)
        ab_test(config.EVAL_SET_FILE, VARIANTS_RERANK)
    else:                                                     # 第 2 步:单次评测(默认原序)
        m = evaluate(config.EVAL_SET_FILE)
        print(f"hit@5={m['hit@5']:.4f} hit@10={m['hit@10']:.4f} MRR={m['MRR']:.4f}")

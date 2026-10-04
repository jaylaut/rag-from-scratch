# src/code/ — D5~D9 代码参考副本

> **⚠️ 本目录性质：参考副本，不可直接运行。**
> 所有文件的 `import`（如 `from common import config`）是按 `src/` 目录结构写的；
> 若要运行，需把对应文件放回 `src/` 下的原位置（见下表"原位路径"列），本目录仅作阅读与日后复制用。
> （用户约定：本轮禁止改动 `src/code/` 之外的任何文件，故原位文件保持占位状态。）

## 文件 ↔ 阶段对照表

| 本目录文件 | 原位路径（运行时放回此处） | 阶段 | 状态口径 |
|---|---|---|---|
| `online_retrieve.py` | `src/online/retrieve.py` | D5 最小版 → **D6 升级版**（本文件为 D6 最终版，头部注明三点差异） | 完整版 |
| `online_rerank.py` | `src/online/rerank.py` | **D6**（2026-10-02 由 W5 上调为主线，方案 A：onnxruntime cross-encoder + 失败降级壳） | 全新 |
| `online_qwen.py` | `src/online/qwen.py` | D5 最小版 → **D6 完整版**（count_tokens / 超预算削减 / 重试） | 完整版 |
| `pipeline.py` | `src/pipeline.py` | D5 极简编排 → **D6 完整编排**（接通③ rerank，rerank=False 旁路） | 完整版 |
| `d7_config_additions.py` | 追加到 `src/common/config.py` 末尾 + 跑批手册 | **D7**（零新代码阶段；本文件= config 片段 + 命令清单 + 记录表） | 手册型 |
| `evaluation_eval.py` | `src/evaluation/eval.py` | **D8**（build_eval_set / evaluate / ab_test + 4 辅助函数） | 完整版 |
| `d9_agent_tools.py` | （将来 W5 落地为 `src/agent/tools.py`） | **D9**（只定接口不写实现；**文件头有醒目红线标注**） | 设计骨架 |

## D6/D8 落地时还需追加到 `src/common/config.py` 的常量（本目录不代改）

```python
# ---- rerank(D6 新增,架构 §6-R4/R7)----
RERANK_MODEL_NAME = "bge-reranker-base"   # cross-encoder,本地 onnxruntime 推理
RERANK_MAX_LENGTH = 512                   # tokenizer 截断长度(query+doc 拼接后)

# ---- prompt 预算(D6 新增)----
QWEN_PROMPT_BUDGET_TOKENS = 6000          # prompt 超过此数触发削减

# ---- 评测(D8 新增)----
EVAL_SET_FILE = DATA_EVAL / "eval_set.jsonl"
EVAL_N = 60

# ---- M 主展示档产物路径(D7 新增,§3.3 决策点;两套库并存,不复用也不删 wiki_*)----
CHUNKS_CORPUS_FILE = DATA_CHUNKS / "corpus_chunks.jsonl"
VECTORS_CORPUS_FILE = DATA_VECTORS / "corpus_chunks.vec.npy"
INDEX_CORPUS_FILE = INDEX_DIR / "corpus.faiss"
META_CORPUS_FILE = INDEX_DIR / "corpus_chunks_meta.jsonl"
```

## 依赖变更（D6 开工第一步，落回原位时执行）

- `pip install onnxruntime tokenizers`（**不引入 torch**，架构 §6-R7 豁免仅限 rerank 一个模块）；
- bge-reranker-base 的 onnx 导出件 + tokenizer.json 放 `models/bge-reranker-base/`（gitignore）；
- requirements.txt 头部注释同步改为"rerank 走 onnxruntime(§6-R7)，不引 torch"。

## 已知缺陷登记（因"禁止动其他文件"约束，本轮只登记不修）

| 位置 | 缺陷 | 修法 |
|---|---|---|
| `src/offline/embed.py:337` | `np.vstack()` 漏传参数 | 改为 `np.vstack(parts)` |
| `src/offline/embed.py:334` | 批切片 `texts[i:1 + batch_size]` 首批多拿 1 条 | 改为 `texts[i:i + batch_size]` |
| `src/offline/embed.py:385~389` | `embed_chunks` 把 config 常量写在参数注解位而非默认值位 | 改为 `in_path=config.CHUNKS_SUBSET_FILE, ...` 真默认值写法 |

> ⚠️ 上述缺陷是 D5 的前置依赖（D5 指引 §0：D3 第 2 层 `embed_query` 必须可用），落回原位时**先修再跑**。

## 执行顺序提醒（照各指引"实现顺序"节）

修 D3 缺陷 → D5（6.1→6.6）→ D6（requirements → retrieve → qwen → pipeline 旁路 → rerank → 降级壳 → 12 条单测）→ D7 跑批 → D8（evaluate 假数据手工验算 → build_eval_set → ab_test → 真跑）→ W5 按 D9 骨架落地 agent。

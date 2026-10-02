# tests/ 使用说明

## 怎么跑

在项目根(注意是**项目根**,不是 `src/`、也不是 `tests/`):

```bash
cd ~/program/LLM/agent/rag-from-scratch   # 以本机实际仓库路径为准(macOS)
source .venv/bin/activate

# 全部用例
python -m pytest

# 只看某个文件
python -m pytest -v tests/unit/offline/test_chunk.py

# 日常开发推荐:跳过要大数据和要联网/花钱的
python -m pytest -m "not bigdata and not network"
```

> 为什么必须在项目根跑?因为 `pytest.ini` 里有 `pythonpath = src`,pytest 只有在
> 项目根才能把它识别为 rootdir 配置,进而把 `src` 加进 `sys.path`。
> (`tests/conftest.py` 里还有一道兜底保险,但别依赖它。)

## 目录怎么对应 src

```
tests/unit/common/        -> src/common/       (config.py, utils.py)
tests/unit/offline/       -> src/offline/      (本轮从 chunk.py 起)
tests/unit/online/        -> src/online/
tests/unit/evaluation/    -> src/evaluation/
tests/unit/test_pipeline.py -> src/pipeline.py
tests/integration/        -> 跨模块串联 / 跑真实产物的黑盒验收
```

**src 根的 `check_env.py`(联网自检)和 `export_docx.py`(题库导出工具)不镜像**——
前者是环境脚本、后者是旁路工具,都不属于 RAG 主链路。

**D0(`download.py`)与 D1(`parse.py`)也不建测试文件**:它们已跑完验收,历史断言
(共 76 项)留在**当年 Windows 开发机**工作目录下的 `.workbuddy/tmp/` 验证脚本里
(该路径未随仓库迁到 macOS,如需回溯请查旧机器 / git 历史),不重复投入。

## 命名与跳过约定

- 文件一律 `test_*.py`,测试函数一律 `test_*`(pytest 的默认发现规则,别改);
- 每个文件开头用 `pytestmark = pytest.mark.xxx` 打整篇标记;
- **新文件放占位用例是允许的**,但 skip 原因必须写清三件事:
  ① 对应 src 的哪个模块,② 为什么跳过(未实现 / 待迁移),③ 真实用例从哪来
  (比如"见 D2切片开发指引.md §6 的 T1~T11")。
- ⚠ **skipped ≠ 通过**。跑完看到一片 `SKIPPED` 只代表框架通了、还没写用例,
  别当成绿灯。这也是占位文件里必须写清"未来用例出处"的原因。

## 标记(markers)

| 标记 | 含义 | 日常要不要跑 |
|---|---|---|
| `unit` | 纯函数级,不读大文件、不联网、不花钱 | ✅ 每次都跑 |
| `integration` | 跨模块串联,可能读 `data/` 下真实产物 | 视情况 |
| `network` | 需要 Ollama 服务或 Qwen API(**调 Qwen 会花钱**) | ❌ 默认跳过 |
| `bigdata` | 需要 `data/articles/wiki_zh.jsonl`(GB 级,本机实测约 2.3~3.8 GB),慢 | ❌ 默认跳过 |
| `slow` | 单条超过 10 秒 | ❌ 默认跳过 |

## 两档验收 ↔ pytest 标记对应(2026-10-02)

`开发路线与学习计划.md` 把每个阶段的验收拆成 **A 档(离线可验证)** 与 **B 档(需真实数据)** 两档。对应到测试标记:

| 验收档位 | 判据 | 对应 pytest 标记 | 命令 |
|---|---|---|---|
| **A 档 · 离线可验证** | 合成样本 + monkeypatch,**不读 `data/`、不连 Ollama/Qwen** | `unit` | `pytest -m "not bigdata and not network"` |
| **B 档 · 需真实数据** | 跑真实产物并回填实测数字(**数据未就绪时标 ⏸ 待跑**) | `integration` / `network` / `bigdata` | 单独指定,不进日常回归 |

> **为什么分两档**:2026-10-02 起开发机分两台 —— Mac M1(日常写代码,**不准备数据**)/ Win10(假期后跑真实数据)。不分档的话,Mac 上写代码期间会出现"验收清单勾不了也不敢留空"的尴尬。
> **d3/d4 的 B 档**(吞吐实测、端到端问答、续跑)大概率在 Win10 上完成。

## fixtures/ 铁律

只放**手搓的、几 KB 以内**的合成小样本。严禁把 `data/` 下 GB 级的真实产物
拷进来 —— 真实数据一律用 `skipif(not exists)` + `@pytest.mark.bigdata` 按需加载。

## 写用例的一条铁律

不许写**恒真断言**。项目历史上踩过坑:`a_last + b_first in ''.join(lines)`
因为 `''.join` 去掉了分隔符,相邻两行拼起来必然是子串,断言永远为真——
看着有牙齿,其实一条 bug 都咬不住。

每写一条断言,都要能回答:"我把实现改错成什么样,这条会 FAIL?"
答不上来的断言,重写。

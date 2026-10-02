# 项目长期记忆 — rag-from-scratch

## 环境约定（macOS M1 开发机）

- 解释器：`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`（官方 3.12.4）；venv 在项目根 `.venv`，激活用 `source .venv/bin/activate`。
- 不用 Anaconda 的 3.14（wikiextractor 需要 Python ≤3.12 的 cgi 模块）。
- Ollama 由 Homebrew 安装（0.11.8），向量服务默认 `http://localhost:11434`，模型 bge-m3（1024 维，Metal GPU）。启动用 `bash scripts/start_ollama.sh`。
- pip 全局走清华源（写在 `~/.config/pip/pip.conf`）。
- faiss-cpu 在 macOS 上必须 `<1.10`（1.10+ 无 macOS wheel）。

## 运行约定

- 命令行一律 `cd src && python -m <模块>`（src 下是扁平导入，如 `from common import config`）。
- 测试：`python -m pytest tests -q`（pytest.ini 的 pythonpath=src + tests/conftest.py 双重保险）。
- 密钥放项目根 `.env`（已 gitignore），模板 `.env.example`。

## 环境差异提醒

- 仓库里 `开发环境配置手册.md` 是 **Windows 版**（E 盘、GTX 1050Ti、PyCharm 2025.3.3）；macOS 对应文档是 `开发环境配置手册-macOS.md`。

## 文档口径约定（2026-10-02 起）

- **规模数字的单一事实来源 = `系统架构说明.md` §5.1**（S 冒烟 ~2,000 / M 主展示 1~5 万 / L 规模基线 ~20 万 / 设计上限 ~100 万块）。其他文档**只引用不复制**，避免口径漂移。
- **语料是双语料**：维基 = L 规模基线（只测延迟/内存，不做质量指标）；垂直语料（LLM/AI 中文技术资料）= M 主展示（质量评测 + 作品集演示）。**切片参数对两套语料通用**。
- **验收标准分两档**：A 档 = 离线可验证（`pytest -m "not bigdata and not network"`，合成样本 + monkeypatch）；B 档 = 需真实数据（跑真实产物并回填实测数字，数据未就绪标 ⏸ 待跑）。阶段"完成" = A 档全绿 且 B 档至少在最小规模样本上跑过一次。
- **开发机分两台**：Mac M1 16GB（日常写代码，不准备数据）/ Win10 32GB（假期后跑真实数据）。
- **状态表一律用「分层状态 + 日期」**（如「第 1 层 7 函数 ✅ / 第 2~4 层 ⬜」），不用笼统的「✅ / 占位」。
- **已知代码缺陷只写进各 D 指引的勘误节**（D3 §12 / D4 §12），不在文档重排轮次里顺手改代码。
- `wiki` 前缀是**历史命名**，语义上已泛指主语料（技术债，待后续重命名）。
- rerank 已从 W5 提到 **D6 主线**；Agent 是 **D9（只定接口，不写实现）**。

# 开发环境配置手册 — macOS (Apple Silicon / M1) 版

> 本文是 `开发环境配置手册.md`(Windows 版)在 **MacBook Air M1** 上的对应落地记录。
> 架构完全一致,只把"Windows 专属的部分"换成 macOS 做法。
> 📌 **语料规模口径**(wiki 规模基线 / 垂直语料主展示 / 设计上限)见 [系统架构说明.md](系统架构说明.md) **§5.1** —— 本文不复制具体数字,以免两处口径漂移。

| 项目 | 本机实测值 |
|---|---|
| 机器 | MacBook Air M1(arm64),16GB 统一内存,macOS 13.0 Ventura |
| 解释器 | `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` → **Python 3.12.4**(官方安装包自带,无需再装) |
| 虚拟环境 | `<项目根>/.venv`(venv,Python 3.12.4) |
| 向量模型 | Ollama 本地服务 **bge-m3**,1024 维,**Metal GPU 100%** |
| 检索 | faiss-cpu **1.9.0.post1**(macOS arm64 轮子) |
| 生成 | 远程 Qwen API(dashscope) |
| Ollama | Homebrew 装 **0.11.8**,模型库 `~/.ollama/models`(1.1GB) |
| 磁盘 | 131GB 可用(**无需任何缓存重定向**) |
| IDE | PyCharm 2026.2.3 |

**与 Windows 版的四个关键差异**

1. **没有 NVIDIA 显卡**——Ollama 在 Mac 上走 **Metal**,`ollama ps` 显示 `100% GPU` 即正常(统一内存 10.7GB 可分配给模型)。
2. **不用管磁盘**——单块 228GB SSD 且剩余 131GB,Windows 版"缓存全指向 E 盘"那一整章可跳过,只保留一条 `HF_ENDPOINT` 镜像。
3. **Python 3.12 已就位**——`/Library/Frameworks` 下已有官方 3.12.4(另有 3.10)。注意 **不要用 Anaconda 的 3.14**,wikiextractor 依赖的 `cgi` 模块在 3.13 已被移除。
4. **faiss-cpu 必须锁 <1.10**——上游从 1.10 起不再发布 macOS 轮子(只剩源码包需要 cmake+swig 编译)。已在 `requirements.txt` 用平台标记处理,Windows 侧行为不变。

---

## 阶段 1 — 前置确认(只读)

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -V   # 期望 Python 3.12.4
/opt/homebrew/bin/brew --version                                    # 期望 4.6.20
df -h / | tail -1                                                    # 确认剩余空间 ≥ 20GB
```

- [x] Python 3.12.4 就位
- [x] Homebrew 4.6.20 可用(注意:**不要执行 `brew update`**,会把 tap 拉到不支持 Ventura 的新版本)
- [x] 磁盘 131GB 可用

---

## 阶段 2 — 装 Ollama

```bash
export PATH="/opt/homebrew/bin:$PATH"
export HOMEBREW_NO_AUTO_UPDATE=1     # 防止自动 update 到不支持 Ventura 的 brew
brew install ollama                  # 实测倒入 arm64_ventura 预编译包,24 秒装完,0.11.8
```

### 启动服务(⚠️ macOS 上的坑)

Homebrew 会写入 `~/Library/LaunchAgents/homebrew.mxcl.ollama.plist` 做登录自启,但 **该 LaunchAgent 要到"下次登录/重启"才会被 launchd 载入**。在自动化/非 Aqua 会话里执行会报:

```
Bootstrap failed: 5: Input/output error
```

**两种解法**

```bash
# ① 临时拉起(下次登录前用这个)—— 项目里已备好脚本,幂等:
bash scripts/start_ollama.sh          # 已在跑就什么都不做;--force 可强制重启

# ② 永久生效:在你自己的“终端 App”里执行下面这条(图形会话中 launchctl 才可用)
brew services start ollama
# 之后每次登录自动启动;也可 brew services stop ollama 停止
```

验证:

```bash
curl -s http://localhost:11434/api/version     # 期望 {"version":"0.11.8"}
```

---

## 阶段 3 — 创建虚拟环境

```bash
cd /Users/program/LLM/agent/rag-from-scratch
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate        # macOS 用 source,不是 Windows 的 Scripts\activate
python -V                        # 期望 Python 3.12.4,前缀出现 (.venv)
```

> 系统 PATH 里 Anaconda 排在前面(`~/.zshrc` 有 conda init),**必须用上面的完整路径建 venv**,否则会命中 3.14。

---

## 阶段 4 — pip 换清华源(全局,永久)

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config list    # 期望 global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

写入位置:`~/.config/pip/pip.conf`(macOS 没有 pip.ini)。

---

## 阶段 5 — 装依赖

```bash
cd /Users/program/LLM/agent/rag-from-scratch
source .venv/bin/activate
pip install -r requirements.txt      # 已走清华源,约 3~5 分钟
```

`requirements.txt` 已加平台标记(macOS 最高只能到 1.9.x):

```
faiss-cpu>=1.8,<1.10; sys_platform == "darwin"
faiss-cpu>=1.8; sys_platform != "darwin"
```

> **若遇到 `EEXIST: file already exists, mkdir '.../pip-install-xxx'`**
> 这是 WorkBuddy 终端沙箱注入的 `sitecustomize` 拦截了 `os.mkdir`,导致 pip 解包 sdist(jieba 是唯一的源码包)失败,**不是你环境的问题**。在你自己的终端里正常装即可;自动化环境下可临时绕过:
> `env -u PYTHONPATH pip install -r requirements.txt`

验证:

```bash
python -c "import faiss, requests, dashscope, jieba, tiktoken, wikiextractor, datasets; print('全部安装成功')"
```

---

## 阶段 6 — 向量模型 bge-m3

```bash
ollama pull bge-m3     # 1.2GB,实测几分钟;存到 ~/.ollama/models
python -c "import requests; v=requests.post('http://localhost:11434/api/embed', json={'model':'bge-m3','input':['测试']}, timeout=180).json()['embeddings'][0]; print('维度:', len(v))"
# 期望:维度: 1024(首次调用约 20s,之后常驻内存很快)
ollama ps
# 期望:bge-m3:latest  1.8 GB  100% GPU  ← Metal 加速生效
```

---

## 阶段 7 — API Key 与 .env

项目根已建好 `.env`(占位)和 `.env.example`(模板),`.gitignore` 已排除 `.env`。

```bash
cp .env.example .env    # 已存在则跳过
# 编辑 .env,把 DASHSCOPE_API_KEY 换成阿里云百炼真实 Key
# 申请:https://bailian.console.aliyun.com/
```

填好后验证(会花几分钱):

```bash
source .venv/bin/activate
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from dashscope import Generation; r=Generation.call(model='qwen-plus', prompt='用一句话介绍维基百科', api_key=os.environ['DASHSCOPE_API_KEY']); print(r.output.text)"
```

可选:把镜像写进 shell,让 `datasets` 等命令行工具也走国内源(`.env` 里的 `HF_ENDPOINT` 只对 Python 进程生效):

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
```

---

## 阶段 8 — 一键自检

```bash
cd /Users/program/LLM/agent/rag-from-scratch
source .venv/bin/activate
python src/check_env.py               # 完整检查
python src/check_env.py --skip-embed  # 快速检查
```

**本次实测结果(2026-10-01)**

```
[PASS] Python 版本 = 3.12        Python 3.12.4
[PASS] Ollama 服务可达           Ollama 0.11.8 @ http://localhost:11434
[PASS] bge-m3 已拉取             bge-m3 已就绪
[PASS] bge-m3 向量推理           维度 1024 | 向量范数 1.0000
[PASS] faiss-cpu 可用            faiss 1.9.0,IndexFlatIP(1024) 创建成功
[FAIL] DASHSCOPE_API_KEY 已配置  ← 仅剩这项:填入真实 Key 后即全绿
[PASS] wikiextractor 可用        wikiextractor 可导入
```

测试套件冒烟:`python -m pytest tests -q` → **11 passed, 10 skipped**(skip 的都是等 D3~D8 实现的占位用例)。

### 全景验收脚本(推荐日常用这个)

`src/check_env.py` 只覆盖 7 项核心;项目里另配了 **16 项全景验收**,一条命令连 venv、pip 源、src 导入链路、faiss 检索往返、GPU 占用、pytest 冒烟一起查:

```bash
bash scripts/verify.sh                # 自动激活 venv + 拉起 Ollama + 全量验收(不花钱)
bash scripts/verify.sh --skip-embed   # 快速版(跳过向量推理)
bash scripts/verify.sh --live         # 额外真实调一次 Qwen(约花几分钱)
python scripts/verify_setup.py --json # 机器可读输出,便于 CI/脚本解析
```

覆盖项:Python 3.12 + venv / .venv 位置 / 依赖导入 / pip 源 / `from common import config` / Ollama 服务 / bge-m3 已拉取 / 向量推理 1024 维 / GPU 占用 / faiss 检索往返 / API Key / Qwen 真实调用 / wikiextractor CLI / 目录结构 / 磁盘余量 / pytest 冒烟。
退出码:`0` 全通过(允许 WARN),`1` 有 FAIL。

---

## 阶段 9 — PyCharm 集成

1. **打开项目**:File → Open → `/Users/program/LLM/agent/rag-from-scratch`
2. **绑定解释器**:Settings → Project → Python Interpreter → Add Local Interpreter → **Select existing** → 选 `<项目根>/.venv/bin/python`
   - 注意 macOS 下是 `bin/python`,不是 `Scripts/python.exe`
3. **标记源码根**:右键 `src` → Mark Directory as → Sources Root(src 下是扁平导入 `from common import config`,必须让 src 在 sys.path 上)
4. **验证终端**:PyCharm 底部 Terminal 里 `python -V` → 3.12.4

- [ ] 右下角显示 `Python 3.12 (rag-from-scratch)`
- [ ] 能运行 `src/check_env.py` 且 7 项全 PASS

---

## 阶段 10 — 固化依赖

```bash
source .venv/bin/activate
pip freeze > requirements-freeze.txt    # 已生成,57 行
```

复原:`pip install -r requirements-freeze.txt`(本架构无 torch 的 `+cu126` 后缀问题)。

---

## 常见问题(macOS 专属)

| 症状 | 原因 | 解决 |
|---|---|---|
| `Bootstrap failed: 5: Input/output error` | 在非图形会话里 `launchctl bootstrap` | 用 `scripts/start_ollama.sh`;或在你自己的终端 App 里 `brew services start ollama` |
| `ollama server not responding - could not find ollama app` | CLI 想拉起 Ollama.app,但 brew 装的是纯 CLI 版 | 先 `bash scripts/start_ollama.sh` 再跑 `ollama` 命令 |
| pip 装 faiss-cpu 去下载 `.tar.gz` 并要 cmake | 版本 ≥1.10 无 macOS 轮子 | `requirements.txt` 已锁 `<1.10`;手动装用 `pip install "faiss-cpu<1.10"` |
| `No module named 'cgi'` | 用了 Python 3.13+ | 必须用 3.12(官方包已在 `/Library/Frameworks`) |
| 敲 `python` 进了 conda base | Anaconda 在 PATH 前 | 项目根 `source .venv/bin/activate` |
| 首次 embed 慢(~20s) | 模型加载进统一内存 | 正常,`check_env.py` 已给 120s 超时 |

---

## 日常速查

```bash
cd /Users/program/LLM/agent/rag-from-scratch
source .venv/bin/activate     # 激活 venv
bash scripts/start_ollama.sh  # 确保向量服务在跑
python src/check_env.py       # 环境自检
deactivate                    # 退出 venv
```

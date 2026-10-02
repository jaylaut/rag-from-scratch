# -*- coding: utf-8 -*-
"""环境验收脚本 —— 一条命令确认"开发环境全部配置成功"

与 src/check_env.py 的分工:
    src/check_env.py      7 项核心自检(项目自带,跨 Windows/macOS 通用)
    本脚本                14 项"全景验收",额外覆盖:
                          venv 是否激活 / pip 源 / src 扁平导入 / faiss 检索往返 /
                          Ollama 算力分配 / wikiextractor CLI / 目录结构 / 磁盘 /
                          pytest 冒烟 / Qwen 真实调用(可选)

用法(项目根 · venv 已激活):
    python scripts/verify_setup.py               # 全量验收(不花钱)
    python scripts/verify_setup.py --skip-embed  # 跳过向量推理(最快)
    python scripts/verify_setup.py --live        # 额外真实调一次 Qwen(会花几分钱)
    python scripts/verify_setup.py --json        # 机器可读输出

更省事:
    bash scripts/verify.sh                       # 自动激活 venv + 拉起 Ollama + 跑本脚本

退出码:0 = 全部通过(允许 WARN);1 = 有 FAIL;2 = 脚本自身异常
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PASS, FAIL, WARN, SKIP = "[PASS]", "[FAIL]", "[WARN]", "[SKIP]"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# 让 `from common import config` 这类扁平导入生效(项目约定:src 本身在 sys.path 上)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


class WarnOnly(Exception):
    """检查项"没通过但不算致命",只记 WARN 不影响退出码"""


class Skipped(Exception):
    """检查项被命令行参数主动跳过"""


def check(name: str, fn):
    """执行一个检查项:异常即判 FAIL,并把异常信息作为修复提示"""
    try:
        detail = fn()
    except Skipped as e:
        record(SKIP, name, str(e))
        return
    except WarnOnly as e:
        record(WARN, name, str(e))
        return
    except Exception as e:  # noqa: BLE001 - 验收脚本就是要兜住所有异常
        record(FAIL, name, f"{type(e).__name__}: {e}")
        return
    record(PASS, name, detail or "")


# ---------------------------------------------------------------- 1. 解释器
def check_interpreter():
    v = sys.version_info
    assert v[:2] == (3, 12), f"当前 {v.major}.{v.minor},项目约定 3.12(wikiextractor 需 cgi 模块)"
    in_venv = sys.prefix != sys.base_prefix
    assert in_venv, (
        "没在虚拟环境里运行。修复: cd 项目根 && source .venv/bin/activate"
    )
    return f"Python {v.major}.{v.minor}.{v.micro} @ {sys.prefix}"


def check_venv_dir():
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    assert venv_py.exists(), "项目根缺少 .venv,修复: python3.12 -m venv .venv"
    assert Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve(), (
        f"当前解释器不在项目 .venv 里({sys.prefix})"
    )
    return f"已绑定项目虚拟环境 {PROJECT_ROOT / '.venv'}"


# ---------------------------------------------------------------- 2. 依赖
def check_dependencies():
    import faiss  # noqa: PLC0415
    import numpy  # noqa: PLC0415
    import requests  # noqa: PLC0415
    import tiktoken  # noqa: PLC0415
    import jieba  # noqa: PLC0415
    import wikiextractor  # noqa: PLC0415
    import datasets  # noqa: PLC0415
    import dashscope  # noqa: PLC0415
    import dotenv  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    assert numpy.__version__.startswith("1."), (
        f"numpy {numpy.__version__} 与 faiss-cpu 冲突,修复: pip install 'numpy<2'"
    )
    return (
        f"faiss {faiss.__version__} | numpy {numpy.__version__} | "
        f"dashscope {dashscope.__version__} | datasets {datasets.__version__} | "
        f"tiktoken/jieba/wikiextractor/dotenv/pytest 均可导入"
    )


def check_pip_index():
    out = subprocess.run(
        [sys.executable, "-m", "pip", "config", "list"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    if "tuna" in out:
        return f"已配置清华源({out.strip().splitlines()[0]})"
    raise AssertionError(
        "pip 未配置国内镜像,下载会超时。修复: "
        "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    )


# ---------------------------------------------------------------- 3. 项目代码可导入
def check_src_importable():
    from common import config  # noqa: PLC0415 - 验证 src 扁平导入链路

    missing = [p for p in ("OLLAMA_URL", "EMBED_MODEL", "EMBED_DIM") if not hasattr(config, p)]
    assert not missing, f"config.py 缺少常量: {missing}"
    assert config.EMBED_DIM == 1024, f"EMBED_DIM={config.EMBED_DIM},期望 1024(bge-m3)"
    return f"from common import config 成功 | EMBED_MODEL={config.EMBED_MODEL} EMBED_DIM={config.EMBED_DIM}"


# ---------------------------------------------------------------- 4. Ollama
def _ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST") or "http://localhost:11434"


def check_ollama_service():
    import requests  # noqa: PLC0415

    r = requests.get(f"{_ollama_url()}/api/version", timeout=5)
    r.raise_for_status()
    return f"Ollama {r.json().get('version', '?')} @ {_ollama_url()}"


def check_model_pulled():
    import requests  # noqa: PLC0415
    from common import config  # noqa: PLC0415

    r = requests.get(f"{_ollama_url()}/api/tags", timeout=10)
    names = [m["name"] for m in r.json().get("models", [])]
    ok = any(n == config.EMBED_MODEL or n.startswith(config.EMBED_MODEL + ":") for n in names)
    assert ok, (
        f"未找到 {config.EMBED_MODEL}。修复: ollama pull {config.EMBED_MODEL}\n"
        f"  当前已有: {names}"
    )
    return f"{config.EMBED_MODEL} 已就绪(模型库共 {len(names)} 个)"


def check_embed(skip: bool):
    if skip:
        raise Skipped("已跳过(--skip-embed)")
    import requests  # noqa: PLC0415
    from common import config  # noqa: PLC0415

    t0 = time.time()
    r = requests.post(
        f"{_ollama_url()}/api/embed",
        json={"model": config.EMBED_MODEL, "input": ["环境验收测试"]},
        timeout=config.EMBED_TIMEOUT,
    )
    r.raise_for_status()
    vec = r.json()["embeddings"][0]
    assert len(vec) == config.EMBED_DIM, f"维度异常 {len(vec)},期望 {config.EMBED_DIM}"
    norm = sum(x * x for x in vec) ** 0.5
    return f"维度 {len(vec)} | 范数 {norm:.4f} | 首次加载耗时 {time.time() - t0:.1f}s"


def check_ollama_gpu():
    """确认推理跑在 GPU 上(Mac = Metal,Windows = NVIDIA);查不到只 WARN 不判失败"""
    exe = shutil.which("ollama") or "/opt/homebrew/opt/ollama/bin/ollama"
    if not Path(exe).exists():
        raise WarnOnly("未找到 ollama 可执行文件,跳过")
    out = subprocess.run([exe, "ps"], capture_output=True, text=True, timeout=60).stdout
    gpu_lines = [l.strip() for l in out.splitlines() if "GPU" in l]
    if gpu_lines:
        # 列格式:NAME ID SIZE(数值+单位) PROCESSOR CONTEXT UNTIL
        # → 直接抓 "GPU" 前一个 token 拼出 "100% GPU",比按固定列号切更稳
        desc = []
        for l in gpu_lines:
            tk = l.split()
            i = tk.index("GPU")
            desc.append(f"{tk[0]} → {' '.join(tk[i - 1:i + 1])}")
        return "GPU 加速生效 —— " + " ; ".join(desc)
    raise WarnOnly(
        f"未检测到 GPU 占用(可能模型已卸载)。可先跑一次 embed 再 `ollama ps` 复看。"
        f"当前输出: {out.strip() or '(空)'}"
    )


# ---------------------------------------------------------------- 5. faiss 检索往返
def check_faiss_roundtrip():
    import faiss  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from common import config  # noqa: PLC0415

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, config.EMBED_DIM)).astype("float32")
    faiss.normalize_L2(vecs)                       # IndexFlatIP 前必须 L2 归一化
    index = faiss.IndexFlatIP(config.EMBED_DIM)
    index.add(vecs)
    assert index.ntotal == 4, f"add 后 ntotal={index.ntotal}"
    d, i = index.search(vecs[2:3], 1)
    assert i[0][0] == 2, f"自查失败:第 3 条向量最近邻是 {i[0][0]}"
    return f"IndexFlatIP({config.EMBED_DIM}) add 4 条 → 检索 top1 命中自身,相似度 {d[0][0]:.4f}"


# ---------------------------------------------------------------- 6. Key 与 Qwen
def check_api_key():
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(PROJECT_ROOT / ".env")
    env_file = PROJECT_ROOT / ".env"
    assert env_file.exists(), f"缺少 {env_file}。修复: cp .env.example .env 并填入真实 Key"
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    assert key and not key.startswith("sk-xxx"), (
        "DASHSCOPE_API_KEY 还是占位值。修复: 编辑 .env 填入百炼真实 Key"
        "(https://bailian.console.aliyun.com/)"
    )
    return f".env 已就位,Key = {key[:6]}****"


def check_qwen_live(live: bool):
    if not live:
        raise Skipped("已跳过(加 --live 才会真实请求,约花几分钱)")
    import dashscope  # noqa: PLC0415
    from dashscope import Generation  # noqa: PLC0415
    from common import config  # noqa: PLC0415

    r = Generation.call(
        model=config.QWEN_MODEL,
        prompt="用一句话介绍维基百科",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    assert r.output is not None and r.output.text, f"调用返回异常: {r}"
    return f"{config.QWEN_MODEL} 返回:{r.output.text[:40]}..."


# ---------------------------------------------------------------- 7. 工具链与工程结构
def check_wikiextractor_cli():
    import wikiextractor  # noqa: PLC0415

    exe = shutil.which("wikiextractor")
    detail = f"包路径 {Path(wikiextractor.__file__).parent}"
    if exe:
        return f"{detail} | 命令行 {exe}"
    raise AssertionError(f"{detail} —— 未找到 wikiextractor 命令行入口")


def check_layout():
    required = [SRC_DIR, PROJECT_ROOT / "tests", PROJECT_ROOT / "scripts",
                PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "pytest.ini"]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"缺失:{missing}"
    # 数据目录会被流水线写入,提前确认可写
    # (先 exists() 再 mkdir:某些受限终端对"目录已存在时的 mkdir"会误报 EEXIST)
    for d in ("data", "index"):
        target = PROJECT_ROOT / d
        if not target.exists():
            target.mkdir(parents=True)
        testfile = target / ".write_test"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink()
    return "src/ tests/ scripts/ requirements.txt pytest.ini 齐备;data/ index/ 可写"


def check_disk():
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    assert free_gb >= 10, (
        f"仅剩 {free_gb:.0f}GB。维基 dump 解压后约 8GB + 索引若干 GB,至少留 10GB"
    )
    return f"剩余 {free_gb:.0f} GB"


def check_pytest():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()][-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        raise AssertionError(f"pytest 退出码 {proc.returncode}\n{tail}\n{proc.stdout[-800:]}")
    return f"pytest 全绿 —— {tail}"


# ---------------------------------------------------------------- 汇总输出
def render(title: str) -> str:
    width = 66
    lines = ["", "=" * width, f"  {title}", "=" * width]
    for status, name, detail in results:
        lines.append(f"{status} {name}")
        if detail:
            for seg in str(detail).splitlines():
                lines.append(f"        {seg}")
    n_fail = sum(1 for r in results if r[0] == FAIL)
    n_warn = sum(1 for r in results if r[0] == WARN)
    n_skip = sum(1 for r in results if r[0] == SKIP)
    n_pass = sum(1 for r in results if r[0] == PASS)
    lines.append("-" * width)
    lines.append(f"  PASS {n_pass} | FAIL {n_fail} | WARN {n_warn} | SKIP {n_skip}")
    if n_fail == 0:
        lines.append("  ✅ 环境配置全部验证通过,可以开始下一步:下载维基百科 dump。")
    else:
        lines.append(f"  ❌ 有 {n_fail} 项失败,按上面的提示修复后重跑本脚本。")
    lines.append("=" * width)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 项目开发环境全景验收")
    ap.add_argument("--skip-embed", action="store_true", help="跳过向量推理(最快)")
    ap.add_argument("--live", action="store_true", help="额外真实调一次 Qwen API(会花钱)")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    check("Python 3.12 + 虚拟环境", check_interpreter)
    check(".venv 位于项目根", check_venv_dir)
    check("依赖全部可导入", check_dependencies)
    check("pip 国内镜像", check_pip_index)
    check("src 扁平导入链路", check_src_importable)
    check("Ollama 服务可达", check_ollama_service)
    check("bge-m3 已拉取", check_model_pulled)
    check("bge-m3 向量推理", lambda: check_embed(args.skip_embed))
    check("Ollama 算力分配", check_ollama_gpu)
    check("faiss 检索往返", check_faiss_roundtrip)
    check("DASHSCOPE_API_KEY 已配置", check_api_key)
    check("Qwen API 真实调用", lambda: check_qwen_live(args.live))
    check("wikiextractor 可用", check_wikiextractor_cli)
    check("工程目录结构", check_layout)
    check("磁盘余量 ≥ 10GB", check_disk)
    check("pytest 冒烟", check_pytest)

    n_fail = sum(1 for r in results if r[0] == FAIL)
    if args.json:
        print(json.dumps(
            [{"status": s, "item": n, "detail": d} for s, n, d in results],
            ensure_ascii=False, indent=2,
        ))
    else:
        print(render("RAG 开发环境验收结果"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(2)

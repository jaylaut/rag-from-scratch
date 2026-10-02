# -*- coding: utf-8 -*-
"""
环境自检脚本 — 逐项验证 RAG 开发环境是否配置成功
架构:Ollama(bge-m3 向量) + faiss-cpu(检索) + Qwen API(生成)
用法(venv 激活后):
    python src/check_env.py            # 完整检查
    python src/check_env.py --skip-embed   # 跳过向量推理测试(快速检查)
"""
import os
import sys

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024  # bge-m3 输出 1024 维

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((PASS, name, detail))
    except Exception as e:
        results.append((FAIL, name, str(e)))


def main():
    skip_embed = "--skip-embed" in sys.argv

    # 0. 加载 .env(如果存在)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        results.append((WARN, "python-dotenv 未安装", "requirements.txt 中有,建议安装"))

    # 1. Python 版本
    def c_python():
        v = sys.version_info
        assert v[:2] == (3, 12), f"当前 {v.major}.{v.minor},项目约定 3.12"
        return f"Python {v.major}.{v.minor}.{v.micro}"
    check("Python 版本 = 3.12", c_python)

    # 2. Ollama 服务
    def c_ollama():
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/version", timeout=5)
        r.raise_for_status()
        return f"Ollama {r.json().get('version', '?')} @ {OLLAMA_HOST}"
    check("Ollama 服务可达", c_ollama)

    # 3. 向量模型已拉取
    def c_model_pulled():
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        assert any(n == EMBED_MODEL or n.startswith(EMBED_MODEL + ":") for n in names), (
            f"未找到 {EMBED_MODEL}。修复: ollama pull {EMBED_MODEL}\n"
            f"  当前已有: {names}"
        )
        return f"{EMBED_MODEL} 已就绪"
    check(f"{EMBED_MODEL} 已拉取", c_model_pulled)

    # 4. 向量推理测试(首次调用需把模型加载进显存,可能稍慢)
    def c_embed():
        import requests
        r = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": ["环境自检测试"]},
            timeout=120,
        )
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        assert len(vec) == EMBED_DIM, f"维度异常: {len(vec)} (期望 {EMBED_DIM})"
        norm = sum(x * x for x in vec) ** 0.5
        return f"维度 {len(vec)} | 向量范数 {norm:.4f}(FAISS 建库前需 L2 归一化)"
    if skip_embed:
        results.append((WARN, "向量推理测试", "已跳过(--skip-embed)"))
    else:
        check(f"{EMBED_MODEL} 向量推理", c_embed)

    # 5. faiss
    def c_faiss():
        import faiss
        idx = faiss.IndexFlatIP(EMBED_DIM)
        return f"faiss {faiss.__version__},IndexFlatIP({EMBED_DIM}) 创建成功"
    check("faiss-cpu 可用", c_faiss)

    # 6. Qwen API Key
    def c_key():
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        assert key and not key.startswith("sk-xxx"), (
            "未检测到有效的 DASHSCOPE_API_KEY。\n"
            "  修复:复制 .env.example 为 .env 并填入真实 Key"
        )
        return f"Key 已配置({key[:6]}****)"
    check("DASHSCOPE_API_KEY 已配置", c_key)

    # 7. wikiextractor
    def c_wiki():
        import wikiextractor
        return "wikiextractor 可导入"
    check("wikiextractor 可用", c_wiki)

    # 汇总
    print("\n" + "=" * 60)
    print("RAG 开发环境自检结果")
    print("=" * 60)
    for status, name, detail in results:
        print(f"{status} {name}")
        if status != PASS or detail:
            print(f"       {detail}")
    n_fail = sum(1 for r in results if r[0] == FAIL)
    print("=" * 60)
    if n_fail == 0:
        print("全部通过!可以开始下一步:下载维基百科 dump。")
    else:
        print(f"有 {n_fail} 项失败,按提示修复后重跑。")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

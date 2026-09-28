# -*- coding: utf-8 -*-
"""全局配置与路径常量 —— 单一事实来源

约定:所有模块的路径和参数一律从本文件导入,不写死。
     路径基于本文件位置自动推算,与 cwd 无关(在任何目录运行都正确)。
"""
from pathlib import Path

# 项目根 = src/common/config.py 往上两级(common → src → 项目根)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- 数据目录(离线流水线三级产物)----
DATA_RAW = PROJECT_ROOT / "data" / "raw"          # 维基 dump(.xml.bz2)
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"  # 清洗后文章 JSONL
DATA_CHUNKS = PROJECT_ROOT / "data" / "chunks"    # 切片 JSONL
DATA_EVAL = PROJECT_ROOT / "data" / "eval"        # 评测集(D8)
INDEX_DIR = PROJECT_ROOT / "index"                # FAISS 索引 + 元数据
DATA_ARTICLES = PROJECT_ROOT / "data" / "articles"          # 二次清洗分片 + 合并产物目录
ARTICLES_FILE = DATA_ARTICLES / "wiki_zh.jsonl"             # D1 全量合并产物,本模块的输入
CHUNKS_FILE = DATA_CHUNKS / "wiki_chunks.jsonl"             # 全量切片产物
CHUNKS_SUBSET_FILE = DATA_CHUNKS / "wiki_chunks_subset.jsonl"  # 调试子集(前 10 万篇)

# ---- 切片参数(D2 定稿:450~550 弹性区间,完整性优先于精确 500)----
CHUNK_MIN_CHARS = 450       # 下限:当前块不足此数时必须继续加句(允许暂时突破上限)
CHUNK_MAX_CHARS = 550       # 上限:下一句放不进来就封块(原 500 的语义升级)
CHUNK_OVERLAP = 50          # 重叠下限(候选切点 = 上块尾部恰好 50 字处)
CHUNK_OVERLAP_MAX = 100     # 重叠上限(句对齐回退的最远距离,超过则放弃对齐)
CHUNK_HARDCUT_CHARS = 500   # 单句超过 CHUNK_MAX_CHARS 时,硬切的目标片长
CHUNK_TAIL_MERGE = 100      # 尾块不足此数时并入前块(防止碎片块)
CHUNK_MERGE_LIMIT = 650     # 尾块并入后的块长上限(防止合并出超长块)
# ---- Ollama 向量服务 ----
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024          # bge-m3 输出维度;换模型必须同步改,且索引需重建
EMBED_TIMEOUT = 300       # 首次调用需把模型加载进显存,给足余量


# ---- Qwen 生成 ----
QWEN_MODEL = "qwen-plus"
QWEN_SYSTEM_PROMPT = (
    "你是一个基于参考资料回答问题的助手。仅根据参考资料回答,"
    "引用时标注 [来源N];资料不足以回答时明确说明。"
)

# ---- 在线检索参数(两层口径:召回宽、精读窄)----
RECALL_K = 50   # 向量检索召回数(宽口径)
TOP_M = 5       # 进 prompt 的来源数(窄口径)

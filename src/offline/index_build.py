# -*- coding: utf-8 -*-
"""D4 - index_build.py:建 FAISS 索引(离线流水线终点)

职责(自底向上四层):
    第 1 层  _fmt_bytes / _default_paths / _infer_rows / _open_vectors / _audit_vectors /
             _new_index / _add_shards / _write_index / _iter_meta_rows / _write_meta /
             _count_rows / _read_index / _verify_alignment
             原子能力:路径解析、memmap 打开、向量体检、索引构建与落盘、元数据落盘、对齐校验
    第 2 层  build_index / build_subset_index / build_full_index
             主流程编排 + 两套产物(subset / full)的一层壳
    第 3 层  _sample_row_ids / _fetch_meta_rows / _embed_query / sanity_check
             建库自检:库内已有 chunk 重新 embed → 检索 → top1 必须是自己
    第 4 层  __main__
             命令行入口(--full / --limit N / --no-sanity / --no-verify / --k / --n)

全模块唯一的约束:FAISS 第 i 行向量 ↔ chunks_meta.jsonl 第 i 行 ↔ chunks 文件第 i 行。
   三者一旦错位,检索会把"A 的向量"配上"B 的文本",而且【不报错】——典型的静默降级。

对应:路线计划 D4 | 架构 §3.5 / §5 | 开发指引 D4建索引开发指引.md
运行:python -m offline.index_build                (建 subset 索引,默认带自检)
     python -m offline.index_build --full         (D7 全量)
     python -m offline.index_build --limit 2000   (只取前 2000 行,秒级冒烟)
"""
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import faiss                # faiss-cpu:索引的构建/罗盘/检索全靠它
import numpy as np

from common import config
from common.utils import iter_jsonl   #jsonl 流式读

# 模块级 logger
logger = logging.getLogger(__name__)

_ADD_SHARD = 200_000        # 每次index.add()行数，20万行 * 1024 * 4B = 800MB
_META_WRITE_BATCH = 10_000      # 元数据凑足1万行，才真正写盘
_META_HEARTBEAT = 200_000        # 元数据每次写满20万行，输入一条进度日志
_AUDIT_SAMPLES = 200            # 入库前，随机抽200行做 向量检查
_NORM_TOL = 1e-3                # 范数容差
_ZERO_TOL = 1e-6                # 判定 ‘零向量’的范数下限
_MIN_SELF_SCORE = 0.999         # 自检：自己检索自己的余弦下限
_SEED = 42                      # 固定随机种子


def _fmt_bytes(n: float) -> str:
    '''
    把字节数转换成实际容易理解的字符串，例如：1024 -> 1kb
    参数 n : 字节数 (允许传float 或者 int)
    返回 ： 例如 '812.23MB'
    '''

    # 转成浮点数，避免后面小数丢失
    v = float(n)
    for unit in ('B','KiB','MiB','GiB','TiB'):
        # 1024 进制，不是 1000 进制
        # 当前数值落在这里，就可以输出了，用abs()是为了防止传入负数也不会出错
        if abs(v) < 1024.0:
            # :.1f 表示 "保留1位小数的浮点数"
            return f"{v:.1f} {unit}"

        # 除完后继续下一轮循环
        v /= 1024.0
    #兜底，本项目不会超过1PiB
    return f"{v:.1f} PiB"

def _default_paths(full: bool = False) -> tuple[Path,Path,Path,Path]:
    '''
    一次给出"四件套"路径:(向量文件, 切片文件, 索引文件, 元数据文件)。

    为什么要把四个路径绑成一个函数返回?
        因为这四个必须【成套使用】。如果拿全量的向量去配子集的 chunks,
        行数立刻对不上。所以设计上就不允许调用方一个一个拼,要么整套走默认,
        要么自己在 build_index 里整套覆盖。

    两套的含义:
        full=True  → 全量:整个开发流程验证无误后，全量跑,索引约 9.35 GiB
        full=False → 调试子集:开发期全程用这个,秒级/分钟级，节省时间

    参数:
        full: True 用全量四件套,False 用调试子集四件套(默认 False,防止手滑跑全量)
    返回:
        四元组 (vectors_path, chunks_path, index_path, meta_path)

    明确不做:
        不检查文件是否存在 —— 那是 build_index 开头的事("先检查、后打开")。
    '''
    if full:
        return (
            config.VECTORS_FILE,
            config.CHUNKS_FILE,
            config.INDEX_FILE,
            config.META_FILE
        )
    return (
        config.VECTORS_SUBSET_FILE,
        config.CHUNKS_SUBSET_FILE,
        config.INDEX_SUBSET_FILE,
        config.META_SUBSET_FILE
    )

def _infer_rows(vectors_path: Path, dim: int,expect_rows:int | None = None) -> int:
    '''
    推断(或校验)向量矩阵到底有多少行 —— 【本模块最容易写错的一处】。

    ────────────────────────────────────────────────────────────────────
    原理讲解:为什么不能简单地用 "文件大小 ÷ 每行字节数" 来算行数
    ────────────────────────────────────────────────────────────────────
    错误写法是:行数 = 文件大小 / (dim × 4)。实际这种并不准确
    举个具体反例:假设文件是 3,072,000 字节 ——
        · 按 dim=768 解读: 3,072,000 / (768×4) = 1000 行  ，没问题
        · 按 dim=1024 解读: 3,072,000 / (1024×4) = 750 行  ，没问题
    看起来合理，如果换了 embedding 模型(比如换成 768 维的)
    却忘了改 config.EMBED_DIM,程序不会报错,它仍会建出一个维度错误的索引。

    所以本函数的策略是分两条路:
        · 给了 expect_rows → 它是实际(来自 chunks 文件数出来的行数),
          用它当答案,并且反过来断言 "文件字节数必须 == rows × dim × 4",
          对不上立刻 raise,把隐患变成当场报错;
        · 没给 expect_rows → 才退化成整除推断(只在单测/手工调用时走这条路)。

    参数:
        vectors_path: 向量文件路径
        dim:          向量维度(应当等于 config.EMBED_DIM)
        expect_rows:  期望行数;None 未确认

    返回:int —— 行数
    抛出:ValueError —— 文件为空 / 字节数对不上 / 与 expect_rows 矛盾
    '''
    #stat() 读取文件的元信息(不打开文件内容),st_size 就是字节数
    size = vectors_path.stat().st_size

    #每行字节数：1024 维度 * float32(4字节) = 4096字节 = 4KB
    item = dim * 4

    # 分支1：已知期望行数，走‘校验’模式
    if expect_rows is not None:
        #行数必须是正整数，且不能位0，如果是0，则说明chunks 文件是空的或者报错
        if expect_rows <= 0:
            raise ValueError(f"期望行数必须是正整数，而实际是{expect_rows}")

        # 按这个行数算出来的是‘应该有的字节数’
        want = expect_rows * dim

        # 关键断言：实际字节数必须严格等于理论字节数
        # 不能有任何差异，否则说明维度配错 / 行数不等 / 文件格式不对
        if size != want:
            # 报错信息要详细，保证尽量知道错误原因
            raise ValueError(
                f"向量文件字节数与期望不符:{vectors_path}\n"
                f"  实际 {size} 字节,按 {expect_rows} 行 × {dim} 维应为 {want} 字节。\n"
                f"  常见原因:① config.EMBED_DIM({dim})与生成该文件的模型维度不一致;\n"
                f"            ② 向量文件行数与 chunks 行数不等(D3 是否完整跑完?);\n"
                f"            ③ 该文件是 np.save 产物(带 npy 文件头),而本模块按原始格式读取。"
            )
        #校验通过，行数确定
        return expect_rows

    # 分支2：没有期望行数，退化为整除推断
    # 空文件，或者字节数不能被 ‘每行字节数’ 整除 -> 说明不是希望的原始矩阵
    if size == 0 or size % item != 0:
        raise ValueError(
            f"向量文件不是 {dim} 维 float32 矩阵的原始 memmap:{vectors_path}"
            f"({size} 字节,不是 {item} 的整数倍)"
        )
    # 能整除，保证返回整数
    return size // item

def _open_vectors(vectors_path: Path,
                  dim: int = config.EMBED_DIM,
                  expect_rows: int | None = None) -> np.memmap:
    """以【只读】方式打开向量矩阵,拿到一个形状为 (N, dim) 的 float32 数组视图。

        ────────────────────────────────────────────────────────────────────
        原理讲解:为什么用 np.memmap 而不是 np.load / np.fromfile
        ────────────────────────────────────────────────────────────────────
        全量向量矩阵是 245 万行 × 1024 维 × 4 字节 ≈ 9.35 GB。
        机器内存 16 GB,如果整个读进来(np.load),内存直接吃紧甚至爆掉。
        np.memmap 的做法是"把磁盘文件映射成一块看起来像内存的数组":
            · 你写 vecs[0] 时,操作系统才去磁盘读那一页进来;
            · 你写 vecs[100:200] 时,才读那 100 行所在的页;
            · 进程内存占用始终是 O(几 MB),跟文件多大没关系。
        而且 FAISS 的 add() 是按行区间拷贝的,适合"一片一片喂"。

        参数:
            vectors_path: 向量文件路径
            dim:          向量维度
            expect_rows:  期望行数(传给 _infer_rows 做校验)
        返回:
            np.memmap —— 形状 (N, dim),dtype=float32,只读(不能赋值)

        明确不做:
            不做归一化、不改任何数据 —— 那是 embed.py 的功能。本模块只搬运,不做加工。
        """
    # 先检查文件存不存在，放在所有操作之前
    if not vectors_path.is_file():
        raise FileNotFoundError(
            f"向量文件不存在:{vectors_path}\n请先执行embed.py"
        )
    if dim <= 0:
        raise ValueError(f"维度必须为整数，收到{dim}")

    # 行数由 _infer_rows 决定
    rows = _infer_rows(vectors_path,dim,expect_rows)

    #np.memmap 的四个关键参数：
    #   str(path) —— SWIG/numpy 的老接口不认 pathlib.Path,必须转成普通字符串
    #   dtype=np.float32 —— FAISS 的 add() 【只接受 float32】,其它类型会报错
    #   mode="r"  —— ★★ 只读!写成 "w+" 会在打开的一瞬间把 embed.py 几小时的产物清零
    #   shape=(rows, dim) —— memmap 必须预先声明形状,所以行数必须先算好
    mm = np.memmap(str(vectors_path),dtype=np.float32,mode='r',shape=(rows,dim))

    logger.info(
        '向量已映射:%s | %d 行 * %d 维 | %s(float32)',
        vectors_path,rows,dim,_fmt_bytes(rows * dim * 4)
    )
    return mm

def _audit_vectors(vecs: np.memmap,
                   n_samples: int = _AUDIT_SAMPLES,
                   seed: int = _SEED) -> dict:
    """入库前的"向量检查":随机抽若干行,检查能不能安全喂给 IndexFlatIP。

        ────────────────────────────────────────────────────────────────────
        原理讲解:为什么必须在 add() 之前体检(三类致命数据)
        ────────────────────────────────────────────────────────────────────
        ① 未归一化(最常见)
           IndexFlatIP 的分数语义 "内积 == 余弦相似度" 【依赖向量已归一化】。
           如果 embed.py 的归一化被改坏或漏了,FAISS 不报错、不崩溃,
           只是把"模长大的向量"(通常是长句)无理由排在前面。
           现象:检索结果答非所问,但一切指标看起来正常。→ 静默降级
        ② nan / inf(一旦进库,整库报废)
           任何向量只要有一个 nan,跟它算内积的结果就是 nan,
           排序时 nan 的行为未定义,可能导致任何查询的分数都是 nan。
        ③ 全零行(embed.py 断点续跑的残留)
           embed.py 用了"预先分配 np.memmap 再逐行填充"的写法。如果中途中断,
           文件尾部会留下一堆没写入的全零行。零向量余弦恒为 0,
           它们会莫名其妙地挤进 top-k。

        参数:
            vecs:      待检查的向量矩阵(memmap)
            n_samples: 抽样行数
            seed:      随机种子(固定种子 → 每次抽同样的行 → 结果可复现)
        返回:
            dict —— 检查结果,包含上述三类的检查结果
        """
    # 解包：总行数，维度
    rows,dim = vecs.shape

    # 不足200行时，就全抽取
    size = min(n_samples,rows)

    # np.random.default_rng(seed) 是 numpy推荐的现在随机数生成器，传入固定的seed -> 每次运行抽到完全相同的行号
    rng = np.random.default_rng(seed)

    # 从0...row -1 里不重复抽取  size 个
    idx = rng.choice(rows,size=size,replace=False)

    # 索引 vecs[idx] 会把这几行 真正读进内存：
    # 200 行 * 1024 维度 * 4 字节 = 800KB
    # np.asarray(...dtype=np.float32)确保类型干净
    sample = np.asarray(vecs[idx],dtype=np.float32)

    #np.isfinite(x)对Nan 和 ± inf 都返回false,一次覆盖两种情况
    # .all()表示 '所有元素都必须是有限值'
    finite = bool(np.isfinite(sample).all())

    # np.linalg.norm(sample,axis=1):对每一行求L2范数->向量长度
    # axis=1:沿着第一轴(列方向)求和  -> 逐行算
    # 归一化后，这里每个值都应该是 1.0
    norms = np.linalg.norm(sample,axis=1)

    # 组装检查结果
    result = {
        'rows': int(rows),
        'dim': int(dim),
        'sampled': int(size),
        'dtype': str(vecs.dtype),           # 必须是float32
        'c_contiguous': bool(vecs.flags['C_CONTIGUOUS']),   # add()前提：C 连续存储
        'finite': finite,                   # 没有Nan / inf 才算True
        'norm_min': float(norms.min()) if size else  0.0, # 抽样里最短的向量长度
        'norm_max': float(norms.max()) if size else 0.0,
        'zero_rows': int((norms < _ZERO_TOL).sum()),   # 范数 < 1e-6的行数（零向量）
        # 是否已归一化的判断：最短的 > 1 - 容差  且 最长的 < 1 + 容差
        'normalized': bool(
            size > 0                    # 至少抽到行
            and  finite                 # 不能有Nan / inf
            and norms.min() > 1.0 - _NORM_TOL   # 下限
            and norms.max() < 1.0 + _NORM_TOL   #  上限
        ),
    }
    #详细信息打印
    logger.info(
        "向量体检:抽样 %d 行 | dtype=%s C连续=%s | 有限值=%s | "
        "范数 %.6f~%.6f | 零行 %d | 已归一化=%s",
        result["sampled"], result["dtype"], result["c_contiguous"], result["finite"],
        result["norm_min"], result["norm_max"], result["zero_rows"], result["normalized"],
    )
    return result

def _new_index(dim: int) -> 'faiss.IndexFlatIP':
    """创建一个【空的】IndexFlatIP 索引。

        ────────────────────────────────────────────────────────────────────
        原理讲解:为什么是 FlatIP(两个词各代表一个决定)
        ────────────────────────────────────────────────────────────────────
        Flat = 暴力(brute force):
            查询时拿查询向量和库里【每一个】向量各算一遍相似度,取前 k 个。
            没有聚类、没有图、没有任何黑盒参数 —— 结果 100% 精确,100% 可解释。
            复杂度 O(N),但 N=245 万时一次查询只要几十毫秒,完全够用。
            学习阶段就该用这种完全透明的索引(ANN 索引留到 后面 做对照实验)。

        IP = Inner Product(内积):
            为什么不用距离?因为向量已 L2 归一化,此时:
                cos(θ) = (A·B) / (|A| × |B|) = (A·B) / (1 × 1) = A·B
            也就是说:内积【直接等于余弦相似度】。
            于是 search 返回的那个分数就是余弦,落在 [-1, 1],
            可以跨不同的查询互相比较,也可以设绝对阈值(比如 "> 0.7 才采纳")。

        四类索引的取舍(路线计划 D4 那张表):
            IndexFlatL2    暴力欧氏距离        精确,O(N)
            IndexFlatIP    暴力内积  ← 本项目   精确,O(N)
            IndexHNSWFlat  分层导航小世界图    近似,O(logN),快但有召回损失
            IndexIVFFlat   倒排+聚类          近似,O(√N),需训练

        参数:dim —— 向量维度(bge-m3 = 1024)
        返回:faiss.IndexFlatIP —— 空索引,此时 index.ntotal == 0
        """
    # FAISS内部就是维护一块(N,d)的float32 连续内存 + BLAS 矩阵乘
    # 内存消耗： N * d * 4字节 = N * 4kb
    index = faiss.IndexFlatIP(dim)
    logger.info('已创建 IndexFlatIP(dim=%d):暴力内积检索，归一化后分数即余弦相似度',dim)
    return index

def _add_shards(index: 'faiss.Index',vecs: np.memmap,shard: int=_ADD_SHARD) ->dict:
    """分片把向量喂进索引,并打印进度。

        ────────────────────────────────────────────────────────────────────
        原理讲解:为什么要分片
        ────────────────────────────────────────────────────────────────────
        index.add(x) 会把 x 【拷贝】进 FAISS 自己的存储空间。
        如果整块 9.35 GB 一次性 add,峰值内存 = mmap 页缓存 + 9.35 GB 拷贝 ≈ 19 GB,必爆。
        按 20 万行(≈800 MB)一片来喂,峰值只多出一片的内存。

        最重要的一点:分片【不改变结果】。
        FAISS 给向量的编号就是 add 的先后顺序,一片接一片喂和一次性喂,
        得到的编号完全一致(第 i 行的编号就是 i)。
        这是"索引第 i 行 ↔ meta 第 i 行"这条铁律能成立的前提。
        """
    # 总行数
    total = vecs.shape[0]

    # 计时
    t0 = time.perf_counter()
    #start=当前喂到第几行;n_shard = 已经喂了几片
    start,n_shard = 0,0

    while start < total:
        #本片的结束位置：最后一片要截取到total
        end = min(start + shard,total)

        # np.ascontiguousarray:确保内存是连续的 C 风格布局。
        # FAISS 的 add() 有两个硬性前提:① float32 ② C 连续。
        # memmap 的行切片本身已满足,这时本函数【返回同一个对象,不产生拷贝】;
        # 万一将来换成非连续的数据源,这一步能当场补好,省下大量排查时间。
        block = np.ascontiguousarray(vecs[start:end],dtype=np.float32)

        # 顺序即编号：第 i 片紧跟在第 i -1 片后面，编号连续，add()没有返回值，结果体现在index.ntotal上
        index.add(block)

        start = end
        n_shard += 1
        logger.info(
            "  已入库 %d/%d 行(%.1f%%) | 累计耗时 %.1fs",
            end, total, end / total * 100, time.perf_counter() - t0,
        )
    return {
        'added':int(index.ntotal),
        "add_sec": round(time.perf_counter() - t0, 2),
        'shards':n_shard,
    }


def _count_rows(path: Path) ->int:
    '''
    快速确认一个文本文件有多少行 -- 字节层面，不解析json
    不解析json是因为json.loads()，250万行就执行250万次 JSON解析。
    对于确认行数，通过数换行符最快
    '''
    # 累计统计到的行数
    total = 0

    # 记住最后一次读取的字符，b'' 是空的字符串(bytes)，不是字符串str - 因为是按二进制模式打开的文件
    last_byte = b''

    #open(path,'rb'):r = 只读，b = 二进制模式（不用UTF-8解码，直接度原始字节）
    # with ：离开这个代码块时自动关闭文件句柄，无需手写f.close()
    with open(path,'rb') as f:
        while True:



def build_index(vectors: "np.ndarray", meta_source: Path) -> None:
    """建 IndexFlatIP 并落盘索引 + 元数据。TODO: D4 实现"""
    raise NotImplementedError


def sanity_check(k: int = 5) -> None:
    """随机抽 3 个已有 chunk 验证 top1 命中自身。TODO: D4 实现"""
    raise NotImplementedError

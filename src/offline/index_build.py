# -*- coding: utf-8 -*-
"""D4 - index_build.py:建 FAISS 索引(离线流水线终点)

================================================================================
一句话职责
================================================================================
把 D3 产出的"向量矩阵"(只有数字)和 D2 产出的"切片文件"(只有文本),
变成两份能被在线检索直接加载的产物:
    ① index/*.faiss          —— FAISS 索引,只存向量数字
    ② index/chunks_meta.jsonl —— 元数据,存文本与溯源字段

================================================================================
本模块唯一的铁律:行序 1:1(看懂这一条,就看懂了半个模块)
================================================================================
FAISS 第 i 行向量  ↔  chunks_meta.jsonl 第 i 行  ↔  chunks 文件第 i 行。

这两份产物之间【唯一】的关联就是"行号"。一旦错位:
    不报错、不崩溃,只会把 "A 的向量" 配上 "B 的文本"
    → 答案张冠李戴,是最难查的一类线上故障。
由此推出四条硬性禁令:
    1. 不许排序   —— FAISS 的向量编号就是 add() 的先后顺序,顺序即主键
    2. 不许去重   —— 相邻块有 50~100 字重叠,重复是设计的一部分,不是脏数据
    3. 不许跳行   —— 行数对不上就 raise,绝不能"取较小值悄悄截断"
    4. 不许改 text —— meta 里的 text 一个字符都不许动(与 D2/D3 同一条铁律)

================================================================================
分层设计(四层,自底向上,调用方向自上而下)
================================================================================
第 1 层(原子能力,不涉及流程决策,可独立测试):
    _fmt_bytes / _default_paths / _infer_rows / _open_vectors / _audit_vectors
    _new_index / _add_shards / _write_index
    _iter_meta_rows / _write_meta / _count_rows / _read_index / _verify_alignment
第 2 层(编排 + 两套产物的一层壳):
    build_index / build_subset_index / build_full_index
第 3 层(建库自检,唯一需要 Ollama 的部分,刻意与建库主流程隔开):
    _sample_row_ids / _fetch_meta_rows / _embed_query / sanity_check
第 4 层(命令行入口):
    _read_args / __main__

================================================================================
★ 签名口径统一说明(照抄 D4指引 §5.14 的说明)
================================================================================
占位文件原来写的是 build_index(vectors: np.ndarray, meta_source),
路线计划写的是 build_index(vectors_path, chunks_path, index_dir)。
本模块统一为【传路径】:整份向量矩阵(设计上限档约 4 GB)
不可能整体当函数参数传,必须走"路径 + memmap"。

================================================================================
运行方式(macOS,先激活 venv,再 cd src)
================================================================================
python -m offline.index_build                 建 subset 索引(默认,带自检)
python -m offline.index_build --full          建全量索引(D7 用)
python -m offline.index_build --limit 2000    只取前 2000 行(秒级冒烟)
python -m offline.index_build --limit=2000    同上(等号写法也必须认)
python -m offline.index_build --no-sanity     跳过自检(Ollama 未起 / D3 未完成时)
python -m offline.index_build --no-verify     跳过落盘后的回读校验(不建议)
python -m offline.index_build --k 10 --n 5    自检时取 top-10、抽 5 个样本
"""
import json      # 标准库:把 dict 序列化成 JSON 字符串(json.dumps),写 meta 用
import logging   # 标准库:打日志。注意 basicConfig 只能在 __main__ 里配,见文件末尾
import random    # 标准库:自检时抽行号(random.Random().sample),不想为抽 3 个整数引入 numpy
import sys       # 标准库:sys.argv 读命令行参数;sys.exit(1) 让脚本能带失败码退出
import time      # 标准库:time.perf_counter() 高精度计时,统计各阶段耗时
from pathlib import Path      # 标准库:面向对象的路径对象,比字符串拼路径安全(跨 Windows/Mac)
from typing import Iterator   # 标准库:类型注解用,标明 _iter_meta_rows 是"生成器"

import faiss                # 第三方:faiss-cpu。索引的创建/写入/读取/检索全靠它
import numpy as np          # 第三方:向量矩阵的唯一载体(memmap / ascontiguousarray / norm 等)

from common import config           # 项目内:全部路径与参数常量的单一事实来源
from common.utils import iter_jsonl # 项目内:JSONL 逐行流式读的生成器(内存 O(1))

# 模块级 logger
logger = logging.getLogger(__name__)

#   每次 index.add() 喂多少行向量。
#   一行 = 1024 维 × 4 字节 = 4 KB,5 万行 ≈ 200 MB。
_ADD_SHARD = 50_000

#   元数据攒够多少行才真正写一次盘。
#   原因:一行一写是 20 万次函数调用("写"这个动作本身的开销被放大 20 万倍),
#   攒成 1 万行一批后,写次数降到 20 次左右。
_META_WRITE_BATCH = 10_000

# 元数据每次写满20万行，输入一条进度日志
_META_HEARTBEAT = 200_000

#   入库前随机抽多少行做"向量体检"(见 _audit_vectors)。
#   200 行 × 1024 维 × 4 字节 ≈ 800 KB,读进内存毫无压力。
_AUDIT_SAMPLES = 200            # 入库前，随机抽200行做 向量检查

#   范数(向量长度)容差。
#   为什么是 1e-3 这么松:float32 下归一化后每行模长的严格值是 1 ± 1e-7,
#   这里放宽到 1e-3 只为抓住"压根没归一化"这种【量级】错误,不是卡精度。
_NORM_TOL = 1e-3

#   判定"零向量"的范数下限:模长小于它就算全零行。
_ZERO_TOL = 1e-6

#   自检时"自己检索自己"的余弦相似度下限(见 sanity_check)。
#   归一化后,同一段文本再次编码,自己跟自己的余弦应当 ≈ 1.0。
_MIN_SELF_SCORE = 0.999         # 自检：自己检索自己的余弦下限

#   固定随机种子。架构原则 P4「一切可复现」:固定种子 → 每次抽到同一批行号
_SEED = 42


def _fmt_bytes(n: float) -> str:
    """把字节数变成人能读的字符串。

    ===========================================================================
    功能
    ===========================================================================
    838860800 → "0.8 GiB",而不是让人去数有几个零。

    ===========================================================================
    实现原理
    ===========================================================================
    循环除以 1024,直到数值落在 [0, 1024) 区间内,就用当前单位输出。
    刻意保留一位小数:监控体积变化时,需要能看出"增量"而不只是量级。
     是 1024 进制不是 1000 进制:磁盘/内存的行业习惯就是 1024,
      用 KiB/MiB/GiB 这套写法本身就是在提醒读者"这是 1024 进制"。

    参数:
        n: float | int —— 字节数(允许 float,因为调用方常拿估算值来调)
    返回:
        str —— 例如 "78.1 MiB" / "0.8 GiB"
    明确不做:
        不做 SI 单位(1000 进制的 KB/MB/GB)。
    """


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
    """按"全量 / 调试子集"一次给出 (向量, 切片, 索引, 元数据) 四件套路径。

        ===========================================================================
        实现原理
        ===========================================================================
        四件套必须【成套】使用:拿全量的向量配子集的 chunks,行数立刻对不上。
        所以设计上不允许调用方一个一个拼路径 —— 要么整套走默认,要么整套覆盖。
        两套产物的名字必须不同:否则 D7 一次全量建库就把调试产物冲掉了(反之亦然)。

        调用的外部对象(config.py 里的常量,全是 pathlib.Path):
            config.VECTORS_FILE        D3 全量向量产物(中间产物)
            config.VECTORS_SUBSET_FILE D3 调试子集向量产物
            config.CHUNKS_FILE         D2 全量切片产物
            config.CHUNKS_SUBSET_FILE  D2 调试子集切片产物
            config.INDEX_FILE          本模块全量索引产物(交付物)
            config.META_FILE           本模块全量元数据产物(交付物)
            config.INDEX_SUBSET_FILE   本模块调试子集索引产物
            config.META_SUBSET_FILE    本模块调试子集元数据产物

        参数:
            full: bool —— True 取全量四件套;False(默认)取调试子集四件套。
                  默认 False 是故意的:防止手滑跑了全量。
        返回:
            tuple —— (vectors_path, chunks_path, index_path, meta_path)
        明确不做:
            不检查文件是否存在 —— 那是 build_index 开头"先检查后开"的职责。
        """
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
    """推断(或【校验】)向量矩阵有多少行 —— 本模块防止"维度配错"的唯一防线。

        ===========================================================================
        ★ 实现原理(本模块最值钱的一处设计,见 D4指引 §3.6)
        ===========================================================================
        朴素写法:"行数 = 文件字节数 ÷ (dim × 4)"。但它是【猜】:
            3,072,000 字节 = 1000 行 × 768 维 × 4B
                           =  750 行 × 1024 维 × 4B   ← 两种解释都整除!
        换模型后 config.EMBED_DIM 忘了从 1024 改成 768,这个猜法会
        静默建出一个形状错误的索引 —— 能写完、能检索、能出结果,只是【全错】。

        正确做法:行数的权威来自 chunks 文件(它是事实),拿到行数后
        反过来【断言】向量文件字节数必须 == rows × dim × 4,对不上立刻 raise。
        这就是把"静默错误"换成"当场报错"。

        调用的外部方法:
            Path.stat().st_size —— 取文件元信息里的字节数,【不打开文件、不读内容】

        参数:
            vectors_path: Path —— 向量文件路径
            dim:          int  —— 向量维度,应当等于 config.EMBED_DIM(维度即契约)
            expect_rows:  int | None —— 期望行数(来自 chunks 文件数出来的行数);
                          None 表示未知,只走"退化推断"分支(仅单测/手工调用会走)
        返回:
            int —— 行数
        抛出:
            ValueError —— 文件为空 / 字节数对不上 / 与 expect_rows 矛盾
        明确不做:
            不打开文件、不读内容、不做任何"取整兼容"(差一个字节就是错)。
        """
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

def _write_index(index: 'faiss.Index',index_path: Path) -> dict:
    """把索引写进 index/*.faiss,并记录体积与耗时。

        ===========================================================================
        实现原理
        ===========================================================================
        ★ 为什么【不做】"临时文件 + os.replace()" 原子写(D3 的 progress.json 做了):
            索引文件在设计上限档约 4 GB,写一个临时副本会让磁盘占用【翻倍】。
            这里的取舍是:直接写 + 写完用 _verify_alignment 回读校验兜底。

        调用的外部函数:
            Path.parent.mkdir(parents=True, exist_ok=True) —— 建父目录;
                parents=True 表示连缺的各级父目录一起建;
                exist_ok=True 表示"已存在也不报错"(幂等,重复跑不会炸)。
            faiss.write_index(index, fname) —— ★ fname 必须传 str。
                FAISS 是 C++ 库用 SWIG 封装的,不认 pathlib.Path,传 Path 会抛类型错误。
            Path.stat().st_size —— 落盘后立刻量一下实际字节数。

        参数:
            index:      faiss.Index —— 已 add 完的索引
            index_path: Path        —— 索引文件输出路径
        返回:
            dict —— {"index_bytes": 字节数, "write_index_sec": 耗时, "ntotal": 行数}
        明确不做:
            不做压缩、不做分片落盘(FAISS 自己就是一个文件)。
        """
    index_path.parent.mkdir(parents=True,exist_ok=True)
    t0 = time.perf_counter()
    faiss.write_index(index,str(index_path))        #必须使用 str(index_path)
    sec = index_path.stat().st_size     # 落盘耗时
    size = index_path.stat().st_size    # 落盘后立即测量实际体积
    logger.info(
        '索引已落盘:%s | %s | %d 行 | 耗时 %.1fs',
        index_path,_fmt_bytes(size),index.ntotal,sec,
    )
    return {
        'index_bytes': size,            # 索引文件字节数
        'write_index_sec': round(sec,2), # 落盘耗时
        'ntotal': int(index.ntotal)     # 索引里的向量行数
    }

def _iter_meta_rows(chunks_path: Path) -> Iterator[dict]:
    """把索引写进 index/*.faiss,并记录体积与耗时。

        ===========================================================================
        实现原理
        ===========================================================================
        ★ 为什么【不做】"临时文件 + os.replace()" 原子写(D3 的 progress.json 做了):
            索引文件在设计上限档约 4 GB,写一个临时副本会让磁盘占用【翻倍】。
            这里的取舍是:直接写 + 写完用 _verify_alignment 回读校验兜底。

        调用的外部函数:
            Path.parent.mkdir(parents=True, exist_ok=True) —— 建父目录;
                parents=True 表示连缺的各级父目录一起建;
                exist_ok=True 表示"已存在也不报错"(幂等,重复跑不会炸)。
            faiss.write_index(index, fname) —— ★ fname 必须传 str。
                FAISS 是 C++ 库用 SWIG 封装的,不认 pathlib.Path,传 Path 会抛类型错误。
            Path.stat().st_size —— 落盘后立刻量一下实际字节数。

        参数:
            index:      faiss.Index —— 已 add 完的索引
            index_path: Path        —— 索引文件输出路径
        返回:
            dict —— {"index_bytes": 字节数, "write_index_sec": 耗时, "ntotal": 行数}
        明确不做:
            不做压缩、不做分片落盘(FAISS 自己就是一个文件)。
        """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    #   index/ 目录可能还不存在(首次运行),先建出来

    t0 = time.perf_counter()
    #   记下开始时刻

    faiss.write_index(index, str(index_path))
    #   ★ 必须 str(index_path):SWIG 封装不认 Path。这是 FAISS 最常见的坑之一

    sec = time.perf_counter() - t0
    #   落盘耗时(4 GB 索引用时可达数十秒)

    size = index_path.stat().st_size
    #   落盘后立刻量实际体积,用于验收 V3 的记录

    logger.info(
        "索引已落盘:%s | %s | %d 行 | 耗时 %.1fs",
        index_path, _fmt_bytes(size), index.ntotal, sec,
    )
    return {
        "index_bytes": size,  # 索引文件字节数
        "write_index_sec": round(sec, 2),  # 落盘耗时
        "ntotal": int(index.ntotal),  # 索引里的向量行数
    }


def _iter_meta_rows(chunks_path: Path) -> Iterator[dict]:
    """流式把 chunks 的每一行,变成 meta 的一行(补上 faiss_row)。

    ===========================================================================
    ★ 实现原理
    ===========================================================================
    enumerate 给出的行号就是 FAISS 行号 —— 两者同源,不需要额外映射表。
    这是"行序 1:1"铁律能成立的【代码层面】根据,不是靠约定。

    为什么缺 chunk_id 必须炸:
        静默退化成空会让几百万块重名,而后面的行数校验根本查不出来
        (行数是对的,只是主键没了)。

    调用的外部函数:
        common.utils.iter_jsonl(path) —— 逐行 json.loads 的生成器:
            内存 O(1)(不会把几 GB 文件读进来),坏行会带行号抛 ValueError。

    参数:
        chunks_path: Path —— D2 的切片 JSONL
    返回(yield):
        dict —— 一行 meta:{chunk_id, doc_id, title, chunk_index, text, faiss_row}

    """
    for row,obj in enumerate(iter_jsonl(chunks_path)):  #iter_jsonl:逐行产出dict,enumerate 同时给出行号，enumerate 默认从0开始 - 这与FAISS的行号起点一致
        if 'chunk_id' not in obj :          # 住建确实报错,静默退化会造成几百万块重名
            raise ValueError(
                # row + 1 习惯从1开始，list(obj)[:5] 把dict的键转成列表，只显示前5个，便于定位问题
                f"{chunks_path} 第 {row + 1}行缺少 chunk_id字段，字段为{list(obj)[:5]}"
            )

        #使用yield 而不是 return:这是个【生成器函数】，调用它不会立即执行，而是在for里逐行产出
        yield {
            'chunk_id': obj['chunk_id'],            # 主键
            'doc_id': obj.get('doc_id'),            # .get 取不到时，返回None，不报错
            'title': obj.get('title') or '',
            'chunk_index': obj.get('chunk_index'),  # 块在原文内的序号
            'text': obj.get('text') or '',
            'faiss_row': row
        }

def _write_meta(chunks_path: Path,meta_path: Path,expect_rows: int=None) -> dict:
    """把 chunks 流式改写为 index/chunks_meta.jsonl。

        ===========================================================================
        实现原理(四条工程约定)
        ===========================================================================
        1. newline="\\n" —— 产物统一 LF 换行,别让 Windows 的 CRLF 混进来;
        2. 攒批写 —— 20 万行一行一写是 20 万次函数调用,攒成 1 万行一批后约 20 次;
        3. _META_HEARTBEAT 心跳 —— 大档位要跑几分钟,中途不能没动静;
        4. try/finally: gen.close() —— 生成器内部持着输入文件句柄,
           ★ 提前 break 时尤其要显式关,不该交给垃圾回收(GC 时机不确定)。

        调用的外部方法:
            Path.open(mode, encoding, newline) —— 打开文件。
                mode="w" 写模式(文件已存在会被清空);
                encoding="utf-8" 指定编码,中文必须显式写;
                newline="\\n" 强制 LF(否则 Windows 上会写成 \\r\\n)。
            json.dumps(obj, ensure_ascii=False) —— dict 转 JSON 字符串;
                ensure_ascii=False 表示【不】把中文转义成 \\uXXXX,
                写出来的文件人眼能直接看。
            "\\n".join(buf) —— 用换行符把列表里的字符串连起来(只在行间补换行)。

        参数:
            chunks_path: Path       —— 输入:D2 的切片 JSONL
            meta_path:   Path       —— 输出:元数据 JSONL
            expect_rows: int | None —— 最多写多少行(limit 模式);None 表示全写
        返回:
            dict —— {"meta_rows": 行数, "meta_bytes": 字节数, "write_meta_sec": 耗时}
        明确不做:
            不排序、不改写字段、不去重。
        """
    meta_path.parent.mkdir(parents=True,exist_ok=True)
    t0 = time.perf_counter()
    rows = 0            # 已写入的行数计数器
    buf = []            # 凑足缓冲区:临时存放待写入的json字符串
    gen = _iter_meta_rows(chunks_path)      # 获取生成器
    try:
        # with语句保证离开代码时自动关闭文件句柄，不用手写close()
        with meta_path.open('w',encoding='utf-8',newline='\n') as fout:
            for obj in gen:         # 逐行从生成器取dict,内存里一次只有一行
                buf.append(json.dumps(obj,ensure_ascii=False))     # 转为json字符串，保存到buf里
                rows += 1    # 行数 +1
                if len(buf) >= _META_WRITE_BATCH:       # 攒足一万行 -> 真正落盘
                    fout.write('\n'.join(buf))      # join 只在【行与行之间】补换行
                    fout.write('\n')                # 行尾再补一个：否则下一批的第一行会粘在上一批最后一行后面
                    buf.clear()     #清空缓冲区

                if rows % _META_HEARTBEAT == 0:         # 每写满 20 万行打一条心跳日志
                    logger.info(
                        '元数据已写 %d 行 | %s | 耗时 %。0fs',
                        rows,_fmt_bytes(meta_path.stat().st_size,time.perf_counter() -t0,)
                    )

                if expect_rows is not None and rows >= expect_rows:
                    # limit 模式：写够了就结束
                    logger.info('已达上限 %d 行，提前结束(limit 模式)',expect_rows)
                    break
            if buf:
                # 收尾：最后不足一个批量的剩余必须写入，否则最多会丢掉 9999 行
                fout.write('\n'.join(buf))
                fout.write('\n')
    finally:
        gen.close()   # 显示关闭生成器
    return {
        'meta_rows': rows,      # 写入的元数据行数
        'meta_bytes': meta_path.stat().st_size,         # 产物字节数
        'write_meta_sec': round(time.perf_counter() - t0,2),    #写 mete的耗时
    }


def _count_rows(path: Path) ->int:
    """字节级快速数行数(不解析 JSON)。

    ===========================================================================
    ★ 实现原理
    ===========================================================================
    按 1 MiB 二进制块读,直接数块内 b"\\n" 的出现次数,
    【完全不走 Python 字符串解码】。
    UTF-8 的多字节序列里不会出现 0x0A 这个字节,所以"数 \\n 字节"
    严格等于"数行数",不存在误判。

    为什么刻意不用 iter_jsonl:
        它会对每一行做 json.loads —— 20 万行只为数个数太浪费
        (D3 的 _count_rows 是同款写法)。

    调用的外部方法:
        open(path, "rb") —— r=只读,b=二进制模式(不做任何编码解码、不做换行转换)
        bytes.count(sub) —— C 层实现的子串计数,极快
        buf[-1:]         —— 取本块最后一个字节(切片写法,拿到的还是 bytes)

    参数:
        path: Path —— 待数行的文本文件(一般是 .jsonl)
    返回:
        int —— 行数(末尾无换行符时补 1;空文件返回 0)
    明确不做:
        不做 json.loads、不校验行内容。
    """
    # 累计统计到的行数
    total = 0

    # 记住最后一次读取的字符，b'' 是空的字符串(bytes)，不是字符串str - 因为是按二进制模式打开的文件
    last_byte = b''

    #open(path,'rb'):r = 只读，b = 二进制模式（不用UTF-8解码，直接度原始字节）
    # with ：离开这个代码块时自动关闭文件句柄，无需手写f.close()
    with open(path,'rb') as f:
        while True:
            buf = f.read(1 << 20) # 每次读1MiB。
            if not buf:     #读到文件末尾会返回空的bytes -> 空是'假' ->退出循环
                break
            total += buf.count(b'\n')       # 累加总字节数
            last_byte = buf[-1:]  # 记录本块最后一个字节

    if last_byte and last_byte != b'\n':    # 文件非空，且结尾不是换行符 -> 说明最后一行没收尾，要补算1行
        total += 1
    return total

def _read_index(index_path: Path) -> tuple:
    """回读索引,并顺便记下"加载耗时"。

        ===========================================================================
        实现原理
        ===========================================================================
        回读是【唯一能证明落盘产物可用】的动作 —— 只写不读,
        可能写了个坏文件而毫无察觉。
        它给出的加载耗时,正是 D6 在线链路首次提问时那 1~2 秒延迟的来源
        (架构 §4 步骤 0),所以 D6 的 retrieve.load_index 必须做进程内单例缓存;
        ★ 这里刻意【不缓存】—— 本模块每次都是全新的校验,缓存了就等于没校验。

        调用的外部函数:
            faiss.read_index(fname) —— ★ 同样只收 str,不认 pathlib.Path

        参数:
            index_path: Path —— 索引文件路径
        返回:
            tuple —— (faiss.Index, float 加载耗时秒)
        明确不做:
            不缓存单例、不做 Warm-up 检索。
        """
    if not index_path.is_file():
        raise FileNotFoundError(
            f"索引文件不存在:{index_path}\n，请先运行index_build.py"
        )
    t0 = time.perf_counter()

    index = faiss.read_index(str(index_path))
    sec = time.perf_counter() - t0

    logger.info(
        '索引已回读:%s | %d 行 / dim=%d |加载耗时 %.2fs',index_path,index.ntotal,index.d,sec,
    )
    return index,round(sec,2)

def _verfify_aligment(index_path: Path,meta_path: Path,expect_rows: int=None) -> dict:
    """对齐校验:索引向量数 == 元数据行数 == 期望行数。

        ===========================================================================
        ★ 实现原理(为什么要做这件事)
        ===========================================================================
        索引与 meta 是两个独立的文件,它们之间【唯一】的关联是行号。
        写完各自【独立再数一遍】,是交付前唯一能抓住"错位 / 截断 / 少写"的手段。
        这类错误在线上表现为"答案张冠李戴" —— 不崩、不报、最难查,
        所以必须在建库时就堵死。

        代价提示:
            回读要重新扫一遍向量 —— L 档 0.78 GiB 几乎瞬间,
            设计上限档 4 GB 需数十秒到一两分钟。所以提供 --no-verify 开关,
            但【默认开】(省这几十秒不值得赌数据正确性)。

        参数:
            index_path:  Path       —— 索引文件路径
            meta_path:   Path       —— 元数据文件路径
            expect_rows: int | None —— 期望行数;None 表示不校验这一项
        返回:
            dict —— {"ntotal", "meta_rows", "load_sec", "aligned"}
        抛出:
            RuntimeError —— 三者不一致时,错误信息里直接给出处理办法
        明确不做:
            不校验字段内容(那是验收脚本的事,不是本函数的职责)。
        """

    index,load_sec = _read_index(index_path)        # 回读索引，顺便拿到加载耗时
    ntotal = int(index.ntotal)      # 索引里的向量行数
    meta_rows = _count_rows(meta_path)  # 元数据行数

    #两个条件同时满足才算通过：
    # 索引行数 == 元数据行数   且    期望行数 == 索引行数
    ok = (ntotal == meta_rows) and (expect_rows is None or expect_rows == ntotal)
    if not ok:
        # 不一致就报错
        raise RuntimeError(
            '索引行数与元数据行数不一致，产物不可信(例如索引会把A的向量配上B的文本):\n'
            f"索引向量数 ntotal = {ntotal}\n"
            f"元数据行数        = {meta_rows}\n"
            f"期望行数          ={expect_rows}\n"
            "处理:删除index/下这两个文件后重跑 index_build.py"
        )
    logger.info("对齐校验通过:碎银 %d 行 == 元数据 % 行",ntotal,meta_rows)
    return {
        'ntotal': ntotal,           #索引向量行数
        'meta_rows': meta_rows,     # 元数据行数
        'load_sec': load_sec,       # 索引加载耗时
        'aligned': True,
    }


# ===========================================================================
#                        第 2 层:编排(所有顺序与闸门都在这里)
# ===========================================================================
def build_index(vectors_path=None,chunks_path=None,index_path=None,meta_path=None,
                dim:int = config.EMBED_DIM,expect_rows:int = None,
                full:bool = False,audit: bool = True,verify:bool = True) ->dict:
    """主流程:向量矩阵 + 切片元数据 → FAISS 索引 + 元数据(行序严格 1:1)。

        ===========================================================================
        ★ 执行顺序本身就是设计,一步都不能调换
        ===========================================================================
        1. 四件套路径解析(None 按 full 取默认)
        2. ★ 先检查后开 —— 两个输入的存在性检查,必须排在一切写操作之前
        3. 权威行数来自 chunks,不是从文件大小猜(见 _infer_rows)
        4. 打开向量矩阵(内部会用权威行数去校验字节数)
        5. 入库前体检 —— 四道闸门
        6. 建索引 → 分片 add → 落盘 → 及时释放内存
        7. 流式写元数据
        8. 对齐校验
        9. 返回统计 dict

        为什么步骤 2 的顺序不可调换:
            _write_index / _write_meta 一旦开始,就等于把上次产物覆盖/清零了;
            此时才发现输入不存在,一次手滑毁掉上次跑了几小时的结果。
            (这是 D2 §8「先筛后开」的同款事故。)

        为什么步骤 6 有 del index:
            索引在设计上限档常驻约 4 GB,写完就别占着 —— 后面写 meta 要跑几分钟。

        四道闸门(步骤 5)的顺序也有讲究:
            先查 nan/inf(最致命)→ 零行 → 归一化 → dtype。

        参数:
            vectors_path / chunks_path / index_path / meta_path: Path | str | None
                —— 四个路径;传 None 表示用 _default_paths(full) 的默认值
            dim:         int      —— 向量维度,默认 config.EMBED_DIM(维度即契约)
            expect_rows: int|None —— 最多处理多少行(limit 模式);None 表示全量
            full:        bool     —— True 用全量四件套,False 用调试子集四件套
            audit:       bool     —— 是否做入库前体检(默认 True)
            verify:      bool     —— 是否做落盘后的回读对齐校验(默认 True)
        返回:
            dict —— 统计信息,键见下方 stats
        明确不做:
            不调 Ollama(那是 sanity_check 的事);不排序去重;不改 text。
        """
    t_start = time.perf_counter()           #计时开始
    dv,dc,di,dm = _default_paths(full)      # 取出默认四件套：向量、切片、索引、元数据
    vectors_path = Path(vectors_path) if vectors_path else dv       # 向量输出路径
    chunks_path = Path(chunks_path) if chunks_path else dc           # 切片输出路径
    index_path = Path(index_path) if index_path else di             # 索引输出路径
    meta_path = Path(meta_path) if meta_path else dm                # 元数据输出路径

    logger.info('=' * 70)
    logger.info("D4 建索引开始 | 模式=%s", "全量" if full else "调试子集")
    logger.info("  输入① 向量:%s", vectors_path)
    logger.info("  输入② 切片:%s", chunks_path)
    logger.info("  输出① 索引:%s", index_path)
    logger.info("  输出② 元数据:%s", meta_path)

    # 步骤2：校验
    if not chunks_path.is_file():       # 切片文件不存在
        raise FileNotFoundError(f"切片文件不存在，{chunks_path}\n请先运行chunk.py")

    if not vectors_path.is_file():      # 向量文件不存在
        raise FileNotFoundError(f"向量文件不存在:{index_path}\n请先运行embed.py")

    # 步骤3：权威行数来自chunks(既定事实),不是从文件大小猜

    n_chunks = _count_rows(chunks_path)             #字节级快速计算chunks行数

    if n_chunks <= 0:       #chunks是空文件
        raise ValueError(f"切片文件为空，无法建库:{chunks_path}")

    if expect_rows is None:
        n_rows = n_chunks       # 没给limit -> 全部行都处理
    else:
        # 给了limit -> 取两者较小值：注意这里取较小值来决定处理量，而不是"取较小值截断“，后面 _infer_rows 会用n_rows 去校验字节数，如果不匹配就raise
        n_rows = min(n_chunks,expect_rows)

    if n_rows <= 0:
        # 如果limit 传了0 或 负数
        raise ValueError(f"待处理行数必须为正整数，收到 n_rows = {n_rows}(chunks={n_chunks},expect_rows={expect_rows})")
    logger.info("权威行数来自 chunks:%d 行(本次处理 %d 行)",n_chunks,n_rows)

    # 步骤4：打开向量矩阵(内部会拿 n_rows 去校验字节数)
    mm = _open_vectors(vectors_path,dim=dim,expect_rows=n_rows)         # mm 是只读的memmap:结构是(n_rows,dim),dtype=float32

    # 步骤5：入库前校验
    audit_report = None         # 先置空：如果跳过校验，返回dict里也是这里None

    if audit:
        audit_report = _audit_vectors(mm)       # 抽样校验

        # 校验1 -> nan/inf : 一旦进库，任何查询的分数都可能变成nan
        if not audit_report['finite']:
            raise ValueError(
                f"向量中出现 nan/inf，禁止入库(抽样{audit_report['sampled']}行):\n 处理:归一化是否产生了 0/0,重跑embedding。"
            )
        # 校验2 -> 全零行:断点续跑没跑完时，尾部会留零行
        if audit_report['zero_rows'] > 0:
            raise ValueError(f"存在{audit_report['zero_rows']} 个全零行(抽样{audit_report['sampled']} 行:\n)"
                             "常见原因，在embed过程中终端，矩阵尾部没写。处理：重跑embed或用 --limit 截断。")

        # 校验3 -> 未归一化：内积 ！= 余弦
        if not audit_report['normalized']:
            raise ValueError(
                f"向量未归一化(范数 {audit_report['norm_min']:.6f}~{audit_report['norm_max']:.6f}),"
                "内积 ≠ 余弦,检索会静默变差:\n"
                "  处理:检查 embed.py中 的 _l2_normalize 是否生效。"
            )

        # 校验4 -> dtype:faiss的add()只接收float32的数据
        if audit_report['dtype'] != 'float32':
            raise ValueError(f"dtype 必须是 float32，实际为{audit_report['dtype']}")
    else:
        logger.warning('已跳过向量校验(audit=False):未归一化/nan/零行都不会被发现')

    # 步骤6：建索引 -> 分片add -> 落盘 -> 释放
    index = _new_index(dim)         # 创建空的IndexFlatIP
    add_stats = _add_shards(index,mm)           # 分片把向量添加索引(顺序即编号)
    write_stats = _write_index(index,index_path)    # 落盘到 index/*.faiss
    del index       # 主动释放：索引在设计上限档约为4GB常驻内存，后面写meta要跑几分钟，没必要一直占用

    # 步骤7：流式写入元数据
    meta_stats = _write_meta(chunks_path,meta_path,expect_rows=n_rows)  # 注意传的是 n_rows(不是expect_rows)：limit 模式下要写前 n_rows 行

    # 步骤8：对齐校验
    if verify:
        align = _verfify_aligment(index_path,meta_path,expect_rows=n_rows)      # 回读索引 + 数meta 行数，三者必须相等
    else:
        logger.warning('已跳过落单后的对齐校验(verify=False) - 产物错位不会被发现')
        align = {'ntotal': int(add_stats['added']),'meta_rows': int(meta_stats['meta_rows']),
                 'load_sec': 0.0,'align':False}
        # 跳过校验时也要把键补全，让返回dict结构一致

    # 步骤9：组装统计 dict
    elapsed = time.perf_counter() - t_start





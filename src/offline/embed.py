# -*- coding: utf-8 -*-
"""D3 - embed.py:向量化(RAG 的核心,离线建库 + 在线查询共用)

职责(自底向上四层,详见 D3开发指引.md §4):
    第 1 层  _get_session / _count_rows / _load_progress / _save_progress / _post_batch / _l2_normalize
             工具层:HTTP 连接、行数统计、续跑进度、单批请求、归一化
    第 2 层  embed_texts / embed_query / cosine
             通用 API:文本 -> 向量(在线、离线都用这一层)
    第 3 层  embed_chunks
             离线批量作业:流式读块 -> 攒批 -> 落 memmap -> 断点续跑
    第 4 层  __main__
             命令行入口(--full / --limit N / --batch N / --no-resume)

★ 全模块唯一的铁律:向量文件的第 i 行,必须对应 chunks 文件的第 i 行。
  由此推出三条禁令:不许排序、不许去重、不许跳行(失败宁可 raise)。

运行:python -m offline.embed              (跑 subset,默认可断点续跑)
     python -m offline.embed --full       (L 规模基线,D7 用)
     python -m offline.embed --limit 2000 (只跑前 2000 块,调试用)
     python -m offline.embed --batch 32   (覆盖批大小)
     python -m offline.embed --no-resume  (忽略进度,从 0 重跑)
"""
import itertools                            # 只用一个函数：islice，断点续跑时跳过已处理的前N行
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np                          # 向量矩阵的唯一载体
import requests
from tqdm import tqdm

from common import config
from common.utils import iter_jsonl

# 模块级logger
logger = logging.getLogger(__name__)

#模块级单例：复用同一个requests.Session
# 为什么是None + 懒加载 而不是 直接建好：如果在import阶段就创建Session,会给导入带来副作用（建连接池、读环境变量代理配置）
_SESSION = None

#模块级工程常亮，与算法无关，所以保留模块里，不仅config
_HEARTBEAT = 100_000        #每处理满100000行打一条日志
_PROGRESS_SUFFIX = '.progress.json'    # 进度文件的后缀，紧跟在.npy文件后面
_EPS = 1e-12                #归一化时给零向量兜底的下限值

# ============================================================================
#                        第 1 层:工具(原子能力)
# ============================================================================
def _get_session() ->requests.Session:
    """
        获取全局唯一的 requests.Session(懒加载单例)。

        实现原理:
            requests.get/post 这类"模块级快捷函数"在内部其实是"每次新建一个 Session,
            用完即关"。新建 Session = 新建 TCP 连接(三次握手)+ HTTP 握手,
            本项目全量要发几万次 /api/embed,这部分开销不可忽略。
            Session 内部维护 urllib3 的连接池,配合 HTTP/1.1 keep-alive,
            同一个主机端口的连接会被复用,省掉每批一次握手。

        调用的外部方法:
            requests.Session():requests 的会话对象,自动管理 cookies、headers、连接池。
            本函数不做任何鉴权(本机 Ollama 服务无需鉴权),也不改 timeout
            (超时是每次请求各自指定的,见 _post_batch)。

        参数:无
        返回:requests.Session —— 全局唯一实例(多次调用返回同一个对象)

        明确不做:
            不做重连状态判断 —— Session 遇到网络错误会自己标记连接失效,下一次 post 自动重连。
        """
    global _SESSION                             # 声明要改的是'模块级'变量
    if _SESSION is None:                        # 懒加载判断：只有第一次调用时才是None,之后都不会执行
        _SESSION = requests.Session()           # 创建实例
    return _SESSION

def _count_rows(path:Path) ->int:
    """
        快速统计 JSONL 文件行数(给 memmap 预分配 shape 用)。

        功能实现原理:
            memmap 必须预先知道 (行数, 维度)。若用 sum(1 for line in f) 会逐行解码字符串,
            2.6 GB 文件要几秒到十几秒;这里按 1 MB 二进制块读,直接数块内 b'\\n' 出现次数,
            完全不过 Python 字符串解码,快得多。UTF-8 多字节序列里不会出现 0x0A,所以
            "数 \\n 字节"严格等于"数行数"。

        调用的外部方法:
            open(path,'rb'):二进制打开,避免编码/换行转换干扰字节统计。
            bytes.count(sub):C 层实现的子串计数,极快。
            buf[-1:]:取本块最后一个字节,用于判断文件末尾有无换行符。

        参数:
            path: Path —— 待统计文件(一般 data/chunks/*.jsonl)
        返回:int —— 行数(末尾无换行符时补 1;空文件返回 0)

        明确不做:
            不做 json.loads —— 这里只数行,不检查内容合法性(坏行由 iter_jsonl 负责报错)。
        """
    total = 0               # 累加器：已数到的换行符数量
    last_byte = b''         # 记住最后一个字节，用来判断文件末尾有没有换行
    with open(path,'rb') as f:          # with 自动关闭文件；'rb' 读/二进制模式，不做任何解码
        while True:
            buf = f.read(1 << 20)       #每次读1MB
            if not buf:                 # 读完了-读到文件末尾返回空bytes
                break;
            total += buf.count(b'\n')   # 统计这一块有多少换行符，累加
            last_byte = buf[-1:]        # 记录本块最后一个字节(切片写法，拿到的还是bytes)
    if last_byte and last_byte != b'\n': # 文件非空，且结尾不是换行符
        total += 1                          # 说明最后一行没有换行结尾，补算1行
    return total                        # 返回行数

def _progress_path(out_path:Path) -> Path:
    """
       由产物路径推出进度文件路径:xxx.vec.npy -> xxx.vec.npy.progress.json

        功能实现原理:
            用 Path(str(...) + 后缀) 重新包成 Path,而不是字符串拼接后忘记转 Path。
            刻意【不用】out_path.with_suffix():那会替换掉 .vec.npy 的最后一段后缀,
            变成 xxx.progress.json,与原产物命名冲突风险更高。

        参数:
            out_path: Path —— 向量产物路径(通常 data/vectors/*.vec.npy)
        返回:Path —— 进度文件路径

        明确不做:不建目录、不判断文件是否存在。
       """
    return Path(str(out_path) + _PROGRESS_SUFFIX)  # 路径转字符串 + 拼后缀 ,再转回Path 返回

def _load_progress(out_path:Path,total:int,dim:int,batch_size:int) -> int:
    """
        读取断点续跑的起始行号。

        功能实现原理:
            续跑可信的前提是"这次参数与上次完全一致"。只要 total / dim / batch_size 任一
            对不上,上次"写到第 k 行"的含义就变了。所以采取"严格校验 + 宽松降级":
            校验不过就返回 0(从头重跑),宁可多花时间,也不冒险让矩阵错位。
            dim 必须参与校验:换模型后维数变了,旧进度若被当有效起点会直接导致行序错位。

        调用的外部方法:
            Path.exists() / Path.read_text() / json.loads()

        参数:
            out_path:   Path —— 向量产物(进度文件由它推出来)
            total:      int  —— 本次计划写入的总行数
            dim:        int  —— 本次向量维度
            batch_size: int  —— 本次批大小
        返回:int —— 已完成的 rows_done(0 表示从头跑)

        明确不做:
            不做"部分有效"的推断 —— 猜错一次的代价是整个矩阵错位。
        """
    p = _progress_path(out_path)                           # 进度文件路径
    if not p.exists():
        return 0
    try:
        info = json.loads(p.read_text(encoding='utf-8'))    # 读文本并解析成dict
    except Exception as e:
        logger.warning(f"进度文件损坏({e}),本次从第0行重跑")
        return 0            # 损坏则作废，从0重跑
    if (info.get('total') != total              # 校验：总行数是否一致
            or info.get('dim') != dim           # 校验：维度是否一致->换模型会不一致
            or info.get('batch_size') != batch_size): # 校验：批大小是否一致
        logger.warning(f"进度文件与本次参数不匹配(上次total=%s/dim=%s/batch=%s)",
                       info.get('total'),info.get('dim'),info.get('batch_size'))
        return 0
    rows_done = int(info.get('rows_done',0))        # 取出上次写完的行数
    return max(0,min(rows_done,total))      #夹到[0,total],防止进度文件被手改成越界值

def _save_progress(out_path:Path,rows_done:int,total:int,dim:int,batch_size:int) -> None:
    """
        原子地记录"已写完多少行"(断点续跑的关键)。

        功能实现原理(为什么必须"先写临时文件再 rename"):
            若直接 open(p,'w') 写到一半被 Ctrl+C / 断电,磁盘上会留半截 JSON,
            下次 json.loads 直接抛错,断点信息全丢。先写 .tmp 再 os.replace() 是
            POSIX 保证的原子操作:目标要么保持旧内容、要么整体换成新内容,绝无中间态。

        调用的外部方法:
            json.dumps(..., ensure_ascii=False):写 UTF-8 明文,便于人手查看。
            Path.write_text / os.replace(src, dst):Windows/POSIX 都保证原子替换。

        参数:
            out_path:   Path —— 向量产物路径(进度文件由它推出来)
            rows_done:  int  —— 当前已写入且已 flush 的行数
            total/dim/batch_size: 本次运行参数,下次续跑时用来校验一致性
        返回:None

        明确不做:不做批量合并、不保证跨设备(不同磁盘)原子。
        """
    p = _progress_path(out_path)            #进度文件目标路径
    tmp = Path(str(p) + '.tmp')             # 临时文件路径(和p在同一目录，保证rename原子)
    info = {                        # 要写进文件的字典
        'rows_done':rows_done,      # 已写完的行数
        'total':total,              # 本次总行数
        'dim':dim,                  # 本次维度
        'batch_size':batch_size     # 本次批大小
    }
    tmp.write_text(json.dumps(info,ensure_ascii=False) + '\n',encoding='utf-8') # 先写临时文件
    os.replace(tmp,p)       #原子替换，要么是旧内容，要么是完整新内容

def _post_batch(texts:list[str]) -> np.ndarray:
    """
        第 1 层核心:把"一批"文本交给 Ollama,返回原始向量(未归一化)。

        功能实现原理:
            一次 HTTP POST 带多条文本,由 Ollama 内部组成一次 GPU batch:模型权重只读一遍
            显存就被这批数据复用,摊薄了 kernel launch 与访存开销(这就是 batch=1 比
            batch=64 慢一个量级的原因)。
            失败按 2/4/8 秒指数退避重试:网络抖动几百毫秒自愈,而"模型被挤出显存"恢复更久,
            指数退避用最小总等待覆盖两种场景。

        调用的外部方法:
            _get_session().post(url, json=..., timeout=...):
                requests.Session.post —— 发 POST;json= 自动序列化 dict 并带 Content-Type。
                timeout 是"发请求到收完响应"的上限;首次要把 1.2GB 模型加载进显存(几十秒),
                所以给 300 秒(config.EMBED_TIMEOUT),后续毫秒级,300 只是上界。
            r.raise_for_status():4xx/5xx 抛 requests.HTTPError(200 不代表内容对)。
            r.json():解析响应体;必须放在 try 里(见"其它注意事项")。
            np.asarray(rows, dtype=np.float32):
                Python 浮点默认 float64,不显式写 dtype 会得到 float64 矩阵 —— 体积翻倍
                且 faiss.add() 拒收。

        参数:
            texts: list[str] —— 一批文本,长度应 <= batch_size(本函数自身不再分批)
        返回:np.ndarray,形状 (len(texts), config.EMBED_DIM),dtype=float32(未归一化)

        明确不做:
            不做归一化 —— 那是 _l2_normalize 的职责(分层,便于各自独立测试)。

        其它注意事项:
            1) 行数校验是离线作业最重要的一道防御:服务端若少返回一行,vstack 会安静地少拼
               一行,矩阵从第 i 行起整体错位,要到 D4 的 sanity_check 才暴露 —— 那时已白跑几小时。
            2) Ollama 的 embeddings 顺序与 input 顺序一致(服务端契约),行数校验就是防它违约。
        """
    payload = {'model':config.EMBED_MODEL,'input':texts}        # 请求体 ： 模型名 + 一批文本
    last = None     # 保存最后一次异常，失败是拼入错误信息
    for attmpt in range(config.EMBED_MAX_RETRIES + 1):      # 一共4次机会(1次正常 + 3次重试)
        try:
            r = _get_session().post(            # 服用连接池发POST
                f"{config.OLLAMA_URL}/api/embed", # OLLAMA 的向量接口地址
                json=payload,       # 请求体自动序列化
                timeout=config.EMBED_TIMEOUT, #超时上限(秒)
            )
            r.raise_for_status()        #4xx/5xx 在这里变成异常状态，执行下面的except
            rows = r.json()['embeddings']  # 取出向量列表
            if len(rows) != len(texts):         # 行数校验：保证服务端返回行数与实际行数一致
                raise ValueError(f"OLLAMA返回{len(rows)}行，期望{len(texts)}行")
            return np.asarray(rows,dtype=np.float32)   # 转成 float32 矩阵返回(未归一化)
        except Exception as e:
            #统一兜住三种异常：requests 网络异常 / HTTPError / .json()解析失败/行数不符
            # 它们都适合重试，因为都可能是服务暂时不可用
            last = e
            if attmpt < config.EMBED_MAX_RETRIES: # 还有重试机会
                time.sleep(config.EMBED_RETRY_SLEEP * (2 ** attmpt))  # 2/4/8秒
                logger.warning('embed 第 %d 次失败:%s,等待后重试',attmpt + 1,e)
    # 所以机会用完，绝不吞掉错误，让上层作业停下来，而不是跳过继续写入embed
    raise RuntimeError(f"embedd批次失败(已重试{config.EMBED_MAX_RETRIES}次):{last}")

def _l2_normalize(vecs:np.ndarray) -> np.ndarray:
    """
        第 1 层核心:逐行 L2 归一化,让每行向量的模长 = 1。

        实现原理:
            余弦相似度 cos(θ) = (A·B) / (|A| × |B|)。
            当 |A| = |B| = 1 时,分母恒为 1,于是 cos(θ) = A·B,也就是【内积(IP)】。
            D4 选的是 faiss.IndexFlatIP(暴力算内积),所以:
                归一化之后,faiss 返回的分数直接就是余弦相似度,可跨查询比较、可设阈值;
                不归一化,测出来的量混进了"向量长度",长句会被无理由地排到前面,
                而且【不报错、不崩溃】,只是悄悄变差 —— 这就是"静默降级"。
            架构 P5「防御性归一化」的含义:实测 Ollama 的 bge-m3 通常已返回单位向量,
            但我们仍显式做一遍,因为这条契约依赖后端实现,换服务就可能不成立。

        调用的外部方法:
            np.ascontiguousarray(x, dtype):
                ① dtype=float32 —— faiss.add() 只收 float32,且全量用 float64 会占 20GB;
                ② C 连续 —— faiss 按 C 语言内存布局直接读这块缓冲区。
                若已是 float32 且连续,它【返回同一个对象】(不拷贝)。
            np.linalg.norm(x, axis=1, keepdims=True):
                按行求欧氏范数,keepdims=True 使结果形状为 (N,1) 而不是 (N,)。
                  keepdims 不能省:(N,1024) / (N,) 会按最后一个维度广播而炸掉,
                  只有 (N,1) 才能正确地"逐行除"。
            np.maximum(norms, _EPS):
                兜零向量:0/0 会产出 nan,而 nan 一旦写进 FAISS,整库检索结果全废。

        参数:
            vecs: np.ndarray —— (N, EMBED_DIM) 的原始向量矩阵

        返回:np.ndarray —— 同形状的归一化矩阵(float32、C 连续)

        注意:
            当入参已是 float32+连续时,本函数会【就地修改】入参(vecs /= norms 是原地运算)。
            调用方若需要保留原值,请先传副本。
        """
    vecs = np.ascontiguousarray(vecs,dtype=np.float32)      #一次调用同时满足 float32 + c 连续，两个约束
    norms = np.linalg.norm(vecs,axis=1,keepdims=True)       # 逐行求模长，形状(n,1),才能正确广播
    norms = np.maximum(norms,_EPS)      # 零向量兜底：模长下限设成 1e-12，避免除以0，得到nan
    vecs /= norms       # 逐行相除，让每行模长 = 1(原地运算)
    return vecs         # 返回归一化后的矩阵

# ============================================================================
#                        第 2 层:通用 API(不碰磁盘)
# ============================================================================
def embed_texts(texts,batch_size=config.EMBED_BATCH_SIZE,show_progress=False):
    """
            任意条文本 -> (N, 1024) float32 已归一化矩阵;顺序与输入严格一致。

            功能实现原理:
                把文本列表按 batch_size 切成若干批 -> 逐批调 _post_batch 拿原始向量 ->
                np.vstack 按顺序拼成一个大矩阵 -> 一次性归一化。
                "顺序正确"由 vstack 保序 + _post_batch 的行数校验共同保证,不需要额外代码。

            调用的外部函数/方法:
                _post_batch(b):发一批,返回 (B,1024) 原始向量。
                np.zeros((0, DIM)):空输入时返回一个"零行矩阵",让调用方的循环/切片照常工作。
                np.vstack(parts):沿第 0 维(行方向)堆叠,第 k 块的行就是输入里第 k 批的行。
                tqdm(batches):可选进度条。
                _l2_normalize(x):对整块结果做一次归一化。

            参数:
                texts:         list[str] —— 任意条文本
                batch_size:    int  —— 每批发多少条(默认取 config.EMBED_BATCH_SIZE=64)
                show_progress: bool —— 是否显示进度条。★ 默认 False:embed_query 内部也调本函数,
                    在线链路每次提问都会调一次,若默认开进度条,每次提问都会滚一行 tqdm。
            返回:np.ndarray,形状 (N, EMBED_DIM),float32,已归一化

            明确不做:
                不落盘、不读文件、不知道这批文本属于哪篇文章(纯函数式,在线/离线共用)。
        """
    if not texts:       # 空列表(或None)的边界情况
        return np.zeros((0,config.EMBED_DIM),dtype=np.float32)      #返回(0,1024)空矩阵，不报错

    batches = [texts[i:1 + batch_size] for i in range(0,len(texts),batch_size)]  # 按批大小切片切批
    it = tqdm(batches,desc='embed',unit='batch') if show_progress else batches   # 需要时才包进度条
    parts = [_post_batch(b) for b in it]  # 逐批请求，得到若干[B,1024]矩阵
    return _l2_normalize(np.vstack())  # 竖向拼成大矩阵 ->归一化 ->返回

def embed_query(query):
    """
            单条查询 -> (1, 1024),可直接喂给 faiss index.search()。

            功能实现原理:
                就是 embed_texts([query]) 的便捷封装。为什么返回二维 (1,1024) 而非一维 (1024,)?
                faiss 的 index.search(x, k) 要求 x 是 (n_queries, dim) 二维矩阵且为 float32。
                返回一维的话调用方得写 q[None, :] 或 np.expand_dims —— 这种"记得要包一层"的
                约定最容易忘,不如让它出来就是能直接喂的形状。

            调用的外部函数:_post_batch(经 embed_texts)。
            参数:
                query: str —— 一条查询文本
            返回:np.ndarray,形状 (1, 1024),float32,已归一化

            明确不做:不做截断(超长由模型侧静默截断,见指引 §3.5)、不缓存结果。
        """
    return embed_texts([query])  # 把单条包成列表,复用批量 API,天然得到 (1,1024)

def cosine(a,b):
    """
            两条向量的余弦相似度(实验与调试用)。

            功能实现原理:
                cos = (a·b) / (|a| × |b|)。刻意【不要求入参已归一化】,对任意两条向量都正确,
                代价是一点重复计算 —— 这个函数的用途正是 D3 的四个实验(热力图/改一字/
                跨语言/长文本稀释)。提醒:若 a、b 都已归一化,np.dot(a,b) 就等于余弦。
                这里对分母做极小值兜底,避免零向量导致 nan。

            调用的外部方法:
                np.dot(a, b):内积(两向量对应元素相乘再求和)。
                np.linalg.norm(x):欧氏模长。

            参数:
                a, b: 两条向量(np.ndarray,可为一维 (D,) 或二维 (1,D))
            返回:float —— 余弦相似度,落在 [-1, 1]
            """
    a = np.asarray(a,dtype=np.float32)          # 同一设置为numpy数组，方便下面做点积 / 求范数
    b = np.asarray(b,dtype=np.float32)          # 同上
    denom = np.linalg.norm(a) * np.linalg.norm(b)  # 分母 |a| * |b|
    denom = max(float(denom),_EPS)          # 兜底：分母为0时，用1e-12替代，避免除0
    return float(np.dot(a.ravel(),b.ravel()) / denom)     # ravel 拉平成一维再点积，返回Python float

# ============================================================================
#                        第 3 层:离线批量主循环(难点)
# ============================================================================
def embed_chunks(in_path:config.CHUNKS_SUBSET_FILE,             #  默认输入：调试子集chunks(约2万块)
                 out_path:config.VECTORS_SUBSET_FILE,            # 默认输出：对应向量产物
                 batch_size: config.EMBED_BATCH_SIZE,                #默认批大小：64
                 limit: int | None = None,                      # 默认不限行；给整数则只跑前N行
                 resume: bool = True) ->dict:                   # 默认允许断点续跑；返回统计字典
    """
            离线主循环:流式读 chunks -> 攒批 -> embed -> 写 memmap -> 维护进度。

            功能实现原理(为什么用 memmap 而不是"攒 list 再 np.save"):
                后者要把整块矩阵(设计上限 4 GB)同时放在内存里;memmap 让虚拟内存替我们做
                磁盘搬运,进程内存 O(1),还能从任意行开始写 —— 这才让断点续跑变得平凡。

            调用的外部函数/方法:
                in_path.is_file() / _count_rows / _load_progress / _save_progress
                np.memmap(path, dtype, mode, shape):内存映射文件。
                    mode='w+':创建/清零;mode='r+':读写不清空(续跑必须用它,否则清掉旧数据)。
                itertools.islice(gen, start, None):跳过前 start 行。
                iter_jsonl(in_path):逐行 yield dict,内存安全。
                embed_texts(batch):算一批向量(已归一化)。
                mm.flush():把改动刷回磁盘。

            参数:
                in_path:    Path/str —— chunks 输入文件
                out_path:   Path/str —— 向量产物文件
                batch_size: int      —— 每次请求的文本条数
                limit:      int|None —— 只跑前 N 行;None = 全部
                resume:     bool     —— 是否尝试从进度文件续跑
            返回:dict —— 统计信息,含 rows/total_rows/dim/batch_size/resumed_from/
                         elapsed_sec/rows_per_sec/in/out/out_bytes
                         ★ rows_per_sec 是 D7 排期的关键输入,不要省。

            明确不做:不写 meta(D4 的活)、不并行(Ollama 单 GPU 串行,多进程只会排队)。
        """

    t0 = time.time()                #开始时间
    in_path = Path(in_path)         # 统一为Path
    out_path = Path(out_path)

    # 步骤1：先检查输入------
    if not in_path.is_file():
        raise FileNotFoundError(f"输入文件不存在:{in_path}")

    # 步骤2：确认行数，决定memmap的shape--------
    total = _count_rows(in_path)        # 确认总行数
    if limit is not None:               #指定了limit
        total = min(total,limit)        # 取小值，防止矩阵分配过大导致尾部多出0行

    # 步骤3：读进度，决定续跑起点----------
    start = 0           # 默认从头跑
    if resume:          # 只有允许续跑才能读进度文件
        start = _load_progress(out_path,total,config.EMBED_DIM,batch_size)      #读并严格校验口径

    # 步骤4：使用memmap ----
    out_path.parent.mkdir(parents=True,exist_ok=True)       #确保data/vectors 存在
    if start > 0 and out_path.exists():     #续跑：有 有效进度且旧产物还在
        mm = np.memmap(out_path,dtype=np.float32,
                       mode='r+',shape=(total,config.EMBED_DIM))    #r+ 读写、不清空，从start行接着写
    else:
        start = 0           #兜底：确保起点为0，避免错位
        mm = np.memmap(out_path,dtype=np.float32,mode='w+',
                       shape=(total,config.EMBED_DIM))      # w+ 创建并清零 -> 重跑

    # 步骤5：主循环------------
    gen = iter_jsonl(in_path)           # 拿到逐行读 jsonl的生成器（一次只在内存放一行）
    last_hb = start                     # 上次标记心跳日志的行号（用于‘每满10万行报一次’）
    row = start                         # 当前待写入的行号(全局行序)
    checked_dim = False                 # 是否校验过维度(只需校验第一批)
    try:
        raw = itertools.islice(gen,start,None)      #try/finally:无论成功失败都要关掉输入句柄
        rows_iter = tqdm(raw,total=total - start,desc='vectorize',unit='row')  #进度条只看剩余行
        batch = []          # 临时容器：凑够batch_size 条就发一次请求
        for obj in rows_iter:           #逐行便利每个块对象
            text = obj.get('text') or ''        #text字段；缺失/None 用空串兜底
            if not text.strip():        # 空文本/纯空白必须报错
                raise ValueError(f"{in_path} 第 {row + 1} 行文本为空")    #停下整个作业，指出行号
            batch.append(text)              #放进容器

            if len(batch) == batch_size:        # 此时容器已满
                vecs = embed_texts(batch,batch_size=batch_size)     # 调第二层：算一批向量->已归一化
                if not checked_dim:         #只对第一批校验维度
                    if vecs.shape[1] != config.EMBED_DIM:       #维度不等于默认的维度(1024)
                        raise  ValueError(f"向量维度{vecs.shape[1]} != {config.EMBED_DIM}")     #停止
                    checked_dim = True              # 标记已校验，后续批次不在重复查
                mm[row:row + len(vecs)] = vecs          #把这批向量写进[row,row + len] 行区间
                mm.flush()          #本地落盘：一批几百毫秒，flush成本可忽略，能少丢数据
                row += len(vecs)        #行号前进一批
                batch.clear()           # 清空容器
                _save_progress(out_path,row,total,config.EMBED_DIM,batch_size)      # 原子记录进度

                if row - last_hb >= _HEARTBEAT:     #距上次心跳已满10万行
                    last_hb = row                   # 更新心跳基准
                    logger.info("已写入 %d / %d 行(%.1f%%)",
                            row, total, 100.0 * row / max(total, 1))  # 打一条进度日志

        if batch:               #收尾：最后不足一批的结尾
            vecs = embed_texts(batch,batch_size=batch_size)         # 把最后的结尾生成向量
            if not checked_dim and vecs.shape[1] != config.EMBED_DIM:       #极小文件可能整批都没触发校验
                raise  ValueError(f"向量维度{vecs.shape[1]} != {config.EMBED_DIM}")     #遵守默认维度
            mm[row:row+len(vecs)] = vecs        # 最后写入
            mm.flush()          # 落盘
            row += len(vecs)    # 增加行数
            _save_progress(out_path,row,total,config.EMBED_DIM,batch_size)          #记录进度
    finally:
        gen.close()                 # 主动关闭生成器
        del mm                      # 释放memmap (等价于关闭这块内存映射)

    # 步骤6：返回统计字典-----------
    elapsed = time.time() - t0          # 总耗时(秒)
    stats = {
        'rows': row,                    #本次实际写入的总行数
        'total_rows': total,            # 计划总行数 = rows
        'dim': config.EMBED_DIM,        #向量维度
        'batch_size': batch_size,       # 批次大小
        'resumed_from': start,          # 本次从第几行续跑(0表示重新跑)
        'elapsed_sec': round(elapsed,2),#耗时，保留两位小数
        'rows_per_sec': round(row / elapsed,2) if elapsed > 0 else 0.0, # 吞吐 (块/秒)
        'in': str(in_path),         # 输入路径
        'out':str(out_path),        #输出路径
        'out_bytes': out_path.stat().st_size,       #产物文件实际字节数(核对体积预算 N * 1024 * 4)
    }
    return stats


# ============================================================================
#                        第 4 层:命令行入口
# ============================================================================
def _read_args(argv):
    """
        解析命令行参数,返回归一化的字典。

        功能实现原理:
            ★ 必须同时认"空格写法"和"等号写法":若判断写成
            `if '--limit' not in argv`,字符串 '--limit=2000' 与 '--limit' 不相等,
            会被误判成"没写 --limit"-> 本想只跑 2000 块,实际会跑起整个语料。
            所以用 arg.startswith('--limit=') 先扫一遍等号写法。

        参数:
            argv: list[str] —— 一般传 sys.argv[1:](不含脚本名)
        返回:dict —— {'full': bool, 'limit': int|None, 'batch': int|None, 'no_resume': bool}

        明确不做:不负责 logger 配置(那是 __main__ 的事)。
        """
    full = '--full' in argv                # 是否带 --full
    no_resume = '--no-resume' in argv      # 是否带 --no-resume
    limit = None                           # 默认不限
    batch = None                           # 默认用 config 值
    for i, arg in enumerate(argv):         # 遍历参数(带下标,便于取"下一个参数")
        if arg.startswith('--limit='):     # 等号写法 --limit=2000
            limit = int(arg.split('=', 1)[1])  # 取等号右边转整数
        elif arg.startswith('--batch='):   # 等号写法 --batch=32
            batch = int(arg.split('=', 1)[1])  # 同上
        elif arg == '--limit' and i + 1 < len(argv):  # 空格写法 --limit 2000
            limit = int(argv[i + 1])       # 取下一个参数作数值
        elif arg == '--batch' and i + 1 < len(argv):  # 空格写法 --batch 32
            batch = int(argv[i + 1])       # 同上
    return {'full': full, 'limit': limit, 'batch': batch, 'no_resume': no_resume}  # 打包返回


if __name__ == "__main__":                # 只有"直接运行本文件"才执行(被 import 时不执行)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")  # 只在入口配日志,别放顶层
    args = _read_args(sys.argv[1:])       # 解析命令行参数(sys.argv[0] 是脚本名,从 [1:] 开始)
    if args['full']:                      # 带 --full:跑全量(L 规模基线档,D7 用)
        in_path, out_path = config.CHUNKS_FILE, config.VECTORS_FILE  # 换成全量输入/输出
    else:                                 # 默认:跑调试子集
        in_path, out_path = config.CHUNKS_SUBSET_FILE, config.VECTORS_SUBSET_FILE  # 子集输入/输出
    batch_size = args['batch'] or config.EMBED_BATCH_SIZE  # 命令行没给就回落 config 默认
    stats = embed_chunks(                 # 调主循环
        in_path=in_path,                  # 输入 chunks 文件
        out_path=out_path,                # 输出向量文件
        batch_size=batch_size,            # 批大小
        limit=args['limit'],              # 行数上限(None=不限)
        resume=not args['no_resume'],     # --no-resume 取反 = 是否允许续跑
    )
    logger.info("D3 向量化完成:%s", stats)  # 打印最终统计(含 rows_per_sec,D7 排期要用)




















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

对应:路线计划 D3 | 架构 §3.4 | 开发指引 D3开发指引.md
运行:python -m offline.embed              (跑 subset,默认可断点续跑)
     python -m offline.embed --full       (D7 全量)
     python -m offline.embed --limit 2000 (只跑前 2000 块,调试用)
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
            requests.Session.__init__():requests 的会话对象,自动管理 cookies、headers、连接池。
            本函数不做任何鉴权(本机 Ollama 服务无需鉴权),也不改 timeout
            (超时是每次请求各自指定的,见 _post_batch)。

        参数:无
        返回:requests.Session —— 全局唯一实例(多次调用返回同一个对象)

        明确不做:
            不做重连状态判断 —— Session 遇到网络错误会自己标记连接失效,下一次 post 自动重连。
        """
    global _SESSION
    if _SESSION is None:
        #首次调用才真正创建；之后每次调用直接复用
        _SESSION = requests.Session
    return _SESSION

def _count_rows(path:Path) ->int:
    """
        快速统计 JSONL 文件的行数(给 memmap 预分配 shape 用)。

        实现原理:
            memmap 必须预先知道 (行数, 维度),所以要先数一遍行数。
            为什么不用 sum(1 for line in f):那样会触发 Python 层逐行解码,
            2.6 GB 的文件要几秒到十几秒;而这里按 1 MB 二进制块读,
            直接数块内 \n 的字节出现次数,完全不过 Python 的字符串解码,快得多。
            因为本项目行内容是 UTF-8,而 UTF-8 的多字节序列里不会出现 0x0A(\n),
            所以"数 \n 字节"严格等于"数行数",不存在误判。

        调用的外部方法:
            path.open('rb'):二进制模式打开,避免任何编码/换行转换干扰字节统计
            bytes.count(sub):返回子串出现次数,C 层实现,极快

        参数:
            path: Path —— 待统计的文件(一般是 data/chunks/*.jsonl)

        返回:int —— 非空行数(文件末尾若缺换行符会补 1 行;空文件返回 0)

        明确不做:
            不做 json.loads —— 这里只数行,不关心内容合法性;坏行由 iter_jsonl 负责报错。
        """
    total = 0
    last_byte = b''         # 记住最后一个字节，用来判断文件末尾有没有换行
    with open(path,'rb') as f:
        while True:
            buf = f.read(1 << 20)       #每次读1MB
            if not buf:                 # 读完了-读到文件末尾返回空bytes
                break;
            total += buf.count(b'\n')   # 统计这一块有多少换行符
            last_byte = buf[-1:]        # 更新最后一个字节
    # 边界处理：最后一行没有换行符时，count会少算一行
    if last_byte and last_byte != b'\n':
        total += 1
    return total

def _progress_path(out_path:Path) -> Path:
    """
       由产物路径推出进度文件路径:xxx.vec.npy -> xxx.vec.npy.progress.json

       参数:out_path: Path —— 向量产物路径(通常是 data/vectors/*.vec.npy)
       返回:Path —— 进度文件路径

       实现原理:
           用 Path(...) 包回去,而不是字符串拼接后忘了转 Path。
           刻意【不用】 out_path.with_suffix():那样会把 .vec.npy 的最后一个
           后缀替换掉,变成 xxx.progress.json,和 .npy 产物同名冲突风险更高。
       """
    return Path(str(out_path) + _PROGRESS_SUFFIX)

def _load_progress(out_path:Path,total:int,dim:int,batch_size:int) -> int:
    """
        读取断点续跑的位置:返回从第几行开始继续。

        实现原理:
            续跑点的可信前提是"这次的运行参数和上次完全一样"。只要 total / dim /
            batch_size 有任何一个对不上,上次写到第 k 行的含义就变了(比如批大小从 64
            改成 32,虽然行边界不受影响,但为省纠纷一律作废重来)。
            所以这里采取"严格校验 + 宽松降级":校验不过就返回 0,宁可重跑也不冒险错位。

        调用的外部方法:
            Path.exists() / Path.read_text() / json.loads()

        参数:
            out_path:    Path —— 向量产物(进度文件由它推出来)
            total:       int  —— 本次计划写入的总行数
            dim:         int  —— 本次的向量维度
            batch_size:  int  —— 本次的批大小

        返回:int —— 已完成的 rows_done(0 表示从头跑)

        明确不做:
            不做"部分有效"的推断 —— 中间状态一律作废,因为猜错一次的代价是整个矩阵错位。
        """
    p = _progress_path(out_path)
    if not p.exists():
        return 0
    try:
        info = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"进度文件损坏({e}),本次从第0行重跑")
        return 0
    if info.get('total') != total or info.get('dim') != dim or info.get('batch_size') != batch_size:
        logger.warning(f"进度文件与本次参数不匹配(上次total=%s/dim=%s/batch=%s)",
                       info.get('total'),info.get('dim'),info.get('batch_size'))
        return 0
    rows_done = int(info.get('rows_done',0))
    return max(0,min(rows_done,total))

def _save_progress(out_path:Path,rows_done:int,total:int,dim:int,batch_size:int) -> None:
    """
        原子地记录"已经写完多少行"(断点续跑的关键)。

        实现原理(为什么必须"先写临时文件再 rename"):
            如果直接 open(p,'w') 写,job 在写到一半时被 Ctrl+C / 断电终止,
            磁盘上会留下半截 JSON;下次启动 json.loads 直接抛错,断点信息全丢。
            先写 .tmp 再 os.replace() 是 POSIX 保证的原子操作:
            目标文件要么保持旧内容、要么整个换成新内容,绝不会出现中间态。

        调用的外部方法:
            json.dumps(obj, ensure_ascii=False) —— 写 UTF-8 明文,便于人手查看
            Path.write_text / os.replace(src, dst) —— Windows/POSIX 都原子

        参数:
            out_path:   Path —— 向量产物路径(进度文件由它推出来)
            rows_done:  int  —— 当前已写入且已 flush 的行数
            total/dim/batch_size: 本次运行参数,下次续跑时用来校验一致性

        返回:None
        """
    p = _progress_path(out_path)
    tmp = Path(str(p) + '.tmp')
    info = {
        'rows_done':rows_done,
        'total':total,
        'dim':dim,
        'batch_size':batch_size
    }
    tmp.write_text(json.dumps(info,ensure_ascii=False) + '\n',encoding='utf-8')
    os.replace(tmp,p)

def _post_batch(texts:list[str]) -> np.ndarray:
    """
        第 1 层核心:把"一批"文本交给 Ollama,返回原始向量(未归一化)。

        实现原理:
            一次 HTTP POST 带多条文本,由 Ollama 内部组织成一次 GPU batch:
            模型权重只要读一遍显存就能被这批数据复用,摊薄了 kernel launch 与访存开销
            (就是 batch=1 比 batch=64 慢一个量级的原因)。
            失败按 2 / 4 / 8 秒指数退避重试:网络抖动几百毫秒自愈,
            而"模型被别的进程挤出显存"恢复得更久,指数退避能用最小的总等待覆盖两种场景。

        调用的外部方法:
            _get_session().post(url, json=..., timeout=...):
                requests.Session.post —— 发 POST;json= 参数自动序列化 dict 并带 Content-Type
                timeout 是从"发请求"到"收完响应"的总时长上限。首次调用要把 1.2GB 模型
                  加载进显存(几十秒),所以给 300 秒(config.EMBED_TIMEOUT)。
            r.raise_for_status():4xx/5xx 抛 requests.HTTPError —— 200 不代表内容对
            r.json():解析响应体; 必须放在 try 里(见下方"其它注意事项")
            np.asarray(rows, dtype=np.float32):
                Python 浮点默认是 float64,不显式写 dtype 会得到 float64 矩阵
                ——体积翻倍(全量 20GB)且 faiss 的 add() 拒收。

        参数:
            texts: list[str] —— 一批文本,长度应 <= batch_size(本函数自身不再分批)

        返回:np.ndarray,形状 (len(texts), config.EMBED_DIM),dtype=float32(未归一化)

        明确不做:
            不做归一化 —— 那是 _l2_normalize 的职责,分层是为了让各自可独立测试。

        其它注意事项:
            1)  行数校验是整个离线作业最重要的一行防御。若服务端少返回一行,
               np.vstack 会安静地少拼一行,于是矩阵从第 i 行起整体错位,
               要等到 D4 的 sanity_check 才发现,而那时已经白跑了几小时。
            2) Ollama 的 embeddings 顺序与 input 顺序一致 —— 这是服务端的契约;
               上面那条行数校验就是防止它(或代理层)违反这条契约。
        """
    payload = {'model':config.EMBED_MODEL,'input':texts}
    last = None
    for attmpt in range(config.EMBED_MAX_RETRIES + 1):      #range(N+1):第0次是正常请求，后面N次是重试，合计N+1次机会
        try:
            r = _get_session().post(
                f"{config.OLLAMA_URL}/api/embed",
                json=payload,
                timeout=config.EMBED_TIMEOUT,
            )
            r.raise_for_status()        #4xx/5xx 在这里变成异常状态，执行下面的except
            rows = r.json()['embeddings']
            if len(rows) != len(texts):
                raise ValueError(f"OLLAMA返回{len(rows)}行，期望{len(texts)}行")
            return np.asarray(rows,dtype=np.float32)
        except Exception as e:
            #统一兜住三种异常：requests 网络异常 / HTTPError / .json()解析失败/行数不符
            # 它们都适合重试，因为都可能是服务暂时不可用
            last = e
            if attmpt < config.EMBED_MAX_RETRIES:
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
    # 两步合成一行，转类型 + 保证连续（两个约束都有这一个调用满足）
    vecs = np.ascontiguousarray(vecs,dtype=np.float32)
    #（N + 1）:keepdims = True 才能按正确的轴广播
    norms = np.linalg.norm(vecs,axis=1,keepdims=True)
    # 0行 superior -> 设置 1e-12，避免除以0，得到nan
    norms = np.maximum(norms,_EPS)
    vecs /= norms
    return vecs


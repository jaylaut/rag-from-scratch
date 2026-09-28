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

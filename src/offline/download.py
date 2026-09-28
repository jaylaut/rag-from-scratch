# -*- coding: utf-8 -*-
"""D0 - download.py:数据获取(离线流水线第一步)

职责:
    1. 从维基媒体官方源下载中文维基 multistream dump data/raw/
    2. 断点续传(HTTP Range):中断后重跑同一命令,从断点接着下
    3. 完整性校验:下载完对照官方公布的哈希值,通过才把 .part 改名为正式文件
       (维基官方 dump 页面提供 md5/sha1 校验文件;本文件 verify_file 支持
        md5/sha256,到时官方给哪种就校哪种,缺的算法补两行即可)

对应:路线计划 D0 | 架构 §3.1
运行(src 目录或 PyCharm):python -m offline.download

重要：如果配置pycharm下载维基百科，过程麻烦。本次只是为了学习流式下载+断点续传+sha256校验，因此选择其他国内镜像。
zhwiki-latest-pages-articles-multistream.xml.bz2 使用浏览器已下载到本地

【当前练习状态 2026-09-19】DUMP_URL 暂指向阿里云镜像上的 Python-3.12.0.tar.xz
(约 26MB),拿这个小文件调试"流式下载 + 断点续传 + 哈希校验"三件套;跑通后把
DUMP_URL 换回上面注释里的维基地址,再接真实 dump(约 2.5GB)。
"""
import os.path           # 只用到 os.path.exists / os.path.getsize,判断 .part 是否存在及大小
import hashlib           # 标准库哈希算法库:md5 / sha256 对象都从这里创建
import requests          # 第三方 HTTP 客户端:发请求 + 流式(iter_content)读响应体

# ---------- 下载地址 ----------
# 正式目标:中文维基 multistream dump(约 2.5GB,.bz2 压缩包;
# 后续 parse 阶段不需要解压,可以直接流式读 bz2)
# DUMP_URL = ("https://dumps.wikimedia.org/zhwiki/latest/"
#             "zhwiki-latest-pages-articles-multistream.xml.bz2")
# 备用小文件:清华镜像上的同款 Python 源码包(练手时也可以换着用)
# https://mirrors.tuna.tsinghua.edu.cn/python/3.12.0/Python-3.12.0.tar.xz
# 【当前练习用】阿里云镜像上的 Python 3.12.0 源码包(约 26MB);
# 它的官方 sha256 可在 python.org 发布页或镜像站同名目录的校验文件里找到
DUMP_URL = ("https://mirrors.aliyun.com/python-release/source/Python-3.12.0.tar.xz")


def verify_file(filepath,expected_hash,algorithm='sha256'):
    '''
    验证文件完整性
    用途:下载完成后核对文件是否损坏/被篡改——校验通过才允许把 .part
         改名为正式文件,保证"正式文件名下永远是完整文件"。
         (与 common/utils.py 里的 sha1_of_file 是同一个套路:
          utils 那个固定 sha1、返回摘要字符串;这个可选算法、返回布尔值,
          维基官方 dump 提供 md5/sha1,两种写法到时候按需取用)

    :param filepath: 本地文件路径(字符串或 Path);文件不存在会抛 FileNotFoundError
    :param expected_hash: 从官网复制的哈希字符串(十六进制;大小写不限,
                          内部统一转小写后再比)
    :param algorithm: 哈希算法,支持 'md5' 或 'sha256'(默认 sha256);
                      传别的值直接抛 ValueError
    :return: 布尔值,True 表示校验通过
    '''
    # 1.创建哈希对象
    # (小技巧:这段 if/elif 也可以用一行 hashlib.new(algorithm) 替代,
    #  还能顺带支持 sha1/sha512;保留显式写法便于理解"算法对象"这个概念)
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError('仅支持 md5 和 sha256')

    # 2.以二进制模式打开文件，分块读取更新哈希值
    '''
    分块读取可以处理大文件，避免内存溢出
    (这里的固定套路:iter(函数, 哨兵值) 反复调用 f.read(8192),
     直到读到 b''(文件末尾)为止——和下面 download 里的 iter_content
     是同一种"流式"思想:内存里任何时刻只有 8KB)
    '''
    '''
    以二进制只读模式打开文件，并把文件对象赋值给 f。
    'rb' 里的 r 是 read，b 是 binary。哈希计算必须用二进制模式，因为文本模式会自动转换换行符，改变字节内容，导致哈希值错误。
    with 保证文件用完自动关闭，即使中间出错也不会泄漏文件句柄
    '''
    with open(filepath,'rb') as f:
        '''
        这是整个循环的核心，它等价于“不断读取 8192 字节，直到读出来是空字节串为止”。

        f.read(8192)：每次从文件当前位置读取最多 8192 字节。读到文件末尾时，会返回空字节串 b''。

        lambda: f.read(8192)：把“读取一次”这个动作包装成一个无参函数，每次调用它才会真正读一次。

        iter(callable, sentinel)：这是 iter 的第二种用法。它会反复调用 callable，每次把返回值作为下一个迭代元素，直到返回值等于 sentinel（这里是 b''）时停止。
        '''
        for chunk in iter(lambda: f.read(8192),b''):
            hasher.update(chunk)
    # 3.计算最终的十六进制摘要，并与期望值对比
    # (hexdigest() 返回 64 个字符的小写十六进制串,如 '9d9eb270...')
    actual_hash = hasher.hexdigest()

    # 4.比对时不区分大小写
    # (有些官网把哈希值印成大写,先统一 .lower() 再比,
    #  避免"看起来一样却判为失败"的乌龙)
    match = actual_hash.lower() == expected_hash.lower()

    print(f"文件:{filepath}")
    print(f"期望哈希:{expected_hash}")
    print(f"实际哈希:{actual_hash}")
    print(f"校验结果:{'通过' if match else '失败，文件可能已损坏或被篡改'}")
    return match

def download(save_path,chunk_size=8192):

    '''
    流式下载 + 断点续传
    用途:把 DUMP_URL 指向的文件下载到 save_path;中断后重跑同一命令,
         若发现 .part 文件就从断点继续(HTTP Range 请求头);
         结果一律先写成 save_path.part,校验通过后再改名(改名逻辑见 main 的 TODO)。
         (与路线计划的 download(url, dest, expected_sha1) 相比:这版把 url 放到了
          模块常量 DUMP_URL、把校验拆成了独立的 verify_file——分工不同,思路等价)

    :param save_path: 保存路径(字符串,含文件名,如 'data/raw/xxx.tar.xz')。
                      注意:相对路径是相对"运行命令时所在目录"而言的——
                      按 src 目录下 python -m offline.download 的约定,
                      会写到 src/data/raw/ 而不是项目根的 data/raw/,
                      详见 main 里的路径提醒
    :param chunk_size: 每次从网络读取并写入磁盘的字节数(默认 8192 = 8KB)。
                      调大 → 写盘次数更少、整体略快、进度条刷新更粗;
                      调小 → 进度更细腻、Python 循环开销略大;
                      内存占用两者都只有这一块,不影响流式特性
    :return: 无(None)。side effect:生成/追加 save_path.part 文件
    '''
    # .part 约定:所有未完成、未校验的数据都进 .part 文件;
    # 正式文件名只有校验通过后才出现——防止把损坏的半成品当成品用
    part_path = save_path + '.part'
    # downloaded 一变量两用:①续传的起点(已有字节数) ②进度条的分子(随写入累加)
    downloaded = 0
    # 1.检查是否已有部分下载的文件
    # (注意:这里只看文件大小、不校验内容——极端情况 .part 自身损坏,
    #  会在最终哈希校验时暴露,属于"先续传、后总验"的合理取舍)
    if os.path.exists(part_path):
        downloaded = os.path.getsize(part_path)
        print(f"发现已有 .part文件，已下载{downloaded}字节，将尝试断点续传")

    # 2.构建Range 请求头
    headers = {
        # 添加浏览器 User-Agent，避免被镜像站拦截（解决 403）
        # (requests 默认 UA 是 'python-requests/x.x',部分镜像站看到直接 403)
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    # Range: bytes=N- 表示"从第 N 字节开始给我,直到文件末尾"
    # N 就是 .part 已有的字节数——断点续传的全部魔法就这一个请求头,
    # 服务器收到后如果支持,会只回传缺失的后半段
    if downloaded > 0:
        headers['Range'] = f"bytes={downloaded}-"

    # 3 发送请求
    # stream=True:先只取响应头,响应体不急着下载,后面用 iter_content 一块一块取
    #             ——内存里永远只有一块 chunk,这就是"流式"下载的本质
    # timeout=30:连接建立/相邻两次收到数据之间的最长等待 30 秒,
    #             不是总时长限制——所以 2.5GB 下多久都行,只要数据别断流 30 秒
    resp = requests.get(DUMP_URL,headers=headers,stream=True,timeout=30)
    # 4xx/5xx(如 403/404)在这里直接抛异常,避免把错误页面写进 .part 污染数据
    resp.raise_for_status()


    # 4 判断服务器是否支持断点续传
    if downloaded > 0 and resp.status_code == 206:
        # 服务器返回206，表示支持Range
        # 只是"剩余部分"的大小,所以总大小 = 已下载 + 剩余,两个都要
        total_size = downloaded + int(resp.headers.get('content-length',0))
        print(f"服务器支持断点续传，总大小约为{total_size} 字节")
    elif downloaded > 0 and resp.status_code == 200:
        # 服务器返回200，表示不支持Range，只能从头开始下载
        print("服务器不支持断点续传，将从头开始下载")
        downloaded = 0
        total_size = int(resp.headers.get('content-length',0))
        # 重写以写入模式打开
        # (resp.close() :stream=True 的响应不关会一直占着连接;
        resp.close()
        resp = requests.get(DUMP_URL,headers=headers,stream=True,timeout=30)
        resp.raise_for_status()
    else:
        # 首次下载(没有 .part,也不带 Range)
        # total_size=0 表示服务器没报文件大小(分块传输/压缩流可能没有),
        # 下面的进度条遇到 0 会自动退化为"只显示字节数",不会除零报错
        total_size = int(resp.headers.get('content-length',0))
        print(f"开始下载，总大小{total_size} 字节")

    # 5.流式写入 .part文件（追加模式）
    # 'ab' = 追加写(续传:新数据接在旧字节后面,不会覆盖已有进度)
    # 'wb' = 覆盖写(从头下载:把可能的旧残留全部清掉重写)
    # 带 'b' 是二进制模式——网络传来的本来就是原始字节,不做任何文本编解码
    mode = 'ab' if downloaded > 0 else 'wb'
    with open(part_path,mode) as f:
        # iter_content 每次最多吐 chunk_size 字节;
        # if chunk 过滤掉空包(连接复用的 keep-alive 心跳块)
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                #简单进度展示
                # 开头的 \r 是回车符:让光标回到行首,原地刷新同一行 → "进度条"效果
                # (如果某些终端里进度不实时刷新,给 print 加 flush=True 试试)
                if total_size:
                    precent = downloaded / total_size * 100
                    print(f"\r 进度:{precent:.1f}% ({downloaded} 字节)",end="")
            
    print("\n下载完成")
    # 【建议】到这里 .part 还没有改名成 save_path——按 main 里的完整流程,
    # 应先 verify_file(part_path, 官方哈希),通过后再 os.replace(part_path, save_path),
    # 这样"正式文件名"下就永远不会出现损坏文件
    # 【建议】stream=True 的响应用完最好显式关闭,或改用 with 语句包住整个请求,
    # 函数一退出连接自动释放(目前靠 Python 回收机制兜底,能跑但不够规范)

if __name__ == "__main__":
    # TODO(D0): 校验已有文件是否完整 → 不完整则断点续传 → 最终校验并报告
    # 建议补全的完整流程(三个函数都已就绪,串起来即可):
    #   1) 若 save_path 已存在 → verify_file 校验,通过就直接结束
    #   2) 不完整或不存在 → download() 续传/下载
    #   3) 下载完 → verify_file(save_path + '.part', 官方哈希)
    #      → 通过后 os.replace('.part 文件', save_path) 完成改名
    # (练手素材:data/raw/ 下已有浏览器下载好的维基 dump 本体,
    #  可以拿它练 verify_file——官方校验值在 dumps 页面同名目录的
    #  md5/sha1 文件里;或直接用 common.utils.sha1_of_file)
    # verify_file('data/raw/python-3.14.7-amd64.exe',
    #             '9d9eb2709ef81bf5cd30db3c2096bdbc4ea10087c22e62f27d356b36f6ae9649',
    #             'sha256')
    # 【注意·路径】'data/raw/...' 是相对路径,基准是"运行命令时所在的目录"——
    # 在 src 目录下跑 python -m offline.download 时会写到 src/data/raw/,
    # 而不是项目根的 data/raw/(项目约定所有路径都从 common.config 取,
    # 那里的路径基于文件位置推算,在任何目录运行都正确)。
    # 更稳的写法:
    #   from common.config import DATA_RAW
    #   download(str(DATA_RAW / 'Python-3.12.0.tar.xz'))
    download('data/raw/Python-3.12.0.tar.xz')


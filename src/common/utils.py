# -*- coding: utf-8 -*-
"""通用小工具:JSONL 流式读写 / 大文件 sha1 —— 离线各模块复用"""
import hashlib
import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict]:
    """逐行流式读 JSONL(内存安全,8GB 语料也不怕);坏行带行号报错"""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {i} 行不是合法 JSON: {e}") from e


def append_jsonl(path: Path, obj: dict) -> None:
    """追加一行 JSON(自动建父目录;断点续跑场景用)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """大文件 sha1(1MB 分块读,不占内存)——下载校验用"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

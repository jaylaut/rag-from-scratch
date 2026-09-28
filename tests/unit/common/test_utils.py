# -*- coding: utf-8 -*-
'''tests/unit/common/test_utils.py —— 对应 src/common/utils.py(JSONL 流式读写、sha1)

当前状态:占位骨架。src/common/utils.py 【已经实现】,随时可以写真用例。

建议优先写的几条(按经验排序,越靠前越值得写):

  1. iter_jsonl 是**流式**的 —— 这是它存在的全部意义。
     怎么验不是恒真断言:用一个 tmp_path 里的多行文件,断言迭代器在
     还没读到末尾时就已经产出了前面的行(而不是一次读进内存).
     更稳的做法:文件里故意在**最后一行**放坏 JSON,断言前面的行已经
     被正常 yield 出来、且最后才抛 ValueError 且带行号。

  2. iter_jsonl 会跳过空行、会给坏行报出正确的行号。
     ★ 行号断言是本项目最值得测的东西之一:D1 合并阶段踩过"分片末行缺换行
       → 两行粘成一行"的坑,行号错了排查会非常痛苦。

  3. append_jsonl 会自建父目录,且每行以 \\n 结尾。
     ★ Windows 上别用 write_text 写这些文件 —— 它会把 LF 静默转成 CRLF,
       本项目所有产物(含 JSONL)统一 LF。

  4. sha1_of_file 对同一内容两次结果一致,且分块读不会因为文件大于 chunk_size
     而变化(tmp_path 里造个 >1MB 的文件即可,但注意别造太大)。

 fixture 提示:用 pytest 内置的 tmp_path(每个用例一个干净临时目录),
 别往 tests/fixtures/ 里写东西 —— 那个目录只放预先手搓的静态样本。
'''
import pytest

pytestmark = pytest.mark.unit


def test_placeholder() -> None:
    '''占位用例:让 pytest 能收集到本文件。写了真实用例后删掉这一个。'''
    pytest.skip('utils.py 已实现,可直接写真用例 —— 待编写(见本文件 docstring 的 4 条建议)')

# -*- coding: utf-8 -*-
'''pytest 全局配置(本项目唯一的 conftest,对 tests/ 下所有用例生效)

只做两件事:

1. 把 src/ 加进 sys.path。
   为什么必须这么做:src 下的模块用的是"扁平导入"——比如 offline/parse.py 里
   写的是 `from common import config`,这要求 **src 目录本身** 在 sys.path 上,
   而不是项目根。项目的命令行运行约定 `cd src && python -m offline.parse`
   恰好就是靠"当前目录 = src"来满足这一点;pytest 没有这个前提,所以要补上。
   正常情况下 pytest.ini 里的 `pythonpath = src` 已经做了这件事,这里是
   第二道保险:万一在别的目录或 IDE 里直接跑 pytest、pytest.ini 没被认成
   rootdir 配置,也不至于报 ModuleNotFoundError。

2. 提供共用 fixture(目前只有 fixtures_dir,后面按需加)。

★ 注意:tests/ 下所有目录都【不建】__init__.py。当前 test_*.py 的文件名互不
  重复,pytest 默认的 prepend 导入模式足够;加了 __init__.py 反而会把
  tests 变成包,在不同运行方式(命令行 / PyCharm)下更容易出现导入差异。
'''
import sys
from pathlib import Path

import pytest

# tests/conftest.py 的上级就是项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'

# 用 not in 判断后再插入,避免重复启动时 sys.path 里堆一串同样的路径
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope='session')
def fixtures_dir() -> Path:
    '''返回 tests/fixtures/ 目录:放手搓的小样本合成数据,总量控制在几 KB

    ★ 铁律:严禁把 data/articles/wiki_zh.jsonl(2.32GB)或 data/ 下任何真实产物
      拷进 fixtures/。需要真实数据的用例请这样写:

          @pytest.mark.bigdata
          @pytest.mark.skipif(not config.ARTICLES_FILE.exists(),
                              reason='需要 E 盘全量语料')
          def test_xxx(): ...

      scope='session' 表示整轮测试只算一次路径,不用每个用例都重算。
    '''
    return Path(__file__).resolve().parent / 'fixtures'

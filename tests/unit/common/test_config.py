# -*- coding: utf-8 -*-
'''tests/unit/common/test_config.py —— 对应 src/common/config.py(路径与参数常量)

当前状态:占位骨架。注意 src/common/config.py 【已经实现】,随时可以写真用例。

建议优先写的几条(都很便宜,但每条都挡住过一个真坑):

  1. 6 个切片常量取值正确
     CHUNK_MIN_CHARS / CHUNK_MAX_CHARS / CHUNK_OVERLAP / CHUNK_OVERLAP_MAX /
     CHUNK_HARDCUT_CHARS / CHUNK_TAIL_MERGE / CHUNK_MERGE_LIMIT
     ★ 历史上真出过事:补充新常量时忘了删掉旧的 500/50 参数块,
       结果 CHUNK_MAX_CHARS 被 500 静默覆盖(运行时验证才抓出来)。
       断言要写死 550,不要写成 `!= 500` 这种能被各种值蒙混过关的形式。

  2. 路径常量全部以 PROJECT_ROOT 为基准
     config 的核心约定是“路径基于文件位置自动推算、与当前工作目录无关”;
     验证方法:断言每个路径常量都是 PROJECT_ROOT 的子路径(startswith)。

  3. ARTICLES_FILE / CHUNKS_FILE / CHUNKS_SUBSET_FILE 落在 data/ 下的正确位置
     —— D2 的 chunk_all 直接依赖它们,写错就会静默读写到错误目录。

  4. 隐含关系:CHUNK_MIN_CHARS < CHUNK_MAX_CHARS <= CHUNK_MERGE_LIMIT 等
     —— 这类"常量之间的关系"比单个值的断言更能挡住手滑。
'''
import pytest

pytestmark = pytest.mark.unit          # 整个文件都打 unit 标记


def test_placeholder() -> None:
    '''占位用例:让 pytest 能收集到本文件。写了真实用例后删掉这一个。'''
    pytest.skip('config.py 已实现,可直接写真用例 —— 待编写(见本文件 docstring 的 4 条建议)')

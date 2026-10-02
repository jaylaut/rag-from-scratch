# fixtures/ —— 合成小样本专用目录

## 放什么

手搓的、几 KB 以内的假数据,用于把被测函数喂进去跑断言。例如:

- 一小段带 `\n\n` 段落边界的假正文(给 `offline/chunk.py` 的句子流/滑窗用);
- 3~5 行的临时 JSONL(给 `common/utils.py` 的 `iter_jsonl` 用)。

建议直接写在测试函数里构造字符串,只有当样本较长(超过十几行)或多个用例要复用时,
才落成文件放这里。

## 不放什么(铁律)

- ❌ `data/articles/wiki_zh.jsonl`(GB 级,本机实测约 2.3~3.8 GB)或它的任何切片拷贝;
- ❌ `data/cleaned/`、`data/chunks/` 下的任何真实产物;
- ❌ 超过几十 KB 的东西。

真实数据的正确用法写在 `tests/conftest.py` 的 `fixtures_dir` fixture 注释里:
`@pytest.mark.bigdata` + `@pytest.mark.skipif(not path.exists())`,
日常 `-m "not bigdata"` 就自动跳过。

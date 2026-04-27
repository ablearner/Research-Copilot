# Research-Copilot SciGraphQA 入库方案

这份文档说明如何把 SciGraphQA 数据整理到 `Research-Copilot` 当前的向量和图谱底座中。

## 1. 目标

把外部 SciGraphQA 数据转成当前系统可消费的：

- 文本块
- 图表对象
- embedding
- graph extraction 结果

## 2. 当前入口

仓库内已有脚本：

- [scripts/ingest_scigraphqa.py](scripts/ingest_scigraphqa.py)

它会把 parquet 行映射成当前系统的内部对象，并写入当前配置的：

- vector store
- graph store

## 3. 默认数据源

脚本默认外部数据根目录：

```text
/home/myc/ChartGPT/data/raw/scigraphqa
```

这只是数据来源，不代表当前项目依赖 ChartGPT 运行。

## 4. 运行示例

```bash
cd <repo-root>
python scripts/ingest_scigraphqa.py --limit 100
```

只做映射检查、不写数据库：

```bash
cd <repo-root>
python scripts/ingest_scigraphqa.py --limit 20 --dry-run
```

## 5. 当前依赖

运行前确认：

- `Milvus` 已启动
- `Neo4j` 已启动
- embedding / LLM provider 配置正确

## 6. 当前结论

SciGraphQA 入库在当前项目里属于数据接入任务，不影响 research 主链路本身。它的价值主要在于：

- 给底座问答补充评测或样例数据
- 给图谱和多模态检索提供外部数据集支持

# Research-Copilot Evaluation

这个目录用于评测 `Research-Copilot` 的当前能力。

评测建议分两层看：

- 底座能力评测：`ask_document`、`ask_fused`、`chart_understand`
- research 主链路评测：论文发现、导入收益、collection QA groundedness、TODO 质量

补充说明：

- 运行中的基础指标与失败归档默认会写到 `.data/research/observability/`
- 可用于和本目录 benchmark 结果一起看，判断“评测是否过关”以及“线上主链路是否稳定”

## 目录说明

- `schemas.py`
  评测 case 与结果结构。
- `metrics.py`
  指标计算逻辑，包括检索、groundedness、参考答案重合度和系统稳定性指标。
- `runner.py`
  负责执行 case、收集状态和汇总结果。
- `sample_runtime.py`
  低资源、可复现实验 runtime。
- `sample_cases.json`
  样例评测集。
- `run_agent_metrics.py`
  主入口，支持 `sample` 和 `live`，也支持在评测前临时加载 benchmark 知识库。
- `run_retrieval_metrics.py`
  只跑检索链路的 `Recall@k` / `Hit@k` / latency，适合先做 RAG 检索 baseline。
- `build_research_benchmarks.py`
  从 BEIR SciFact 与 RAGBench 构建可复现的 `knowledge_base.json` 和 `cases.json`。
- `ingest_benchmark_kb.py`
  把 benchmark 知识库写入当前配置的向量库。
- `summarize_benchmark_report.py`
  对已有 `run_agent_metrics.py` 报告做增强后处理，补齐 benchmark-aware 指标并输出摘要。

## 使用方式

运行 sample runtime：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py --runtime sample --recall-k 5
```

运行 live runtime：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py --runtime live --cases evaluation/sample_cases.json --recall-k 5
```

如果本地没有启动 Neo4j / MySQL，可以临时切到内存后端：

```bash
RUNTIME_BACKEND=local \
GRAPH_STORE_PROVIDER=memory \
SESSION_MEMORY_PROVIDER=memory \
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py \
  --runtime live \
  --cases evaluation/sample_cases.json \
  --recall-k 5
```

## 核心 6 项指标

- `Recall@k`
- `Route Accuracy`
- `Answer Keyword Recall`
- `Groundedness`
- `Tool Call Success Rate`
- `P50 / P95 Latency`

评测报告顶层会输出 `core_6_metrics`，用于快速观察这 6 项核心结果。完整诊断指标仍保留在 `metrics` 和逐 case 结果里，包括 `Task Success Rate`、`Hit@k`、`Validation Retry Rate` 和 `Average Steps per Task`。

## 增强指标

除了核心 6 项，当前还会在聚合结果或增强报告中看这些指标：

- `Task Success Rate`
  单 case 是否整体过关的比例。
- `Reference Answer Precision / Recall / F1`
  回答和 benchmark 参考答案在信息 token 层面的重合度，主要对 `RAGBench PubMedQA` 这类带参考答案的数据有意义。
- `Answer Polarity Accuracy`
  `yes / no / uncertain` 结论方向是否和参考答案一致，适合 `PubMedQA` 一类问答。
- `Insufficient Answer Rate`
  输出 `证据不足` / `insufficient evidence` 的比例，用来判断系统是否过于保守。
- `Warning Case Rate` / `Avg Warning Count`
  有警告的 case 比例，以及平均每条 case 的 warning 数量。
- `Error Free Rate`
  没有报错的 case 比例。

建议解读方式：

- SciFact 优先看 `Recall@k`、`Groundedness`、`Insufficient Answer Rate`。
- RAGBench PubMedQA 优先看 `Groundedness`、`Reference Answer F1`、`Answer Polarity Accuracy`。
- `Latency P50 / P95` 主要看稳定性和长尾；`P95` 明显高于 `P50` 往往说明少数 case 很慢。

## SciFact / RAGBench 评测流程

先构建外部 benchmark：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/build_research_benchmarks.py \
  --limit-scifact-cases 100 \
  --limit-ragbench-cases 100 \
  --ragbench-subset pubmedqa \
  --ragbench-split test
```

这会生成：

- `evaluation/benchmarks/scifact_v1/knowledge_base.json`
- `evaluation/benchmarks/scifact_v1/cases.json`
- `evaluation/benchmarks/ragbench_pubmedqa_test_v1/knowledge_base.json`
- `evaluation/benchmarks/ragbench_pubmedqa_test_v1/cases.json`

如果已经手动下载了 Hugging Face 文件，也可以用本地文件构建：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/build_research_benchmarks.py \
  --local-scifact-corpus path/to/scifact/corpus.parquet \
  --local-scifact-queries path/to/scifact/queries.parquet \
  --local-scifact-qrels path/to/scifact/qrels.tsv \
  --local-ragbench-file path/to/pubmedqa/test.parquet
```

然后把知识库灌入当前配置的向量库：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/ingest_benchmark_kb.py \
  --knowledge-base evaluation/benchmarks/scifact_v1/knowledge_base.json \
  --delete-existing

/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/ingest_benchmark_kb.py \
  --knowledge-base evaluation/benchmarks/ragbench_pubmedqa_test_v1/knowledge_base.json \
  --delete-existing
```

最后分别跑 live 评测：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py \
  --runtime live \
  --cases evaluation/benchmarks/scifact_v1/cases.json \
  --knowledge-base evaluation/benchmarks/scifact_v1/knowledge_base.json \
  --recall-k 5 \
  --output evaluation/benchmarks/scifact_v1/report_full_api_scifact_100.json

/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py \
  --runtime live \
  --cases evaluation/benchmarks/ragbench_pubmedqa_test_v1/cases.json \
  --knowledge-base evaluation/benchmarks/ragbench_pubmedqa_test_v1/knowledge_base.json \
  --recall-k 5 \
  --output evaluation/benchmarks/ragbench_pubmedqa_test_v1/report_full_api_ragbench_pubmedqa_100.json
```

如果只想做 smoke：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_agent_metrics.py \
  --runtime live \
  --cases evaluation/benchmarks/scifact_v1/cases.json \
  --case-id scifact_1 \
  --recall-k 5 \
  --output evaluation/benchmarks/scifact_v1/report_smoke_current_1.json
```

如果只想先看 RAG 检索质量，不跑完整多智能体回答链路：

```bash
RUNTIME_BACKEND=local \
VECTOR_STORE_PROVIDER=memory \
GRAPH_STORE_PROVIDER=memory \
SESSION_MEMORY_PROVIDER=memory \
LLM_PROVIDER=local \
EMBEDDING_PROVIDER=local \
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/run_retrieval_metrics.py \
  --cases evaluation/benchmarks/scifact_v1/cases.json \
  --knowledge-base evaluation/benchmarks/scifact_v1/knowledge_base.json \
  --recall-k 5 \
  --retrieval-mode vector \
  --output evaluation/benchmarks/scifact_v1/report_retrieval_recall5_local.json
```

如果 live 全量评测已经有报告，但想补齐增强指标和摘要，不必整套重跑：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/summarize_benchmark_report.py \
  --report evaluation/benchmarks/scifact_v1/report_full_api_scifact_100.json \
  --cases evaluation/benchmarks/scifact_v1/cases.json \
  --output evaluation/benchmarks/scifact_v1/report_full_api_scifact_100_enriched.json \
  --summary-output evaluation/benchmarks/scifact_v1/summary_full_api_scifact_100_enriched.json

/home/myc/miniconda3/envs/Research-Copilot/bin/python evaluation/summarize_benchmark_report.py \
  --report evaluation/benchmarks/ragbench_pubmedqa_test_v1/report_full_api_ragbench_pubmedqa_100.json \
  --cases evaluation/benchmarks/ragbench_pubmedqa_test_v1/cases.json \
  --output evaluation/benchmarks/ragbench_pubmedqa_test_v1/report_full_api_ragbench_pubmedqa_100_enriched.json \
  --summary-output evaluation/benchmarks/ragbench_pubmedqa_test_v1/summary_full_api_ragbench_pubmedqa_100_enriched.json
```

建议解读方式：

- SciFact 主要看 `Recall@k`，因为 qrels 是科学声明到相关论文的检索金标准。
- RAGBench PubMedQA 主要看 `Groundedness`、`Reference Answer F1`、`Answer Polarity Accuracy`，同时也能看 `Recall@k`，因为它提供支持句 key。
- `Route Accuracy`、`Tool Call Success Rate`、`Latency P50/P95` 两边都能统计，适合比较多智能体编排稳定性。
- `Insufficient Answer Rate` 和 `Warning Case Rate` 适合判断系统是不是经常保守拒答，或者经常因为证据不足触发降级。

为了保证可比性，外部 benchmark 的检索金标准优先使用 `expected_source_ids`，评测时会匹配命中的 `source_id` 或 `document_id`，不依赖向量库内部生成的 embedding id。

## 当前建议

如果你继续扩展评测，优先补 research-task 级 case，而不是只补底座问答 case。

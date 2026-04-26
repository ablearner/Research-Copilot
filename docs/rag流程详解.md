# Research-Copilot RAG 流程详解

这份文档专门解释 `Research-Copilot` 当前代码里的 RAG 流程。

目标不是介绍通用 RAG 概念，而是回答下面几个具体问题：

- 一个 PDF 进入系统后，实际经历了哪些步骤
- chunk 是怎么切的
- embedding 建了哪些索引
- 检索时有哪些召回分支
- 这些分支怎么融合、重排、转成证据
- 最终答案是怎么生成的
- research 场景是怎么接到这套 RAG 底座上的

---

## 1. 一句话总览

这个项目的 RAG 不是单路向量检索，而是一个 tool-first 的分层执行链：

```text
上传/导入 PDF
-> parse_document
-> 生成 page + text_blocks + page_images
-> embedding_index / graph_extraction / graph_index
-> hybrid retrieval(sparse + vector + graph + graph summary)
-> merge hits
-> RRF fusion
-> Cross-Encoder rerank
-> build evidence bundle
-> answer_with_evidence
-> QAResponse
```

从产品角度看，它又分成两层：

- 上层：`research` 任务编排层，负责 discovery / import / collection QA / compare / recommend
- 下层：`RagRuntime` 执行层，负责 parse / index / retrieve / answer / chart
  当前底层已经不再依赖独立 LangGraph 节点调度，而是直接通过 tool pipeline 执行。

这份文档主要讲下层，但会在最后补上它如何被 `research` 层调用。

---

## 2. 主入口在哪里

RAG 主入口在这些文件里：

- API 装配入口：[apps/api/runtime.py](../apps/api/runtime.py)
- 执行运行时：[rag_runtime/runtime.py](../rag_runtime/runtime.py)
- 文档解析器：[adapters/local_runtime.py](../adapters/local_runtime.py)
- 向量索引服务：[rag_runtime/services/embedding_index_service.py](../rag_runtime/services/embedding_index_service.py)
- 混合检索器：[retrieval/hybrid_retriever.py](../retrieval/hybrid_retriever.py)
- 稀疏检索器：[retrieval/sparse_retriever.py](../retrieval/sparse_retriever.py)
- 词法检索器：[retrieval/lexical.py](../retrieval/lexical.py)
- Cross-Encoder 重排：[retrieval/cross_encoder.py](../retrieval/cross_encoder.py)
- 答案生成链：[chains/answer_chain.py](../chains/answer_chain.py)
- 图表理解链：[chains/chart_understanding_chain.py](../chains/chart_understanding_chain.py)

`apps/api/runtime.py` 会把整套依赖装配好：

- `DocumentTools`
- `ChartTools`
- `GraphExtractionTools`
- `RetrievalTools`
- `AnswerTools`
- `GraphIndexService`
- `EmbeddingIndexService`

然后构造出一个统一的 `RagRuntime`。

---

## 3. 建库阶段总流程

当系统需要把一篇 PDF 变成“可检索知识”时，主链路是：

```text
parse_document
-> graph_extraction
-> embedding_index
-> graph_index
```

对应代码入口：

- `RagRuntime.handle_parse_document()`
- `RagRuntime.handle_index_document()`

当前这条建库链已经收敛成直接 tool/service 调用，不再依赖底层图节点调度：

- `DocumentTools.parse_document()`
- `GraphExtractionTools.extract_from_text_blocks()`
- `EmbeddingIndexService.index_*()`
- `GraphIndexService.index_graph_result()`

其中最核心的两件事是：

1. 把 PDF 转成 `ParsedDocument`
2. 把 `ParsedDocument` 变成两类索引
   - embedding 索引
   - graph 索引

---

## 4. PDF 是怎么处理的

### 4.1 真实解析器

当前实际在用的 PDF 解析器是 [adapters/local_runtime.py](../adapters/local_runtime.py) 里的 `LocalDocumentParser`，不是空的 `adapters/document_parser/pdf.py`。

`parse_document()` 会根据文件后缀分流：

- 图片：构造单页 image document
- PDF：走 PDF 解析逻辑
- 其他文本文件：按纯文本读入

### 4.2 PDF 解析顺序

如果是 PDF，处理顺序是：

1. 优先使用 `PyMuPDF`
   - `fitz.open(...)`
   - `pdf_page.get_text("text", sort=True)` 提取页文本
   - 把每页渲染成 PNG，落到 `.data/storage/page_images/<document_id>/`

2. 如果 `PyMuPDF` 失败，再退到 `PyPDF`
   - `PdfReader(...).pages`
   - `extract_text()`

3. 如果仍然没有可用文本，生成 fallback page

所以当前实现更接近：

- 文本抽取型 PDF parser
- 附带 page image 渲染
- 不是完整 OCR-first 流程

### 4.3 输出对象

PDF 解析后的统一结构是 `ParsedDocument`：

- `ParsedDocument`
  - `pages: list[DocumentPage]`
- `DocumentPage`
  - `image_uri`
  - `text_blocks`
- `TextBlock`
  - `text`
  - `page_id`
  - `page_number`
  - `block_type`

这几个 schema 在 [domain/schemas/document.py](../domain/schemas/document.py)。

---

## 5. chunk 是怎么划分的

这个项目里有两层 chunk，不要混起来。

## 5.1 第一层 chunk：`TextBlock`

这是 PDF 文本进入向量索引和图谱抽取前的基础切分。

切分逻辑在 `LocalDocumentParser._build_text_blocks()` 和 `_chunks()`：

1. 先规范化文本
   - 去掉 `\x00`
   - 统一换行
   - 压缩连续空格
   - 连续三行以上空行压成两行

2. 按空行切 paragraph

3. 每个 paragraph 再按固定长度切片

默认参数：

- 每个文本块约 `900` 字符

所以第一层 chunk 的规则可以概括成：

```text
页文本
-> normalize
-> 按空行切段
-> 超长段按 900 字符硬切
-> 生成 TextBlock
```

这意味着当前 chunk 策略是：

- 简单
- 稳定
- 适合工程落地
- 但不是语义分块，也不是版面结构感知分块

## 5.2 第二层 chunk：图谱抽取批次

图谱抽取不会把整篇文档一次喂给 LLM，而是把多个 `TextBlock` 再打包成“批次 chunk”。

实现见 [tools/graph_extraction_toolkit.py](../tools/graph_extraction_toolkit.py)。

默认参数：

- `text_graph_chunk_size = 48`
- `text_graph_chunk_chars = 24000`
- `max_text_graph_chunks = 6`
- `text_graph_timeout_seconds = 12.0`

含义：

- 一个图谱抽取批次最多 48 个 `TextBlock`
- 或总字符数超过 24000 就切新批次
- 最多只保留前 6 个批次
- 每个批次单独超时控制

所以第二层 chunk 的作用不是为了检索，而是为了：

- 控制图谱抽取的 LLM 上下文
- 降低长文档失败概率
- 允许局部失败、整体继续

### 5.3 为什么要两层 chunk

这两层职责不同：

- `TextBlock`
  - 面向 embedding 检索
  - 是最细粒度证据单位

- 图谱抽取批次
  - 面向 LLM 调用
  - 是执行稳定性单位

这就是项目里“chunk 看起来有两套”的原因。

---

## 6. embedding 索引是怎么建的

embedding 建库由 [rag_runtime/services/embedding_index_service.py](../rag_runtime/services/embedding_index_service.py) 负责。

当前会建三类 embedding：

1. `text_block` 级
2. `page` 级
3. `chart` 级

### 6.1 text block embedding

`index_text_blocks()` 会：

- 过滤掉空文本块
- 对每个 `TextBlock.text` 生成 embedding
- 写入向量库

每条记录会保留：

- `document_id`
- `page_id`
- `page_number`
- `block_type`
- `content = block.text`

这是最核心的检索粒度。

### 6.2 page embedding

`index_pages()` 会：

- 只处理有 `image_uri` 的页
- 把 `page_image + page_text` 一起编码

这一步是为了支持：

- page image 检索
- 图像页级证据
- 视觉相关问答

### 6.3 chart embedding

`index_charts()` 会：

- 读取 chart image
- 用 `chart summary/title` 作为语义文本
- 写入 chart embedding

### 6.4 向量库后端

向量库后端通过适配器切换，当前支持：

- Milvus
- PgVector
- Qdrant
- 本地内存实现

装配入口在 [apps/api/runtime.py](../apps/api/runtime.py)。

---

## 7. GraphRAG 部分是怎么建的

图谱建库分两步：

1. `graph_extraction`
2. `graph_index`

### 7.1 graph extraction

图谱抽取由 [tools/graph_extraction_toolkit.py](../tools/graph_extraction_toolkit.py) 调用 [chains/graph_extraction_chain.py](../chains/graph_extraction_chain.py) 完成。

输入是：

- `document_id`
- `text_blocks`
- `page_summaries`

输出是 `GraphExtractionResult`：

- `nodes`
- `edges`
- `triples`
- `status`

对于多批次 chunk：

- 每个批次单独抽取
- 局部失败会记录为 `partial` 或 `failed`
- 最后通过 `merge_graph_candidates_run()` 合并

### 7.2 graph index

`graph_index_service` 会把抽取出的结果写入图存储。

当前图存储支持：

- Neo4j
- 本地内存图存储

所以这个项目的 GraphRAG 不是直接把文本塞图数据库，而是：

```text
TextBlock
-> LLM 抽 triples
-> GraphExtractionResult
-> GraphStore
```

---

## 8. 查询阶段总流程

查询时的核心路径是：

```text
question
-> RetrievalTools.retrieve()
-> HybridRetriever.retrieve()
-> 并行召回
   - sparse
   - vector
   - graph
   - graph summary
-> merge_hits()
-> apply_rrf()
-> Cross-Encoder rerank
-> build_evidence_bundle()
-> AnswerChain.ainvoke()
-> QAResponse
```

---

## 9. 混合检索怎么做

混合检索的核心在 [retrieval/hybrid_retriever.py](../retrieval/hybrid_retriever.py)。

### 9.1 输入

统一查询对象是 `RetrievalQuery`，包括：

- `query`
- `document_ids`
- `mode`
- `top_k`
- `filters`
- `graph_query_mode`

### 9.2 并行召回分支

`HybridRetriever.retrieve()` 会根据模式并行调度：

- `SparseRetriever`
- `VectorRetriever`
- `GraphRetriever`
- `GraphSummaryRetriever`

默认常见模式是 `hybrid`，也就是四路里的可用分支一起跑。

### 9.3 各分支的作用

#### sparse

`SparseRetriever` 依赖向量库的稀疏文本检索接口。

默认主要搜这些 source type：

- `text_block`
- `page`
- `graph_summary`

它更擅长：

- 精确词匹配
- 专有名词
- 公式名、缩写、术语

#### vector

`VectorRetriever` 会：

- 先把 query 编码成 embedding
- 再到向量库里做向量相似度搜索

它更擅长：

- 语义相近
- 表述改写
- query 与原文不完全同词

#### graph

`GraphRetriever` 会：

- 先从问题里抽 entity-like keywords
- 对图存储执行 `query_subgraph`
- 在 `entity/auto` 模式下继续按关键词搜实体
- 在 `subgraph/auto` 模式下补 subgraph 命中

它输出的 hit 可能是：

- `graph_node`
- `graph_edge`
- `graph_triple`
- `graph_subgraph`

#### graph summary

`GraphSummaryRetriever` 检索的是 community summary，而不是原始节点边。

它适合：

- 宏观主题概括
- 跨块整合
- GraphRAG summary 视角

---

## 10. 多路结果怎么融合

融合逻辑在 [retrieval/fusion.py](../retrieval/fusion.py)。

### 10.1 merge key

不同来源召回的结果不是简单拼接，而是先按 `merge_key(hit)` 合并。

例如：

- 同一个 `text_block`
- 同一个 `page`
- 同一个 `chart`
- 同一个 `graph_summary`

会被折叠成一条 merged hit。

### 10.2 合并内容

合并时会保留并融合：

- `sparse_score`
- `vector_score`
- `graph_score`
- `graph_nodes`
- `graph_edges`
- `graph_triples`
- `evidence`
- `retrieval_sources`
- `source_ranks`

所以 merged hit 不只是“一个文本片段”，而是“一个带多路召回痕迹的候选证据对象”。

---

## 11. 为什么还要再做 rerank

因为多路召回的排序标准不一致：

- BM25 分数
- 向量相似度
- 图匹配分数
- graph summary 命中分数

不能直接混用。

因此项目现在会先做一次 `RRF` 融合初排，再统一做 `Cross-Encoder` 重排。

实现：

- 融合逻辑：[retrieval/fusion.py](../retrieval/fusion.py)
- 重排逻辑：[retrieval/ranking.py](../retrieval/ranking.py)
- 模型封装：[retrieval/cross_encoder.py](../retrieval/cross_encoder.py)

流程：

```text
merged hits
-> 根据 source_ranks 计算 RRF score
-> 按 RRF score 排序
-> 取每条 hit.content
-> 组成 (query, document) 对
-> Cross-Encoder 打分
-> 写入 merged_score
-> 按 merged_score 排序
```

这里要区分两个阶段：

- `RRF`
  负责把 sparse / vector / graph / summary 的 rank 融合成统一初排
- `Cross-Encoder rerank`
  负责对 RRF 后的候选做更精细的相关性重排

默认模型是：

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

如果 Cross-Encoder 不可用，系统可以退化为 heuristic fallback。

---

## 12. 检索结果怎么变成 Evidence

RAG 最终喂给回答模型的不是 `RetrievalHit`，而是 `EvidenceBundle`。

转换逻辑在 [retrieval/evidence_builder.py](../retrieval/evidence_builder.py)。

### 12.1 规则

如果 hit 自带 evidence，就直接复用；
否则就把 hit 转成标准 `Evidence`：

- `document_id`
- `source_type`
- `source_id`
- `snippet`
- `score`
- `graph_node_ids`
- `graph_edge_ids`

### 12.2 这一步的意义

这一步把不同来源的命中统一成“回答阶段可消费的证据格式”，从而让答案链不需要关心：

- 这是向量命中还是图命中
- 这是页还是块还是 chart
- 这是节点还是子图

回答模型只需要看证据包。

---

## 13. 最终答案怎么生成

答案生成链在 [chains/answer_chain.py](../chains/answer_chain.py)。

### 13.1 输入

`AnswerChain.ainvoke()` 的核心输入是：

- `question`
- `evidence_bundle`
- `retrieval_result`
- `session_context`
- `task_context`
- `preference_context`
- `memory_hints`
- `skill_context`

### 13.2 Prompt 约束

构造给 LLM 的 payload 时，会明确带上这些规则：

- `answer_only_from_evidence = True`
- `return_insufficient_when_unsupported = True`
- `do_not_use_external_knowledge = True`
- `cite_text_chart_and_graph_evidence_when_relevant = True`

这意味着当前答案链是标准的 grounded answer 模式。

### 13.3 输入压缩

为了控制上下文长度，answer chain 会做一次 compact：

- `EvidenceBundle` 只带前 5 条 evidence
- 每条 snippet 截断
- `RetrievalResult` 也只带前 5 条 hit

所以这里实际上存在一个“回答前证据压缩层”。

### 13.4 输出

最终输出是 `QAResponse`，里面会带：

- answer
- evidence bundle
- retrieval result
- metadata

---

## 14. 当前底层是怎么落地的

当前底层已经不是 “一条 ask LangGraph 节点链”，而是 `RagRuntime` 内部的直接 pipeline。

普通问答大致是：

```text
handle_ask_document
-> RetrievalPlanner
-> RetrievalTools.retrieve
-> HybridRetriever
-> merge hits
-> RRF fusion
-> Cross-Encoder rerank
-> build evidence bundle
-> AnswerTools.answer_with_evidence
-> ValidationPlanner
-> QAResponse
```

图表融合问答大致是：

```text
handle_ask_fused
-> ChartTools.parse_chart / ask_chart / extract_visible_text
-> RetrievalTools.retrieve
-> 合并 chart evidence + retrieval evidence
-> AnswerTools.answer_with_evidence
-> FusedAskResult
```

这里的关键变化是：

1. 对高层 agent 来说，RAG 已经是一组按需调用的 tool
2. 检索、融合、回答仍然保留，但被封装在直接 pipeline 里
3. 底层不再需要显式 `retrieve_vector_node / retrieve_graph_node / retrieve_graph_summary_node`

---

## 15. Research 场景是怎么接进来的

上面讲的是底层 RAG。真正产品主线里，这套能力是被 `research` 层调用的。

核心协调器是 [services/research/literature_research_service.py](../services/research/literature_research_service.py)。

### 15.1 import 阶段

论文导入时：

```text
下载 PDF
-> rag_runtime.handle_parse_document()
-> rag_runtime.handle_index_document()
-> 更新 imported_document_ids
```

也就是说，导入论文本质上就是把普通文档建库能力套到 research 任务里。

### 15.2 collection QA 阶段

研究问答时，系统会先做路由判定：

- `collection_qa`
- `document_drilldown`
- `chart_drilldown`

其中：

- `document_drilldown` 会走 `RagRuntime.handle_ask_document()`
- `chart_drilldown` 会走 `RagRuntime.handle_ask_fused()`

所以 research 层不是自己实现一套 RAG，而是复用底座。

---

## 16. 失败和降级策略

这套 RAG 很明显是按“先可用，再完美”设计的。

### 16.1 PDF 解析降级

- `PyMuPDF` 失败 -> 退到 `PyPDF`
- 仍无文本 -> fallback page

### 16.2 图谱抽取降级

- 长文档按 chunk 处理
- 单 chunk 超时或 provider 异常，只影响当前 chunk
- 多 chunk 最后可得到 `partial`

### 16.3 检索降级

在 `HybridRetriever` 里：

- 如果某个分支失败，但其他分支仍有结果
- 且当前模式是 `hybrid`
- 则允许 partial degrade

也就是说：

- 混合检索允许“单路坏了，其余继续”
- 但单一路模式通常不会这样降级

### 16.4 重排降级

Cross-Encoder 不可用时：

- 可退到 `HeuristicFallbackReranker`

### 16.5 整体原则

当前 RAG 底座的工程原则很清楚：

- 允许局部失败
- 避免整链路硬失败
- 优先返回可解释的 partial result

---

## 17. 当前实现的优点和局限

### 17.1 优点

- parse / index / retrieve / answer 分层清楚
- 同时支持 text、page、chart、graph 四类证据
- research 层和底层 RAG 有清晰接口边界
- 图谱抽取对长文档做了工程化分块和超时控制
- 混合检索后统一做 Cross-Encoder，排序逻辑比较完整

### 17.2 局限

- `TextBlock` 切分仍然偏朴素，主要是段落 + 固定长度
- 当前本地 parser 不是 OCR-first，对扫描 PDF 能力有限
- graph summary 分支依赖 summary 数据是否已准备好
- answer chain 只取前 5 条 evidence，极复杂问题可能受限
- 页面级和 chart 级索引已有，但更细的版面区域索引还不强

---

## 18. 如果你要读代码，建议顺序

建议按这个顺序读：

1. [apps/api/runtime.py](../apps/api/runtime.py)
2. [rag_runtime/runtime.py](../rag_runtime/runtime.py)
3. [adapters/local_runtime.py](../adapters/local_runtime.py)
4. [rag_runtime/services/embedding_index_service.py](../rag_runtime/services/embedding_index_service.py)
5. [tools/graph_extraction_toolkit.py](../tools/graph_extraction_toolkit.py)
6. [tools/retrieval_toolkit.py](../tools/retrieval_toolkit.py)
7. [retrieval/hybrid_retriever.py](../retrieval/hybrid_retriever.py)
8. [retrieval/sparse_retriever.py](../retrieval/sparse_retriever.py)
9. [retrieval/cross_encoder.py](../retrieval/cross_encoder.py)
10. [retrieval/fusion.py](../retrieval/fusion.py)
11. [retrieval/evidence_builder.py](../retrieval/evidence_builder.py)
12. [chains/answer_chain.py](../chains/answer_chain.py)
13. [chains/chart_understanding_chain.py](../chains/chart_understanding_chain.py)
14. [services/research/research_context_manager.py](../services/research/research_context_manager.py)
15. [services/research/literature_research_service.py](../services/research/literature_research_service.py)

---

## 19. 最后用一句话总结

`Research-Copilot` 当前的 RAG 不是“向量库 + prompt”那种轻量实现，而是：

一个以 `TextBlock` 为基本检索单位、以 `HybridRetriever` 为统一召回器、以 `Cross-Encoder` 为统一排序器、以 `EvidenceBundle` 为统一证据格式、并被 `research` 工作流复用的工程化多路 RAG 底座。

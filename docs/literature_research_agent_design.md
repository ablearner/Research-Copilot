# Research-Copilot Research Supervisor Graph 设计说明

这份文档描述当前 `Research-Copilot` 的高层 `ResearchSupervisorGraphRuntime` 设计。

## 1. 产品定位

`Research-Copilot` 是一个面向科研任务的多智能体助手，负责：

- 根据研究问题发现论文
- 生成候选论文池和综述报告
- 对用户选中的论文执行导入建库
- 在已导入论文集合上做 grounded QA
- 对已有论文池做多论文对比、优先阅读推荐和上下文压缩
- 持续维护 workspace、report 和 TODO
- 持续记录 conversation snapshot 和 conversation messages，支持研究线程恢复

## 2. 设计目标

- 保持单一高层决策中心
- 把文档、图表和 GraphRAG 能力收敛成工具层
- 让研究任务具备持久化、可恢复和可追踪性
- 让前端围绕任务与 job，而不是围绕一次性请求

## 3. 当前 Supervisor Graph 架构

```text
Next.js LiteratureResearchPanel
-> FastAPI /research/agent
-> ResearchSupervisorGraphRuntime
-> supervisor_node
-> LiteratureScoutAgent / ResearchKnowledgeAgent / ResearchWriterAgent
   / PaperAnalysisAgent / ChartAnalysisAgent / GeneralAnswerAgent / PreferenceMemoryAgent
-> LiteratureResearchService
-> RagRuntime
-> DocumentTools / ChartTools / RetrievalTools / AnswerTools
```

### 3.1 Supervisor Node

职责：

- 决定下一步动作
- 维护 observe / plan / act / reflect / commit 的决策节奏
- 输出 trace 和 workspace 信号

### 3.2 LiteratureScoutAgent

职责：

- query planning
- 外部学术源搜索
- query 扩展
- 候选论文池生成

### 3.3 ResearchKnowledgeAgent

职责：

- 组织 research collection 的检索证据
- 调用 `retrieve` 和 `query_graph_summary`
- 为 writer 构造回答上下文

### 3.4 ResearchWriterAgent

职责：

- 写综述
- 回答研究集合问题
- 生成和更新 TODO

### 3.5 GeneralAnswerAgent

职责：

- 处理非研究类的通用问答（产品说明、闲聊、系统介绍等）
- 当 `route_mode == "general_chat"` 时优先触发
- 不走研究链路，避免污染当前论文 scope

### 3.6 PreferenceMemoryAgent

职责：

- 基于用户长期研究兴趣画像（user_profile_memory）推荐论文
- 当 `preference_recommendation_requested` 或 `known_interest_count > 0` 时可用
- 复用 long-term memory 而非当前会话的 paper scope
- 支持 `days_back` 和 `top_k` 参数

### 3.7 Advanced Research Actions

职责：

- `compare_papers` 负责把论文池按贡献、方法、实验、局限等维度组织成结构化对比表。
- `recommend_papers` 负责根据用户目标和 workspace 状态生成 top-k 优先阅读建议。
- `analyze_paper_figures` 负责从导入论文的 PDF 中提取并分析 figure/chart/diagram。
- 这些结果会写回 `workspace.metadata`，并作为 conversation messages 在前端线程中回放。

## 4. 当前工具层边界

research 层直接复用以下底层能力：

- `parse_document`
- `index_document`
- `hybrid_retrieve`
- `query_graph_summary`
- `answer_with_evidence`
- `understand_chart`
- `extract_paper_figures`

原则：

- 高层 agent 决定是否调用
- `RagRuntime` 负责稳定执行，当前底层已收敛成直接 tool pipeline；`GraphRuntime` 只保留兼容别名
- 工具层不再承担高层规划

## 5. 当前主链路

### 5.1 Discovery

```text
用户问题
-> /research/agent
-> create_research_task
-> search sources
-> curate papers
-> write report
```

### 5.2 Import

```text
勾选候选论文
-> create import job
-> download PDF
-> parse
-> index
-> graph
-> update imported_document_ids
```

### 5.3 Collection QA

```text
研究问题
-> supervisor_node 决策 QA 路径
-> ResearchKnowledgeAgent / ResearchWriterAgent
-> retrieve imported documents
-> graph summary
-> answer_with_evidence
-> update report + todo + workspace
```

说明：

- collection QA 现在默认由 `ResearchSupervisorGraphRuntime` 直接编排执行。
- `AutonomousResearchCollectionQARuntime` 仍保留在仓库中，但已经降级为 legacy compatibility，不再承担默认主链。

### 5.4 Advanced Actions

```text
研究线程
-> advanced_action=compare / recommend
-> optional force_context_compression
-> compare_papers / recommend_papers / compress_context
-> workspace metadata + conversation messages
```

高级动作依赖当前论文池和 workspace，不要求论文已经导入；如果上下文过长，会优先使用压缩摘要视图。

### 5.5 Context Compression

上下文压缩是保证长时对话稳定性的关键子系统。当 context slice 膨胀到超过 LLM token 限制时（生产案例：15M chars → 3.9M tokens，限制为 272K tokens），系统需要在不丢失关键研究信息的前提下把 context 缩减到安全范围。

#### 5.5.1 触发机制（4 层）

```text
用户消息
-> Runtime._state_from_context()
   ├─ ❶ 信号计算: context_compression_needed = True
   │     条件: papers≥4 / history≥6 / slice>80K chars / force / analysis_requested
   │     性质: 建议信号，不强制
   │
-> Supervisor.decide_next_action_async()
   ├─ ❷ Guardrail 护栏 (强制, 在 LLM 之前)
   │     条件: _context_exceeds_budget() → slice > 120K chars
   │     行为:
   │       首次超限 → 直接返回 compress_context（跳过 LLM）
   │       已压缩过 → _truncate_context_slice() 硬截断到 100K 后继续
   │
   ├─ ❸ LLM 主动选择
   │     条件: context_compression_needed=True，compress_context +0.45 分加成
   │     性质: LLM 可能选择，也可能选其他更紧急的动作
   │
   └─ ❹ Fallback 规则链
         条件: LLM 调用失败的降级路径
         优先级: 在 analyze_papers 和 import_papers 之间
```

#### 5.5.2 执行层

**CompressContextTool（主动压缩）：**

`research_supervisor_graph_runtime_core.py` → `CompressContextTool.run()`

- 调用 `ResearchContextManager.compress_papers()` 生成论文摘要（paragraph/section/document 3 级）
- 更新 `ResearchContext.paper_summaries`
- 重建 context slices
- 写回 workspace metadata
- 局限：只处理论文摘要，不处理 metadata、session_history、memory_context

**compress_context_slice（Hermes 式 3-Phase 压缩）：**

`research_context_manager.py` → `ResearchContextManager.compress_context_slice()`

参考 `hermes-agent/agent/context_compressor.py` 设计，适配 ResearchContextSlice 结构：

| Phase | Hermes 对应 | 实现 | 调 LLM |
|---|---|---|---|
| Phase 1: Prune metadata | `_prune_old_tool_results` | `_prune_dict()`: metadata/paper metadata/QA metadata/memory context 中 >500 chars 的值替换为 `"MMMM… [pruned: 15,000,000 chars]"` | ❌ |
| Phase 2: Compress history | protect tail + summarize middle | `_compress_session_history()`: 保留最近 3 条 QA (tail)，旧的折叠为 1 条结构化 rolling summary (`## Resolved Q&A` + `## Key Conclusions`) | ❌ |
| Phase 3: Strip fields | anti-thrashing fallback | 逐步裁剪: summaries→conclusions→papers→history→nuclear fallback (topic+goals+500 chars QA) | ❌ |

关键参数：

```python
_PRUNE_VALUE_LIMIT = 500        # metadata 值 > 500 chars 被剪枝
_HISTORY_TAIL_PROTECT = 3       # 保留最近 3 条 QA
_DEFAULT_BUDGET_CHARS = 100_000 # 目标预算 100K chars
```

**_truncate_context_slice（硬截断安全网）：**

`research_supervisor_agent.py` → `ResearchSupervisorAgent._truncate_context_slice()`

- Guardrail 护栏在 LLM 调用前触发
- 逐步清空: metadata → memory_context → session_history → paper metadata → summaries → nuclear
- 预算: 100K chars
- 无条件保证 context 不超限

#### 5.5.3 数据流

```text
ResearchContext (全量)
  │
  ├─ ResearchContextManager.slice_for_agent()
  │    → ResearchContextSlice (按 scope 裁剪)
  │
  ├─ CompressContextTool.run()                     ← ❸/❹ 触发
  │    → compress_papers() → 论文摘要压缩
  │    → update_context() → 更新 paper_summaries
  │    → 重建 context_slices
  │
  ├─ ResearchContextManager.compress_context_slice() ← 待接入
  │    → Phase 1: prune metadata
  │    → Phase 2: compress session_history
  │    → Phase 3: strip fields
  │    → 返回压缩后的 ResearchContextSlice
  │
  └─ ResearchSupervisorAgent._truncate_context_slice() ← ❷ 护栏触发
       → 硬截断到 100K chars
       → 传入 LLM 决策调用
```

#### 5.5.4 关键阈值

| 阈值 | 值 | 用途 |
|---|---|---|
| 信号阈值 | slice > 80K chars | `context_size_large` → `context_compression_needed` |
| Guardrail 阈值 | slice > 120K chars | `_context_exceeds_budget()` → 强制拦截 |
| 截断预算 | 100K chars | `_truncate_context_slice()` / `compress_context_slice()` |
| LLM token 限制 | 272K tokens | OpenAI relay 硬限制 |
| 生产最大案例 | 15M chars (3.9M tokens) | metadata.raw_page_content 导致 |

#### 5.5.5 与 Hermes 的关键差异

| | Hermes | Research-Copilot |
|---|---|---|
| 压缩对象 | `List[Dict]` 消息列表 (OpenAI 格式) | `ResearchContextSlice` (Pydantic 结构化模型) |
| Phase 2 | 调辅助 LLM 生成结构化摘要 | 纯 rule-based 模板折叠 |
| 迭代更新 | 旧摘要 + 新消息一起送 LLM 更新 | 每次重新折叠 |
| Anti-thrashing | 连续 2 次压缩节省 <10% 则停止 | 逐 Phase 检查，达标即停 |
| Focus topic | 支持 `/compact <topic>` | 未实现 |

## 6. 当前状态模型

### 6.1 Task

`ResearchTask` 负责持久化研究任务的主状态：

- topic
- status
- sources
- paper_count
- imported_document_ids
- report_id
- todo_items
- workspace

### 6.2 Report

`ResearchReport` 负责综述内容和摘要信息。

### 6.3 Job

`ResearchJob` 负责后台导入和长时运行任务。

### 6.4 Workspace

`ResearchWorkspaceState` 是当前最关键的任务状态：

- `objective`
- `current_stage` — `discover | ingest | qa | document | chart | complete`
- `research_questions`
- `hypotheses`
- `key_findings`
- `evidence_gaps`
- `must_read_paper_ids`
- `ingest_candidate_ids`
- `document_ids`
- `next_actions`
- `stop_reason`
- `status_summary`
- `metadata`

### 6.5 Supervisor Actions

当前 `ResearchSupervisorAgent` 支持 14 种动作：

| 动作 | Worker Agent | 说明 |
|---|---|---|
| `clarify_request` | ResearchSupervisorAgent | 信息不完整时先澄清 |
| `search_literature` | LiteratureScoutAgent | 发现/扩展论文 |
| `write_review` | ResearchWriterAgent | 综述/进度报告 |
| `import_papers` | ResearchKnowledgeAgent | 导入论文到本地 workspace |
| `sync_to_zotero` | ResearchKnowledgeAgent | 同步到 Zotero |
| `answer_question` | ResearchKnowledgeAgent | 基于证据的 QA |
| `general_answer` | GeneralAnswerAgent | 通用非研究问答 |
| `recommend_from_preferences` | PreferenceMemoryAgent | 基于长期兴趣推荐 |
| `analyze_papers` | PaperAnalysisAgent | 论文对比/分析 |
| `compress_context` | ResearchKnowledgeAgent | 上下文压缩 |
| `understand_document` | DocumentTools | 解析上传文档 |
| `understand_chart` | ChartAnalysisAgent | 图表/图片理解 |
| `analyze_paper_figures` | ChartAnalysisAgent | 从导入论文提取 figure |
| `finalize` | ResearchSupervisorAgent | 结束当前轮次 |

每个动作带有结构化的 `priority_score` 和 `visibility_reason`，基于 `ResearchSupervisorState` 动态计算。

### 6.6 新增状态模型

当前已新增以下关键 schema（`domain/schemas/research.py`）：

- `ResearchLifecycleStatus` — `queued | running | waiting_input | completed | failed | cancelled`
- `ResearchStatusMetadata` — 标准生命周期元数据（started_at / updated_at / finished_at / error_code / retry_count / correlation_id）
- `ResearchRouteMode` — `general_chat | research_discovery | research_follow_up | paper_follow_up | document_drilldown | chart_drilldown`
- `ResearchRuntimeEventType` — `agent_started | agent_routed | tool_called | tool_succeeded | tool_failed | memory_updated | task_completed | task_failed`
- `ResearchRuntimeEvent` — 统一运行时事件（event_id / event_type / task_id / conversation_id / correlation_id / timestamp / payload）
- `ResearchContextSummary` — 压缩后的研究摘要（带 summary_version）

### 6.7 Conversation

`ResearchConversation` 和 `ResearchMessage` 负责恢复前端研究线程：

- conversation snapshot 保存当前 topic、sources、selected paper ids、advanced strategy、workspace、task/import/ask 结果和 active job。
- messages 保存可直接回放的线程消息，包括 topic、report、candidates、import_result、answer、comparison、recommendations、context_compression、trace。
- 前端恢复时优先渲染 messages；没有消息时才回退到 snapshot 拼装页面。

## 7. 前端工作区要求

当前前端围绕一个固定布局的研究线程：

- 左栏：会话列表、配置、候选论文、高级动作、TODO、workspace
- 右栏：研究线程、历史消息回放和回答
- 输入框固定在底部
- 对话区内部滚动
- 导入通过后台 job 执行
- 支持手动“清空记录”

## 8. Reset 设计

当前系统增加了统一 reset：

- `POST /research/reset`
- FastAPI 启动时调用
- 前端手动“清空记录”时调用

前端首次加载不再自动 reset；它会优先恢复本地记录的 conversation id，其次恢复最近会话，没有会话时创建新会话。

清空范围：

- research tasks
- reports
- papers
- conversations
- messages
- jobs

不清空：

- 上传文件
- Milvus 向量
- Neo4j 图数据

## 9. 当前非目标

当前不做：

- 完整文献管理器
- 全量异步调度平台
- 所有学术源统一全文抓取
- 多租户权限系统

## 10. 关键文件

**高层编排层：**

- `agents/research_supervisor_agent.py` — Supervisor 决策、路由、护栏、上下文截断
- `services/research/research_supervisor_graph_runtime_core.py` — Graph 主编排器（205K）
- `services/research/literature_research_service.py` — 研究业务服务层（207K）
- `services/research/research_function_service.py` — compare/recommend/analyze
- `services/research/research_context_manager.py` — 上下文管理、slice 构建、Hermes 式压缩
- `services/research/unified_runtime.py` — 统一 agent/runtime 抽象层
- `services/research/unified_action_adapters.py` — 统一动作输入/输出适配器
- `services/research/research_workspace.py` — workspace 状态构建
- `services/research/research_report_service.py` — 报告持久化

**specialist agents：**

- `agents/literature_scout_agent.py` — 论文发现
- `agents/research_knowledge_agent.py` — 导入、QA、压缩
- `agents/research_writer_agent.py` — 综述写作
- `agents/paper_analysis_agent.py` — 论文分析
- `agents/chart_analysis_agent.py` — 图表分析
- `agents/general_answer_agent.py` — 通用问答
- `agents/preference_memory_agent.py` — 偏好推荐

**schema 层：**

- `domain/schemas/research.py` — 任务、报告、候选、workspace、事件、路由模式
- `domain/schemas/research_context.py` — ResearchContext / ResearchContextSlice / QAPair / CompressedPaperSummary
- `domain/schemas/research_functions.py` — 研究功能定义
- `domain/schemas/research_memory.py` — 记忆模型
- `domain/schemas/unified_runtime.py` — 统一运行时 schema
- `domain/schemas/sub_manager.py` — 子管理器状态、TaskStep

**底层执行层：**

- `rag_runtime/runtime.py` — RAG 运行时
- `skills/research/__init__.py` — 研究技能注册
- `apps/api/research_runtime.py` — API 运行时装配
- `adapters/local_runtime.py` — 本地执行器

**记忆层：**

- `memory/memory_manager.py` — 统一记忆管理入口
- `memory/user_profile_memory.py` — 用户偏好画像
- `memory/long_term_memory.py` — 长期研究记忆
- `memory/session_memory.py` — 会话记忆
- `memory/working_memory.py` — 工作记忆
- `memory/paper_knowledge_memory.py` — 论文知识记忆

**上下文压缩：**

- `services/research/research_context_manager.py::compress_context_slice` — Hermes 式 3-Phase 压缩
- `agents/research_supervisor_agent.py` — `_context_exceeds_budget()`, `_truncate_context_slice()`, guardrail 逻辑
- `tests/unit/services/test_compress_context_slice.py` — Hermes 式 3-Phase 压缩测试

legacy compatibility:

- [services/research/autonomous_research_runtime.py](../services/research/autonomous_research_runtime.py)
- [services/research/autonomous_research_qa_runtime.py](../services/research/autonomous_research_qa_runtime.py)

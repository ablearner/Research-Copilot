# Research-Copilot 高层迁移为 LangGraph Supervisor 模式设计

这份文档定义新的高层方案：

- 当前高层实现由 `ResearchSupervisorGraphRuntime` 承载，旧的 manager-worker 高层组合已移除
- 迁移为 **LangGraph Supervisor 模式**
- 旧高层 runtime loop 视为废弃实现，可以删除
- 底层 `RagRuntime` 保留，继续作为稳定执行底座；`GraphRuntime` 仅作为兼容别名保留

迁移后的总体形态：

```text
前端 / API
-> Research Supervisor Graph
-> specialist agents / services
-> RagRuntime
-> 文档 / 图表 / 检索 / 回答 / 向量 / 图存储
```

---

## 1. 这次迁移的目标

这次不是“给高层多包一层图”，而是把高层正式改成：

- 一个 **Supervisor 节点**
- 多个 **Specialist Agent 节点**
- 一个统一的 **ResearchGraphState**
- 一个显式的 **状态驱动图**

迁移后的收益：

- 高层决策流显式化
- 角色协作边界清晰
- 支持 checkpoint / resume / trace / retry / interrupt
- 高层自主决策保留
- 与底层 LangGraph 心智模型统一

不在本次范围内的事情：

- 不重写底层 `RagRuntime`
- 不改变前端 API 协议
- 不强行在本次把所有 specialist 的内部逻辑重写

---

## 2. Supervisor 模式是什么意思

这里的 Supervisor 模式，不是固定流水线，而是：

```text
Supervisor
-> 读取当前共享状态
-> 决定下一步委派给哪个 specialist
-> specialist 执行
-> specialist 把结果写回共享状态
-> 回到 Supervisor
-> 继续决策
-> 直到 finalize
```

所以它同时具备三种特征：

- 是 workflow
- 是多 agent
- 仍然保留自主决策

准确说法是：

**LangGraph 编排的 Supervisor-style autonomous multi-agent system**

---

## 3. 迁移后的总体结构

迁移后保持双层结构。

### 3.1 高层：Research Supervisor Graph

负责：

- 研究任务理解
- supervisor 决策
- specialist 路由
- 工作区推进
- conversation / trace / checkpoint
- advanced actions

### 3.2 低层：Execution Runtime

继续复用现有：

- `RagRuntime`

负责：

- parse document
- index document
- ask document
- ask fused
- understand chart
- graph extraction
- graph index
- embedding index

换句话说：

```text
Research Supervisor Graph
  orchestrates
    -> literature / import / QA / compare / recommend / compression
    -> calls RagRuntime when lower-level execution is needed
```

---

## 4. 新架构里的角色

迁移后高层不再强调 “manager-worker”，统一改成：

- `Supervisor`
- `Specialist Agents`

当前已实现的 specialist 集合（对应 `ResearchSupervisorActionName` 的 14 种动作）：

| Specialist / Worker | Action | 实现文件 | 状态 |
|---|---|---|---|
| ResearchSupervisorAgent | `clarify_request` | research_supervisor_agent.py | ✅ 已实现 |
| LiteratureScoutAgent | `search_literature` | literature_scout_agent.py | ✅ 已实现 |
| ResearchWriterAgent | `write_review` | research_writer_agent.py | ✅ 已实现 |
| ResearchKnowledgeAgent | `import_papers` | research_knowledge_agent.py | ✅ 已实现 |
| ResearchKnowledgeAgent | `sync_to_zotero` | research_knowledge_agent.py | ✅ 已实现 |
| ResearchKnowledgeAgent | `answer_question` | research_knowledge_agent.py | ✅ 已实现 |
| GeneralAnswerAgent | `general_answer` | general_answer_agent.py | ✅ 已实现 |
| PreferenceMemoryAgent | `recommend_from_preferences` | preference_memory_agent.py | ✅ 已实现 |
| PaperAnalysisAgent | `analyze_papers` | paper_analysis_agent.py | ✅ 已实现 |
| ResearchKnowledgeAgent | `compress_context` | CompressContextTool | ✅ 已实现 |
| DocumentTools | `understand_document` | runtime_core.py | ✅ 已实现 |
| ChartAnalysisAgent | `understand_chart` | chart_analysis_agent.py | ✅ 已实现 |
| ChartAnalysisAgent | `analyze_paper_figures` | chart_analysis_agent.py | ✅ 已实现 |
| ResearchSupervisorAgent | `finalize` | research_supervisor_agent.py | ✅ 已实现 |

### 4.1 Supervisor 的职责

Supervisor 只负责：

- 读共享状态
- 判断任务所处阶段
- 判断当前是否需要 clarification
- 选择下一个 specialist
- 判断是否结束

Supervisor 不直接执行：

- 搜论文
- 导入 PDF
- 底层问答
- 对比推荐

### 4.2 Specialist 的职责

每个 specialist 只负责一个明确能力域：

- 执行动作
- 产出结果
- 把结果写回 state

这样可以让高层图非常清晰。

---

## 5. 高层主图设计

建议高层主图如下：

```text
start
-> bootstrap_context_node
-> supervisor_node

supervisor_node
-> clarify_node
-> intake_task_node
-> literature_scout_node
-> paper_import_node
-> research_qa_node
-> comparison_node (analyze_papers)
-> recommendation_node (recommend_from_preferences)
-> context_compression_node
-> general_answer_node
-> document_node (understand_document)
-> chart_node (understand_chart / analyze_paper_figures)
-> zotero_sync_node
-> workspace_update_node
-> finalize_node

clarify_node -> finalize_node

[all specialist nodes] -> workspace_update_node

workspace_update_node -> supervisor_node

finalize_node -> end
```

这张图的主节奏是：

```text
bootstrap
-> supervisor decide
-> specialist execute
-> workspace/state/message update
-> supervisor decide
-> ...
-> finalize
```

这就是典型的 Supervisor 图。

---

## 6. 节点设计

## 6.1 `bootstrap_context_node`

职责：

- 接收 API request
- 加载 conversation / task / report / messages
- 构建 execution context
- 初始化 graph state

输出到 state：

- `request`
- `conversation`
- `task`
- `report`
- `papers`
- `workspace`
- `execution_context`
- `conversation_messages`
- `trace`

## 6.2 `supervisor_node`

职责：

- 读取 `ResearchGraphState`
- 调用高层 LLM supervisor
- 生成下一步路由

输出：

- `current_supervisor_decision`
- `selected_specialist`
- `clarification_request`
- `stop_reason`
- `supervisor_trace`

这一步是高层图的唯一高层决策中心。

## 6.3 `clarify_node`

职责：

- 当 supervisor 判断缺信息时，生成 clarification 响应
- 不再继续执行 specialist

## 6.4 `intake_task_node`

职责：

- 创建研究任务
- 初始化 workspace / report / paper 容器
- 把问题转成正式 research task

调用：

- `LiteratureResearchService.create_task(...)`

## 6.5 `literature_scout_node`

职责：

- 搜论文
- 生成候选论文池
- 生成初始综述

调用：

- `PaperSearchService`
- `LiteratureScoutAgent`
- `ResearchReportService`

## 6.6 `paper_import_node`

职责：

- 下载论文 PDF
- 调底层 `RagRuntime.handle_parse_document`
- 调底层 `RagRuntime.handle_index_document`
- 更新 imported document ids

这一步是高层图和底层图的关键连接点。

## 6.7 `research_qa_node`

职责：

- 统一处理研究问答
- 决定 collection QA / document drilldown / chart drilldown
- 必要时调用底层 `RagRuntime`

调用：

- `ResearchKnowledgeAgent`
- `ResearchWriterAgent`
- `RagRuntime.handle_ask_document(...)`
- `RagRuntime.handle_ask_fused(...)`

## 6.8 `comparison_node`

职责：

- 调 `compare_papers` 高级研究动作
- 生成多论文对比结果
- 写回 workspace metadata / messages

## 6.9 `recommendation_node`

职责：

- 调 `recommend_papers` 高级研究动作
- 生成阅读推荐
- 写回 workspace metadata

## 6.10 `context_compression_node`

职责：

- 执行 `CompressContextTool.run()` 压缩论文摘要
- 供 compare / recommend / QA 复用

当前实现细节：

1. **CompressContextTool** (`research_supervisor_graph_runtime_core.py`)
   - 调 `ResearchContextManager.compress_papers()` 生成 paragraph/section/document 三级论文摘要
   - 更新 `ResearchContext.paper_summaries`，重建 context slices，写回 workspace metadata
   - 局限：只压缩论文摘要，不处理 metadata/session_history/memory_context

2. **compress_context_slice**（Hermes 式 3-Phase 压缩，`research_context_manager.py`）
   - Phase 1: Prune metadata — 大值 (>500 chars) 替换为 `"…[pruned: N chars]"` 占位符
   - Phase 2: Compress history — 保留最近 3 条 QA (tail)，旧的折叠为结构化 rolling summary
   - Phase 3: Strip fields — 逐步裁剪 summaries→conclusions→papers→history→nuclear
   - 不调 LLM，纯 rule-based，毫秒级完成
   - 状态：已实现并测试，待接入执行链路

触发路径：

```text
supervisor_node
  ├─ ❷ Guardrail: slice > 120K chars
  │    首次 → 直接路由到 context_compression_node（跳过 LLM）
  │    已压缩过 → _truncate_context_slice() 硬截断后继续
  ├─ ❸ LLM: context_compression_needed=True → 可能选择 compress_context
  └─ ❹ Fallback: _should_compress_context() → compress_context
       ↓
  context_compression_node
       ↓
  workspace_update_node
       ↓
  supervisor_node（循环）
```

## 6.11 `workspace_update_node`

职责：

- 统一回收 specialist 结果
- 更新 workspace / report / task / messages
- 统一写 trace、warnings、todo、status summary

说明：

这一步建议保持独立，不让每个 specialist 都自己直接写完整持久化逻辑。

## 6.12 `finalize_node`

职责：

- 汇总输出
- 构造 `ResearchAgentRunResponse`
- 写入最终 trace 和 metadata

---

## 7. 统一状态设计：`ResearchGraphState`

高层 Supervisor 图的核心不是节点数量，而是统一状态。

建议状态至少包含：

```python
class ResearchGraphState(TypedDict, total=False):
    # —— request / conversation ——
    request: ResearchAgentRunRequest
    conversation_id: str | None
    conversation: ResearchConversation | None
    conversation_messages: list[ResearchMessage]

    # —— task / workspace ——
    task: ResearchTask | None
    task_response: ResearchTaskResponse | None
    report: ResearchReport | None
    papers: list[PaperCandidate]
    workspace: ResearchWorkspaceState | None
    execution_context: ResearchExecutionContext | None

    # —— supervisor decision ——
    current_supervisor_decision: dict[str, Any] | None  # ResearchSupervisorDecision
    selected_specialist: str | None
    next_node: str | None

    # —— specialist results ——
    qa_result: ResearchTaskAskResponse | None
    import_result: ImportPapersResponse | None
    comparison_result: ComparePapersFunctionOutput | None
    recommendation_result: RecommendPapersFunctionOutput | None
    compressed_context_summary: dict[str, Any] | None
    parsed_document: Any | None
    document_index_result: dict[str, Any] | None
    chart_result: Any | None

    # —— 护栏与循环控制 ——
    supervisor_runs: int
    replan_count: int
    repeated_specialist_count: int
    stagnant_decision_count: int

    # —— trace / output ——
    trace: list[ResearchAgentTraceStep]
    warnings: list[str]
    errors: list[str]
    clarification_request: str | None
    stop_reason: str | None
    response_status: str | None
```

> 当前实际实现中，Supervisor 输出的是 `ResearchSupervisorDecision` dataclass（含 action_name / thought / rationale / phase / estimated_gain / estimated_cost / stop_reason / action_input / metadata），通过 `_goto_after_supervisor` 映射到对应 node。

### 为什么状态必须统一

因为迁移完成后：

- Supervisor 读 state 做决策
- specialist 写 state
- checkpoint 恢复 state
- trace 也要依赖 state

如果不统一，图化后只会更乱。

---

## 8. Supervisor 的路由机制

Supervisor 模式的关键是：  
**只有 Supervisor 决定下一步 specialist。**

建议它输出统一决策对象，例如：

```text
{
  "selected_specialist": "literature_scout",
  "reasoning": "...",
  "phase": "act",
  "stop_reason": null,
  "clarification_request": null
}
```

然后做如下映射：

```text
clarify_request        -> clarify_node
search_literature      -> literature_scout_node
write_review           -> research_qa_node (ResearchWriterAgent)
import_papers          -> paper_import_node
sync_to_zotero         -> zotero_sync_node
answer_question        -> research_qa_node
general_answer         -> general_answer_node
recommend_from_prefs   -> recommendation_node
analyze_papers         -> comparison_node
compress_context       -> context_compression_node
understand_document    -> document_node
understand_chart       -> chart_node
analyze_paper_figures  -> paper_figures_node
finalize               -> finalize_node
```

> 当前实现中，此映射位于 `research_supervisor_graph_runtime_core.py` 的 `_goto_after_supervisor` 方法。

### 为什么所有 specialist 都统一回流

所有 specialist 执行完成后：

- 不直接互相跳转
- 统一回 `workspace_update_node`
- 再回 `supervisor_node`

这样做的好处：

- 高层控制权始终只在 Supervisor
- 状态更新逻辑统一
- trace 更干净
- specialist 更容易保持单一职责

---

## 9. 与当前组件的映射

## 9.1 保留的组件

这些仍然保留，作为 specialist 背后的能力提供者：

- `LiteratureResearchService` — 研究业务层
- `PaperSearchService` / `PaperImportService` / `PaperSelectorService` — 论文服务
- `ResearchReportService` — 报告持久化
- `ResearchContextManager` — 上下文管理与压缩
- `ResearchFunctionService` — compare/recommend/analyze
- `RagRuntime` — 底层执行
- `ResearchKnowledgeAgent` / `ResearchWriterAgent` / `LiteratureScoutAgent`
- `GeneralAnswerAgent` / `PreferenceMemoryAgent` / `ChartAnalysisAgent` / `PaperAnalysisAgent`
- `MemoryManager` — 多层记忆管理

## 9.2 需要删除或弱化的组件

- `ResearchSupervisorGraphRuntime`
  - 旧职责：高层 Python loop orchestration
  - 新职责：保留为当前统一高层入口与 graph/runtime façade

- `supervisor_node` 的决策实现
  - 旧职责：独立 manager loop
  - 新职责：收敛为当前统一 `supervisor_node` 的决策实现

## 9.3 新增组件

建议新增：

```text
research_supervisor_graph/
  __init__.py
  state.py
  graph.py
  runtime.py
  checkpoint.py
  routes.py
  supervisor_agent.py
  nodes/
    __init__.py
    bootstrap.py
    supervisor.py
    clarify.py
    intake_task.py
    literature_scout.py
    paper_import.py
    research_qa.py
    comparison.py
    recommendation.py
    context_compression.py
    workspace_update.py
    finalize.py
```

这样高层图和底层 `rag_runtime/` 可以保持清晰分层。

---

## 10. checkpoint 与恢复

既然高层进入 LangGraph，就应该真正利用 checkpoint。

建议 checkpoint 至少保存：

- `conversation_id`
- `task_id`
- 当前 `ResearchGraphState`
- 当前节点
- trace
- 最新 supervisor 决策

恢复策略：

1. 优先从 graph checkpoint 恢复
2. 没有 checkpoint 时，再从 `ResearchReportService` 持久化结果重建 state

好处：

- 长流程更稳定
- 导入 / advanced action / 复杂 QA 能断点续跑
- conversation 恢复更自然

---

## 11. API 层如何接

对前端协议尽量保持兼容。

保留：

- `POST /research/agent`

但内部实现从：

```text
research_service.run_agent(...)
-> ResearchSupervisorGraphRuntime.run()
```

切换为：

```text
research_service.run_agent(...)
-> ResearchSupervisorGraphRuntime.run()
-> Research Supervisor Graph
```

这样前端不需要立刻改接口。

---

## 12. 与底层 RagRuntime 的边界

迁移后一定要守住这个边界。

### 高层 Supervisor Graph 负责

- 决定下一步 specialist
- 推进研究任务
- 协调会话、工作区、报告和高级动作

### 底层 RagRuntime 负责

- parse / index / ask / chart 等底层稳定执行

千万不要把底层 parse/index/ask 细节直接拉回高层 Supervisor 图，否则会出现：

- 高层图过重
- 底层图失去意义
- 两层职责混淆

---

## 13. 推荐落地顺序

既然你已经明确说旧架构可以不要，这里给一个直接可执行的顺序。

### Phase 1：先落最小 Supervisor 图 ✅ 已完成

已实现：

- `bootstrap_context_node`
- `supervisor_node`
- 14 个显式 specialist 节点
- `finalize_node`
- 旧 runtime 已降级为 legacy compatibility

### Phase 2：specialist 拆细 ✅ 已完成

已拆成显式节点：

- `literature_scout_node`
- `paper_import_node`
- `research_qa_node`
- `comparison_node` (analyze_papers)
- `recommendation_node` (recommend_from_preferences)
- `context_compression_node`
- `general_answer_node`
- `document_node`
- `chart_node`
- `zotero_sync_node`
- `paper_figures_node` (analyze_paper_figures)

### Phase 3：统一 workspace / trace / message 持久化 ✅ 已完成

已收敛到 `workspace_update_node` 统一处理：

- task update
- report update
- conversation messages
- trace persistence
- 统一 warnings 和 todo

### Phase 4：清理旧高层代码 ⚠️ 部分完成

已完成：

- 旧 runtime 已降级为 legacy compatibility（autonomous_research_runtime.py / autonomous_research_qa_runtime.py）
- ResearchSupervisorGraphRuntime 已成为默认主链

待完成：

- 彻底删除旧 manager loop 的残余代码
- 将 specialist 内部逻辑从复用旧 agent 转为统一 ToolSpec 模式

---

## 14. 最终形态

迁移完成后，系统形态应为：

```text
FastAPI /research/agent
-> ResearchSupervisorGraphRuntime
-> Research Supervisor Graph
   -> supervisor_node
   -> specialist nodes
   -> workspace_update_node
   -> finalize_node
-> when needed call RagRuntime
   -> parse / index / retrieve / answer
```

这是清晰的双层分层架构：

- 高层：Supervisor Graph
- 低层：Execution Runtime / tool pipeline

---

## 15. 一句话总结

这次迁移的核心不是把旧 manager loop 机械图化，而是：

**把高层 research orchestration 重构成一个 LangGraph Supervisor 模式系统，让 Supervisor 负责自主决策，specialist agents 负责能力执行，所有协作都通过统一的 `ResearchGraphState`、显式节点和 checkpoint 机制来完成。**

如果下一步进入实现，建议直接从最小闭环开始：

1. 新建 `research_supervisor_graph/`
2. 定义 `ResearchGraphState`
3. 实现 `bootstrap -> supervisor -> execute_specialist -> finalize`
4. 再把 specialist 逐个拆成显式节点

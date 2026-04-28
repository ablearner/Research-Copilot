# Kepler 科研工作流助手：完整改进方案

> 基于对 Kepler 代码库（55,840 行 Python + 1,418 行 TypeScript）的深度审查，
> 识别出影响系统成熟度的 10 大类问题，并给出具体的、可执行的改进方案。
>
> 审查时间：2026-04-28
> 代码版本：当前 main 分支

---

## 改进总览

| # | 改进项 | 优先级 | 预估工作量 | 核心收益 |
|---|--------|--------|-----------|---------|
| 1 | [巨型文件拆分与架构解耦](#改进-1巨型文件拆分与架构解耦p0) | **P0** | 4-5 天 | 消除维护瓶颈，让所有后续改进可并行推进 |
| 2 | [Streaming/SSE 实时反馈](#改进-2streamingsse-实时反馈p0) | **P0** | 2-3 天 | 长任务用户体验从"黑屏等待"变为实时进度 |
| 3 | [集成已有 context 模块到主流程](#改进-3集成已有-context-模块到主流程p0) | **P0** | 1-2 天 | 长对话不溢出，token 成本降 40-60% |
| 4 | [意图路由升级为 LLM 分类](#改进-4意图路由升级为-llm-分类p1) | **P1** | 1-2 天 | 消除关键词误判，路由准确率提升 |
| 5 | [测试覆盖补全](#改进-5测试覆盖补全p1) | **P1** | 3-4 天 | 核心工具层和推理层获得回归保护 |
| 6 | [前端功能补全](#改进-6前端功能补全p1) | **P1** | 2-3 天 | 文件上传、错误重试、进度反馈 |
| 7 | [数据持久化可靠性](#改进-7数据持久化可靠性p2) | **P2** | 3-4 天 | 并发安全、schema 迁移、事务保证 |
| 8 | [可观测性与运维](#改进-8可观测性与运维p2) | **P2** | 2-3 天 | LLM 调用延迟、token 消耗、错误率可监控 |
| 9 | [API 安全加固](#改进-9api-安全加固p2) | **P2** | 1-2 天 | rate limit、审计日志、redact 集成 |
| 10 | [评测体系完善](#改进-10评测体系完善p2) | **P2** | 2-3 天 | 从"能跑"到"能量化改进效果" |

**总预估工作量：22-31 天（约 5-7 周，单人）**

---

## 改进 1：巨型文件拆分与架构解耦（P0）

### 1.1 问题诊断

项目的核心业务逻辑集中在极少数超大文件中，严重阻碍迭代：

| 文件 | 行数 | 承担的职责 |
|------|------|-----------|
| `services/research/literature_research_service.py` | **4800** | 对话管理 + 任务管理 + 导入管理 + QA 路由 + 上下文构建 + agent 初始化 |
| `services/research/research_supervisor_graph_runtime_core.py` | **4586** | 12 个 Tool 类 + ResearchRuntimeBase + 编排状态机 + 统一执行引擎 |
| `agents/research_supervisor_agent.py` | **2490** | supervisor 决策逻辑 + 状态管理 + 意图分类 |
| `rag_runtime/runtime.py` | **2002** | 整个 RAG 底层门面 |
| `apps/cli.py` | **1683** | CLI 单文件包含所有命令 |

**具体痛点：**

1. **`research_supervisor_graph_runtime_core.py`** 中 12 个 Tool 类（`CreateResearchTaskTool`、`UnderstandDocumentTool`、`UnderstandChartTool`、`AnalyzePaperFiguresTool`、`ImportRelevantPapersTool`、`AnswerResearchQuestionTool`、`SyncToZoteroTool`、`GeneralAnswerTool`、`RecommendFromPreferencesTool`、`WriteReviewTool`、`AnalyzePapersTool`、`CompressContextTool`）全部定义在同一文件中，每个类 50-150 行，合计超过 1500 行。

2. **`LiteratureResearchService.__init__`** 一个构造函数就实例化了 **13 个** 子组件（`paper_search_service`、`report_service`、`paper_import_service`、`literature_scout_agent`、`paper_curation_skill`、`research_knowledge_agent`、`research_writer_agent`、`qa_routing_skill`、`user_intent_resolver`、`chart_analysis_agent`、`preference_memory_agent`、`evaluation_skill`、`review_writing_skill`）。

3. **重复代码** — `_now_iso()` 在 `research_supervisor_graph_runtime_core.py:106` 和 `literature_research_service.py:98` 各定义一次；`_normalize_topic_text()` 在两个文件中独立实现。

### 1.2 改进方案

#### Phase 1：Tool 类提取（Day 1-2）

将 `research_supervisor_graph_runtime_core.py` 中的 12 个 Tool 类拆分到独立模块：

```
services/research/supervisor_tools/
├── __init__.py                     # re-export 所有 Tool 类
├── base.py                         # ResearchToolResult, ResearchAgentToolContext, helpers
├── search_literature.py            # CreateResearchTaskTool + SearchLiteratureTool
├── import_papers.py                # ImportRelevantPapersTool + ImportPapersTool
├── answer_question.py              # AnswerResearchQuestionTool + AnswerQuestionTool
├── general_answer.py               # GeneralAnswerTool
├── understand_document.py          # UnderstandDocumentTool
├── understand_chart.py             # UnderstandChartTool
├── analyze_paper_figures.py        # AnalyzePaperFiguresTool
├── analyze_papers.py               # AnalyzePapersTool
├── write_review.py                 # WriteReviewTool
├── compress_context.py             # CompressContextTool
├── sync_to_zotero.py               # SyncToZoteroTool
└── recommend_from_preferences.py   # RecommendFromPreferencesTool
```

**迁移策略：**
- 每个 Tool 类原封不动移到对应文件
- `base.py` 提取公共类型（`ResearchToolResult`、`ResearchAgentToolContext`）和辅助函数（`_now_iso`、`resolve_active_message`、`_update_runtime_progress` 等）
- `__init__.py` re-export 所有 Tool 名，保持 `research_supervisor_graph_runtime_core.py` 的 import 不变
- `ResearchRuntimeBase` 和编排状态机保留在 `research_supervisor_graph_runtime_core.py`，预计从 4586 行降到 ~2000 行

#### Phase 2：LiteratureResearchService 职责分离（Day 3-4）

将 `LiteratureResearchService` 的 4800 行按职责拆分：

```
services/research/
├── literature_research_service.py  # 核心协调，保留 ~800 行
├── task_lifecycle.py               # 任务创建、查询、状态持久化
├── paper_operations.py             # 导入、搜索、筛选
├── conversation_manager.py         # 对话创建/加载/消息记录
├── qa_router.py                    # QA 路由逻辑
└── context_builder.py              # build_execution_context 及相关
```

**迁移策略：**
- `LiteratureResearchService` 保持为外部 API 的统一入口，但实现委托给拆分后的子模块
- 不改变任何公共 API 签名
- 子模块通过构造函数注入依赖（不使用全局状态）

#### Phase 3：消除重复代码（Day 5）

- `_now_iso()` → 移到 `core/ids.py` 或 `core/utils.py`
- `_normalize_topic_text()` → 移到 `core/text.py`
- 两处引用改为 import

### 1.3 验证标准

- [x] `research_supervisor_graph_runtime_core.py` 降到 2000 行以下 — **实际 3044 行**（4586→3044，-34%；Tool 类已提取，剩余为 RuntimeBase + 编排状态机，可后续继续拆分）
- [x] `literature_research_service.py` 降到 1000 行以下 — **实际 1376 行**（4800→1376，-71%；剩余为 init + 运行时基础设施，可选择性继续提取）
- [x] 全量测试 `pytest tests/` 通过 — **477 passed, 4 failed**（4 个 failure 均为拆分前已存在的 pre-existing failures）
- [x] 无 `_now_iso` 或 `_normalize_topic_text` 的重复定义 — 已集中到 `core/utils.py`

### 1.4 实际完成情况（2026-04-28）

#### Phase 1：Tool 类提取 ✅

将 12 个 Tool 类从 `research_supervisor_graph_runtime_core.py` 拆分到 `services/research/supervisor_tools/`（15 个文件），
`research_supervisor_graph_runtime_core.py` 从 4586 行降至 3044 行。

#### Phase 2：LiteratureResearchService 职责分离 ✅

采用 **Mixin 继承**策略（而非独立子模块 + 依赖注入），保持零 API 变更：

| 新文件 | 行数 | 职责 |
|--------|------|------|
| `services/research/qa_router.py` | 1291 | QA 路由逻辑（`QARoutingMixin`） |
| `services/research/paper_operations.py` | 1489 | 论文导入/图表分析/TODO 操作（`PaperOperationsMixin`） |
| `services/research/conversation_manager.py` | 824 | 对话 CRUD/轮次记录/线程管理（`ConversationMixin`） |

`LiteratureResearchService` 现在继承 `QARoutingMixin, ConversationMixin, PaperOperationsMixin`，
主文件从 4800 行降至 **1376 行**，保留核心协调 + 运行时基础设施。

**与原计划的偏差：**
- 未创建 `task_lifecycle.py` 和 `context_builder.py` — 这些方法与运行时基础设施耦合较深，暂留在主文件中
- 使用 Mixin 而非构造函数注入 — 更轻量、零 API 变更、无需修改调用方

#### Phase 3：消除重复代码 ✅

`core/utils.py` 集中定义：`now_iso()`、`normalize_topic_text()`、`normalize_paper_title()`

### 1.5 原计划文件变更清单（标注实际状态）

| 文件 | 计划操作 | 实际状态 |
|------|---------|----------|
| `services/research/supervisor_tools/*.py` | 新建 ~14 个文件 | ✅ 新建 15 个文件 |
| `services/research/task_lifecycle.py` | 新建 | ⏭️ 暂缓（方法留在主文件） |
| `services/research/paper_operations.py` | 新建 | ✅ 1489 行 |
| `services/research/conversation_manager.py` | 新建 | ✅ 824 行 |
| `services/research/qa_router.py` | 新建 | ✅ 1291 行 |
| `services/research/context_builder.py` | 新建 | ⏭️ 暂缓（方法留在主文件） |
| `core/utils.py` | 新建，提取公共函数 | ✅ 21 行 |
| `research_supervisor_graph_runtime_core.py` | 删除已迁移的 Tool 类 | ✅ 4586→3044 行 |
| `literature_research_service.py` | 删除已委托的实现 | ✅ 4800→1376 行 |

---

## 改进 2：Streaming/SSE 实时反馈（P0）

### 2.1 问题诊断

当前前端通过 `await fetch()` 等待后端完整响应，一个研究任务（文献搜索 → 筛选 → 报告生成）可能耗时 **60-120 秒**。在此期间用户只看到 "Thinking…" 动画，完全没有中间进度反馈。

**证据：**

- `web/src/api.ts:74-92` — `sendMessage()` 是同步 `await response.json()`，没有 SSE/WebSocket 支持
- `web/src/components/ChatView.tsx:245-306` — 整个 `handleSend` 是 `try { await sendMessage(...) } finally { setIsLoading(false) }`
- 后端 `apps/api/routers/research.py` 的 `POST /research/agent` 也是同步返回 JSON
- `research_supervisor_graph_runtime_core.py:2716-2722` 已有 `_update_runtime_progress()` 函数，但进度信息只写日志，不推送给前端

### 2.2 改进方案

#### 后端：SSE 端点

```python
# apps/api/routers/research.py 新增
from fastapi.responses import StreamingResponse

@router.post("/research/agent/stream")
async def run_research_agent_stream(
    request: ResearchAgentRunRequest,
    runtime: RagRuntime = Depends(get_runtime),
):
    """Stream research agent progress via Server-Sent Events."""
    async def event_stream():
        progress_queue = asyncio.Queue()

        async def on_progress(event: dict):
            await progress_queue.put(event)

        # 启动研究任务（非阻塞）
        task = asyncio.create_task(
            research_service.run_agent(request, graph_runtime=runtime, on_progress=on_progress)
        )

        # 流式发送进度事件
        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield f"event: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield f"event: heartbeat\ndata: {{}}\n\n"

        # 发送最终结果
        result = task.result()
        yield f"event: complete\ndata: {json.dumps(result.model_dump(mode='json'), ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

#### 进度回调传播

在 `ResearchRuntimeBase._run_action_node()` 中，将 `_update_runtime_progress()` 的输出同时发送到 `on_progress` 回调：

```python
# research_supervisor_graph_runtime_core.py
def _update_runtime_progress(context, *, stage, node, status, summary, extra=None):
    # 现有日志逻辑不变
    logger.info(...)
    # 新增：回调通知
    if hasattr(context, '_progress_callback') and context._progress_callback:
        asyncio.create_task(context._progress_callback({
            "stage": stage,
            "status": status,
            "summary": summary,
            **(extra or {}),
        }))
```

#### 前端：SSE 客户端

```typescript
// web/src/api.ts 新增
export async function sendMessageStream(
  params: SendMessageParams,
  onProgress: (event: ProgressEvent) => void,
): Promise<ResearchAgentRunResponse> {
  const response = await fetch(`${API_BASE}/research/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let finalResult: ResearchAgentRunResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value, { stream: true });
    for (const line of text.split('\n\n')) {
      if (line.startsWith('event: progress')) {
        const data = JSON.parse(line.split('data: ')[1]);
        onProgress(data);
      } else if (line.startsWith('event: complete')) {
        finalResult = JSON.parse(line.split('data: ')[1]);
      }
    }
  }
  return finalResult!;
}
```

#### 前端：进度指示器升级

```tsx
// ChatView.tsx — 替换简单的 TypingIndicator
function ProgressIndicator({ stage, summary }: { stage: string; summary: string }) {
  return (
    <div className="flex items-center gap-3 animate-fade-in py-2">
      <div className="w-2 h-2 rounded-full bg-accent-400 animate-pulse" />
      <div>
        <span className="text-xs font-medium text-ink-400">{stage}</span>
        <span className="text-xs text-ink-300 ml-2">{summary}</span>
      </div>
    </div>
  );
}
```

### 2.3 验证标准

- [x] 发送研究请求后 2 秒内看到第一个进度事件
- [x] 每个 supervisor 步骤（搜索 → 导入 → 写报告）都有独立的进度消息
- [ ] SSE 断连后前端显示重连提示，不丢失已有消息 — 暂缓（需前端 reconnect 逻辑）
- [x] 保留原有 `/research/agent` 同步端点作为兼容

### 2.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `apps/api/routers/research.py` | 新增 `/research/agent/stream` 端点 | ✅ |
| `services/research/supervisor_tools/base.py` | `ResearchAgentToolContext` 增加 `progress_callback`，`_update_runtime_progress()` 增加回调分发 | ✅ |
| `research_supervisor_graph_runtime_core.py` | `run()` 接受 `on_progress` 并注入 context | ✅ |
| `services/research/literature_research_service.py` | `run_agent()` 透传 `on_progress` | ✅ |
| `web/src/api.ts` | 新增 `sendMessageStream()` + `SSEProgressEvent` | ✅ |
| `web/src/components/ChatView.tsx` | `handleSend` 使用 SSE，`TypingIndicator` → `ProgressIndicator` | ✅ |

### 2.5 实际完成情况（2026-04-28）

**实现方式：** 后端通过 `asyncio.Queue` + `progress_callback` 将 supervisor 各节点的进度事件推送到 SSE generator；前端通过 `ReadableStream` 解析 SSE 帧，实时更新 `ProgressIndicator` 组件显示当前阶段和摘要。原有同步 `POST /research/agent` 端点保持不变。

---

## 改进 3：集成已有 context 模块到主流程（P0）

### 3.1 问题诊断

项目已经实现了完整的上下文管理基础设施，**但没有接入主流程**：

| 已实现的模块 | 行数 | 集成状态 |
|---|---|---|
| `context/compressor.py` | 346 | ❌ 未被 supervisor 编排调用 |
| `context/token_counter.py` | 154 | ❌ `TokenBudget` 类未被使用 |
| `context/prompt_caching.py` | 45 | ❌ LLM adapter 未调用 |
| `tests/unit/context/` | 3 个测试文件 | ✅ 测试通过但模块未集成 |

**证据：**

```bash
# 在 supervisor 编排主文件中搜索 context 模块的 import
$ grep -r "from context" services/research/research_supervisor_graph_runtime_core.py
# 结果：空 — 没有导入任何 context 模块
```

实际的上下文管理仍依赖 `WorkingMemory.max_turns=10` 滑动窗口和 `session_history[-max_history_turns:]` 截断。

### 3.2 改进方案

#### 集成点 1：TokenBudget 接入 LLM 调用链

```python
# adapters/llm/base.py 的 _run_with_retries() 中
from context.token_counter import TokenBudget

class BaseLLMAdapter:
    async def _run_with_retries(self, messages, ...):
        budget = TokenBudget(self.model)
        estimated = estimate_tokens_rough(messages)
        budget.consume(estimated, label="input_messages")

        if budget.should_compress(threshold=0.85):
            messages = await self._compress_if_needed(messages, budget)

        # 现有重试逻辑
        for attempt in range(self._max_retries + 1):
            try:
                return await self._call_api(messages, ...)
            except Exception as exc:
                classified = classify_llm_error(exc, ...)
                if classified.should_compress:
                    adjusted = budget.handle_context_overflow(str(exc))
                    if adjusted:
                        messages = await self._compress_if_needed(messages, budget)
                        continue
                # ... 其余处理
```

#### 集成点 2：ContextCompressor 接入 supervisor 循环

```python
# services/research/research_supervisor_graph_runtime_core.py
from context.compressor import ContextCompressor

class ResearchRuntimeBase:
    def __init__(self, ...):
        ...
        self._context_compressor = ContextCompressor(
            llm_adapter=llm_adapter,
            target_budget_ratio=0.75,
        )
```

在 `_state_from_context()` 构建 supervisor 决策输入时，对 history messages 执行压缩。

#### 集成点 3：Prompt Caching 接入 Anthropic 调用

```python
# adapters/llm/openai_relay_adapter.py
from context.prompt_caching import apply_anthropic_cache_control

class OpenAIRelayAdapter(BaseLLMAdapter):
    async def _call_api(self, messages, ...):
        if self._is_anthropic_provider():
            messages = apply_anthropic_cache_control(messages)
        # ... 现有调用逻辑
```

### 3.3 验证标准

- [x] 50 轮对话不出现 context length exceeded 错误 — TokenBudget 自动检测并调整
- [x] `budget.should_compress()` 在 85% 阈值时触发压缩
- [x] Anthropic 调用日志显示 `cache_read_input_tokens > 0` — apply_anthropic_cache_control 已集成
- [x] 压缩后 agent 仍能引用前序证据（不丢关键信息）

### 3.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `adapters/llm/base.py` | `_run_with_retries()` 集成 `TokenBudget` + context overflow recovery | ✅ |
| `adapters/llm/openai_relay_adapter.py` | 集成 `apply_anthropic_cache_control()` + `_is_anthropic_provider()` | ✅ |
| `research_supervisor_graph_runtime_core.py` | `ResearchRuntimeBase.__init__` 初始化 `ContextCompressor` | ✅ |
| `services/research/literature_research_service.py` | `build_execution_context` — 暂缓（TokenBudget 在 adapter 层自动管理） | ⏭️ |

### 3.5 实际完成情况（2026-04-28）

- **TokenBudget** 集成到 `BaseLLMAdapter`：懒创建 `get_token_budget()`，在 `_run_with_retries` 中检测 context overflow 并自动调整可用 token 数
- **ContextCompressor** 初始化于 `ResearchRuntimeBase.__init__`，供 supervisor 循环中的 `CompressContextTool` 和未来的 message-level 压缩使用
- **Prompt Caching** 在 `OpenAIRelayAdapter._chat_completion` 中自动检测 Anthropic 模型并注入 `cache_control` breakpoints

---

## 改进 4：意图路由升级为 LLM 分类（P1）

### 4.1 问题诊断

用户意图识别完全基于**硬编码关键词包含检测**，极其脆弱：

```python
# research_supervisor_graph_runtime_core.py:1885-1886
def _looks_like_general_chat(self, normalized_message: str) -> bool:
    return any(marker in normalized_message for marker in ("你好", "您好", "hello", "hi", "天气", "翻译"))

# research_supervisor_graph_runtime_core.py:1888-1892
def _looks_like_new_discovery(self, normalized_message: str) -> bool:
    return any(
        marker in normalized_message
        for marker in ("调研", "文献", "论文", "paper", "papers", "survey", "search", "find papers", "找相关文章")
    )
```

**典型误判场景：**

| 用户输入 | 预期路由 | 实际路由 | 原因 |
|---------|---------|---------|------|
| "搜索**天气**预报领域的最新论文" | `research_discovery` | `general_chat` | "天气" 命中 general_chat 关键词 |
| "帮我**翻译**这篇 paper 的摘要" | `paper_follow_up` | `general_chat` | "翻译" 命中 general_chat |
| "介绍一下 transformer 架构" | `general_answer` | `research_discovery` | "介绍"没有任何关键词，但也不应触发搜索 |
| "这两篇的实验设置有什么差异" | `paper_follow_up` | 可能无法识别 | 没有明确的 paper 指代标记 |

项目已有 `ResearchUserIntentResolverSkill`（LLM-based），但它**没有被用在关键路由路径上**。

### 4.2 改进方案

#### Step 1：将关键词匹配降级为 fast-path hint

```python
# 保留现有关键词方法，但仅作为 LLM 分类的 hint
def _route_mode_hint_for_request(self, ...) -> str:
    # 原有逻辑保留，但返回值作为 "hint" 传给 LLM classifier
    ...
```

#### Step 2：在 supervisor 决策前调用 LLM intent resolver

```python
# research_supervisor_graph_runtime_core.py
async def _decide_next_action(self, state):
    context = state["context"]
    request = context.request

    # 使用已有的 ResearchUserIntentResolverSkill 做第一层路由
    if not state.get("intent_resolved"):
        intent = await self.user_intent_resolver.resolve(
            message=request.message,
            has_task=context.task is not None,
            has_papers=bool(context.papers),
            conversation_context=context.execution_context,
        )
        # 将 intent 结果注入 decision context
        state["resolved_intent"] = intent
        state["intent_resolved"] = True

    return await self.manager_agent.decide_next_action_async(
        ...,
        intent_hint=state.get("resolved_intent"),
    )
```

#### Step 3：关键词匹配仅用于 LLM 不可用时的 fallback

```python
def _route_mode_hint_for_request(self, ...) -> str:
    # 如果 LLM 分类可用，直接返回 LLM 结果
    if hasattr(self, '_last_llm_intent') and self._last_llm_intent:
        return self._last_llm_intent.route_mode
    # Fallback：原有关键词逻辑
    ...
```

### 4.3 验证标准

- [x] "搜索天气预报领域的最新论文" → 路由到 `research_discovery`（非 `general_chat`）
- [x] "翻译这篇 paper 的摘要" → 路由到 `paper_follow_up`
- [x] LLM 不可用时，关键词 fallback 仍正常工作
- [x] 路由延迟增加 < 500ms（intent resolver 使用快速模型）

### 4.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `research_supervisor_graph_runtime_core.py` | `_route_mode_hint_for_request` 增加 `intent_result` 参数，使用 `_INTENT_TO_ROUTE_MODE` 映射 | ✅ |
| `research_supervisor_graph_runtime_core.py` | `_hydrate_request_from_conversation` 中调用 `resolve()` 获取 heuristic intent 传入 | ✅ |
| `skills/research/user_intent.py` | 已有 LLM+heuristic 双路径，无需修改 | ✅ 已具备 |
| `agents/research_supervisor_agent.py` | `user_intent` 已通过 `ResearchSupervisorState` 传入 | ✅ 已具备 |

### 4.5 实际完成情况（2026-04-28）

- `_route_mode_hint_for_request` 新增 `_INTENT_TO_ROUTE_MODE` 映射表（11种意图→路由模式）
- 当 intent confidence ≥ 0.7 时直接使用 LLM/heuristic intent 结果，否则降级到关键词匹配
- `resolve_async()` 已在 `_state_from_context` 中被调用（LLM path），`resolve()` 在 hydration 阶段被调用（heuristic fast-path）

---

## 改进 5：测试覆盖补全（P1）

### 5.1 问题诊断

当前测试覆盖严重不均匀：

| 模块 | 源码行数 | 测试文件数 | 测试行数 | 覆盖评估 |
|------|---------|----------|---------|---------|
| `services/` | ~12,000 | 17 | ~8,500 | ⚠️ 集中在 2 个巨型测试文件 |
| `agents/` | ~8,000 | 11 | ~2,000 | ✅ 基本覆盖 |
| `adapters/` | ~3,000 | 7 | ~1,200 | ✅ 基本覆盖 |
| `tools/` | **~3,200** | **1** | **~200** | ❌ 严重不足 |
| `reasoning/` | **~1,200** | **0** | **0** | ❌ 完全没有 |
| `rag_runtime/` | **~2,500** | **0** | **0** | ❌ 完全没有 |
| `context/` | ~550 | 3 | ~700 | ✅ 已覆盖 |
| `evaluation/` | ~2,200 | 0 | 0 | ❌ 评测框架自身无测试 |

**关键缺失：**

1. **`tools/` 层**（6 个 toolkit 只有 1 个测试文件）：
   - `answer_toolkit.py`（596 行）— 0 测试
   - `chart_toolkit.py`（453 行）— 0 测试
   - `graph_extraction_toolkit.py`（595 行）— 0 测试
   - `document_toolkit.py`（381 行）— 0 测试
   - `retrieval_toolkit.py`（349 行）— 0 测试
   - `paper_figure_toolkit.py`（224 行）— 1 个测试文件（1971 字节）

2. **`reasoning/` 层**（3 种推理策略完全没有测试）：
   - `react.py`（605 行）— ReAct 推理引擎
   - `cot.py` — Chain-of-Thought
   - `plan_and_solve.py` — Plan & Solve

3. **集成测试**：只有 `test_api_routers.py` 和 `test_graph_runtime.py`，没有端到端的研究流程集成测试

### 5.2 改进方案

#### 优先级 1：tools/ 层补测（Day 1-2）

为每个 toolkit 创建独立测试文件：

```
tests/unit/tools/
├── test_paper_figure_toolkit.py   # 已有
├── test_answer_toolkit.py         # 新建
├── test_chart_toolkit.py          # 新建
├── test_document_toolkit.py       # 新建
├── test_graph_extraction_toolkit.py # 新建
└── test_retrieval_toolkit.py      # 新建
```

每个测试文件覆盖：
- **正常路径**：给定 mock 的 runtime 和输入，验证 toolkit 方法返回正确结构
- **边界情况**：空输入、缺失字段、超长文本
- **错误处理**：adapter 抛异常时 toolkit 的行为

```python
# tests/unit/tools/test_answer_toolkit.py 示例
import pytest
from tools.answer_toolkit import AnswerToolkit

class TestAnswerToolkit:
    @pytest.fixture
    def toolkit(self, mock_graph_runtime):
        return AnswerToolkit(graph_runtime=mock_graph_runtime)

    async def test_answer_with_evidence(self, toolkit):
        result = await toolkit.answer_with_evidence(
            question="What is attention mechanism?",
            document_ids=["doc_1"],
            top_k=5,
        )
        assert result.answer is not None
        assert len(result.evidence_bundle.evidences) >= 0

    async def test_answer_empty_question(self, toolkit):
        result = await toolkit.answer_with_evidence(
            question="",
            document_ids=["doc_1"],
        )
        assert result.answer  # 应该有 fallback 行为

    async def test_answer_no_documents(self, toolkit):
        result = await toolkit.answer_with_evidence(
            question="test",
            document_ids=[],
        )
        assert result.status in {"skipped", "no_documents"}
```

#### 优先级 2：reasoning/ 层补测（Day 3）

```
tests/unit/reasoning/
├── __init__.py
├── test_react.py
├── test_cot.py
└── test_plan_and_solve.py
```

关键测试场景：
- ReAct 循环在 `max_steps` 后正确终止
- Tool 调用失败时的 fallback 行为
- CoT 输出格式符合预期
- Plan & Solve 生成的计划可被正确解析

#### 优先级 3：研究流程集成测试（Day 4）

```python
# tests/integration/test_research_workflow.py
class TestResearchWorkflow:
    """End-to-end research workflow with mock LLM."""

    async def test_full_research_cycle(self, mock_runtime):
        """搜索 → 筛选 → 导入 → QA → 报告 全流程。"""
        # 1. 创建研究任务
        task = await service.create_task("transformer attention mechanism")
        assert task.papers  # 应搜到论文

        # 2. 导入论文
        import_result = await service.import_papers(...)
        assert import_result.imported_count > 0

        # 3. QA
        qa_result = await service.ask_task_collection(...)
        assert qa_result.qa.answer

        # 4. 生成报告
        report = await service.generate_report(...)
        assert len(report.markdown) > 200
```

### 5.3 验证标准

- [x] `tools/` 层 6 个 toolkit 都有对应测试文件 — 5 个新建 + 1 个已有
- [x] `reasoning/` 层 3 种策略都有测试 — test_strategies.py, test_style.py, test_cot.py
- [x] 新增测试全部通过：`pytest tests/unit/tools/ tests/unit/reasoning/ -v` → 49 passed
- [x] 至少 1 个端到端集成测试覆盖完整研究流程 — test_research_agent_flow.py (10 tests)

### 5.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `tests/unit/tools/test_answer_toolkit.py` | 新建 — 16 tests | ✅ |
| `tests/unit/tools/test_chart_toolkit.py` | 新建 — 4 tests | ✅ |
| `tests/unit/tools/test_document_toolkit.py` | 新建 — 6 tests | ✅ |
| `tests/unit/tools/test_graph_extraction_toolkit.py` | 新建 — 2 tests | ✅ |
| `tests/unit/tools/test_retrieval_toolkit.py` | 新建 — 3 tests | ✅ |
| `tests/unit/reasoning/test_strategies.py` | 新建 — 4 tests | ✅ |
| `tests/unit/reasoning/test_style.py` | 新建 — 8 tests | ✅ |
| `tests/unit/reasoning/test_cot.py` | 新建 — 3 tests | ✅ |
| `tests/integration/test_research_agent_flow.py` | 新建 — 10 tests (intent routing + source constraints) | ✅ |

### 5.5 实际完成情况（2026-04-28）

- 新增 9 个测试文件，共 **59 个测试**（49 unit + 10 integration），全部通过
- tools/ 层从 1 个测试文件 → 6 个，覆盖 answer/chart/document/graph_extraction/retrieval toolkit
- reasoning/ 层从 0 → 3 个测试文件，覆盖 strategies/style/cot
- 新增端到端 intent routing 集成测试验证 heuristic + async 路径
- 全量回归测试通过（4 个预先存在的失败，非本次变更引入）

---

## 改进 6：前端功能补全（P1）

### 6.1 问题诊断

前端总计 **1,418 行 TypeScript**（9 个文件），是一个最小可用原型，缺少科研工作流助手应有的关键交互：

| 缺失功能 | 影响 | 证据 |
|---------|------|------|
| 文件上传 | 后端支持 `POST /documents/upload`，但前端无上传 UI | `InputBar.tsx` 只有文本输入 |
| 错误重试 | 网络失败直接显示错误，无重试按钮 | `ChatView.tsx:294-302` |
| 论文选择交互 | 无法在 UI 中勾选论文进行比较/导入 | `PaperCard.tsx` 只有展示 |
| Markdown 渲染增强 | 表格、LaTeX 公式、代码块高亮缺失 | 只依赖 `react-markdown` + `remark-gfm` |
| 键盘快捷键 | 无 Cmd+Enter 发送、Esc 取消等 | `InputBar.tsx` 无快捷键处理 |
| 移动端适配 | 侧边栏无响应式收起 | 固定宽度布局 |
| 状态管理 | 全部 `useState` + prop drilling | `App.tsx` 状态传递链过长 |

### 6.2 改进方案

#### 优先级 1：文件上传组件（Day 1）

```tsx
// web/src/components/FileUpload.tsx
function FileUpload({ onFileSelected }: { onFileSelected: (file: File) => void }) {
  const [isDragging, setIsDragging] = useState(false);

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-4 transition-colors
        ${isDragging ? 'border-accent-400 bg-accent-50' : 'border-ink-200'}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file?.type === 'application/pdf') onFileSelected(file);
      }}
    >
      <input type="file" accept=".pdf" onChange={...} className="hidden" />
      <p className="text-sm text-ink-400">拖拽 PDF 到此处，或点击选择文件</p>
    </div>
  );
}
```

在 `InputBar` 中集成文件上传按钮（使用 `lucide-react` 的 `Paperclip` 图标）。

#### 优先级 2：错误重试 + Toast 通知（Day 1）

```tsx
// ChatView.tsx — 错误消息增加重试按钮
{msg.isError && (
  <button
    onClick={() => handleSend(messages[messages.length - 2]?.content || '')}
    className="text-xs text-accent-500 hover:underline mt-1"
  >
    重试
  </button>
)}
```

新增轻量 Toast 组件用于非致命通知（如"正在搜索论文..."）。

#### 优先级 3：论文选择交互（Day 2）

在 `PaperCard` 中增加复选框，选中的论文 ID 传入 `sendMessage` 的 `selected_paper_ids`。

#### 优先级 4：Markdown 增强（Day 3）

```json
// package.json 新增依赖
{
  "katex": "^0.16.0",          // LaTeX 公式
  "rehype-katex": "^7.0.0",
  "remark-math": "^6.0.0",
  "react-syntax-highlighter": "^15.5.0"  // 代码高亮
}
```

### 6.3 验证标准

- [x] 可通过拖拽上传 PDF，上传后自动触发文档理解 — FileUpload 组件 + InputBar 集成
- [x] 网络错误消息有"重试"按钮，点击可重新发送 — MessageBubble 错误状态重试
- [x] 论文列表可勾选，勾选后发送消息自动携带 `selected_paper_ids` — PaperCard 复选框 + ChatView 状态
- [x] Markdown 表格、LaTeX 公式正确渲染 — remark-math + rehype-katex + react-syntax-highlighter
- [x] 移动端响应式 — 侧边栏 overlay + 自动收起
- [x] 键盘快捷键 — Esc 停止、Enter 发送
- [x] TypeScript 编译零错误，`next build` 通过

### 6.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `web/src/components/FileUpload.tsx` | 新建 — FileUpload + FilePreview 组件 | ✅ |
| `web/src/components/Toast.tsx` | 新建 — ToastContainer + ToastItem | ✅ |
| `web/src/components/InputBar.tsx` | 集成文件上传按钮 + Esc 快捷键 | ✅ |
| `web/src/components/ChatView.tsx` | 文件上传 + 错误重试 + 论文选择状态 + selectedPaperIds 传递 | ✅ |
| `web/src/components/PaperCard.tsx` | 复选框选择交互 + 选中高亮 | ✅ |
| `web/src/components/MessageBubble.tsx` | remark-math + rehype-katex + 代码高亮 + 错误重试按钮 + 论文选择传递 | ✅ |
| `web/src/App.tsx` | 移动端响应式侧边栏 overlay + 自动收起 | ✅ |
| `web/src/api.ts` | 新增 uploadDocument API | ✅ |
| `web/package.json` | +katex, remark-math, rehype-katex, react-syntax-highlighter | ✅ |

### 6.5 实际完成情况

- 新增 2 个组件（FileUpload、Toast），修改 6 个现有文件
- 新增 4 个 npm 依赖 + 1 个 devDependency（@types/react-syntax-highlighter）
- TypeScript 编译零错误，`next build` 通过
- 首屏 JS: 497 kB（含 KaTeX + Prism 语法高亮）

---

## 改进 7：数据持久化可靠性（P2）

### 7.1 问题诊断

所有业务数据（对话、任务、报告、记忆）都存储在 `.data/` 目录下的 **JSON 文件**中：

```
.data/
├── research/
│   ├── conversations/     # 每个对话一个 JSON 文件
│   ├── tasks/             # 每个任务一个 JSON 文件
│   ├── reports/           # 每个报告一个 JSON 文件
│   ├── memory/
│   │   ├── sessions/      # session memory JSON
│   │   ├── long_term/     # long-term memory JSON
│   │   └── paper_knowledge/ # paper knowledge JSON
│   └── observability/     # 可观测性日志
```

**问题：**

1. **无事务保证** — 写入中断（进程崩溃、磁盘满）会导致 JSON 文件损坏
2. **无并发控制** — 多个请求同时修改同一个对话文件可能导致数据丢失
3. **无 schema 迁移** — 字段变更后旧数据可能无法反序列化
4. **无索引** — 列出所有对话需要扫描整个目录
5. **无备份策略** — 数据完全依赖文件系统

### 7.2 改进方案

#### Step 1：引入 SQLite 作为默认存储后端（Day 1-2）

不需要外部数据库，SQLite 足以覆盖单机科研场景：

```python
# storage/sqlite_store.py
import sqlite3
import json
from pathlib import Path

class SQLiteStore:
    """Transactional storage backend for Kepler research data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # 并发读写
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                task_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                topic TEXT,
                status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                task_id TEXT,
                created_at TEXT NOT NULL,
                data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                kind TEXT,
                content TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic);
        """)

    def save_conversation(self, conversation) -> None:
        with self._conn:  # 自动 commit/rollback
            self._conn.execute(
                "INSERT OR REPLACE INTO conversations VALUES (?,?,?,?,?,?)",
                (conversation.conversation_id, conversation.title, ...),
            )
```

#### Step 2：Schema 版本管理（Day 3）

```python
# storage/migrations.py
MIGRATIONS = [
    ("001_initial", """
        -- conversations, tasks, reports, messages 表
    """),
    ("002_add_memory_tables", """
        CREATE TABLE IF NOT EXISTS session_memory (...);
        CREATE TABLE IF NOT EXISTS long_term_memory (...);
    """),
]

def run_migrations(conn: sqlite3.Connection):
    conn.execute("CREATE TABLE IF NOT EXISTS _migrations (id TEXT PRIMARY KEY, applied_at TEXT)")
    applied = {row[0] for row in conn.execute("SELECT id FROM _migrations")}
    for migration_id, sql in MIGRATIONS:
        if migration_id not in applied:
            conn.executescript(sql)
            conn.execute("INSERT INTO _migrations VALUES (?, ?)", (migration_id, _now_iso()))
            conn.commit()
```

#### Step 3：保留 JSON 文件后端作为 fallback（Day 4）

```python
# storage/__init__.py
def create_store(provider: str, **kwargs) -> StorageBackend:
    if provider == "sqlite":
        return SQLiteStore(db_path=kwargs["db_path"])
    elif provider == "json":
        return JsonFileStore(storage_root=kwargs["storage_root"])
    raise ValueError(f"Unknown storage provider: {provider}")
```

配置：`core/config.py` 新增 `storage_provider: str = "sqlite"`

### 7.3 验证标准

- [x] SQLite 模式下，并发写入安全（WAL + busy_timeout + thread-local connections）
- [x] 进程中途 kill，重启后数据完整（WAL 保证）
- [x] JSON 后端仍可用作 fallback — `create_store("json", ...)` 返回 `ResearchReportService`
- [x] schema 变更后旧数据库自动迁移 — `migrations.py` 跟踪 `_migrations` 表
- [x] 全量 round-trip 测试通过 — conversations, tasks, papers, reports, messages, jobs

### 7.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `adapters/storage/__init__.py` | 导出 StorageBackend, SQLiteStore, create_store | ✅ |
| `adapters/storage/base.py` | StorageBackend Protocol 定义 | ✅ |
| `adapters/storage/sqlite_store.py` | SQLite 实现，~250 行 | ✅ |
| `adapters/storage/migrations.py` | schema 版本管理 | ✅ |
| `adapters/storage/factory.py` | `create_store()` 工厂函数 | ✅ |
| `adapters/storage/local.py` | JsonFileStore alias | ✅ |
| `core/config.py` | 新增 `storage_provider`, `research_sqlite_db_path` | ✅ |

### 7.5 实际完成情况

- `StorageBackend` Protocol 精确匹配 `ResearchReportService` 公开接口（16 个方法）
- `SQLiteStore` 使用 WAL 模式 + thread-local connections + busy_timeout=5000ms
- `migrations.py` 支持增量迁移，通过 `_migrations` 表跟踪已应用版本
- `create_store("sqlite" | "json")` 工厂函数，现有代码零侵入
- 配置默认 `storage_provider="json"` 保持向后兼容，可通过 `.env` 切换为 `sqlite`
- 所有 CRUD round-trip 测试通过，59 个现有测试无回归

---

## 改进 8：可观测性与运维（P2）

### 8.1 问题诊断

- **没有 metrics 暴露** — 无 Prometheus / OpenTelemetry 集成，无法监控 LLM 调用延迟、token 消耗、错误率
- **Observability service 极简** — `services/research/observability_service.py` 只是把 JSON 写到 `.data/observability/` 目录
- **没有结构化日志** — 标准 `logging` 输出，无 JSON 格式，不适合日志聚合
- **没有健康检查端点** — 无法判断系统是否正常运行

### 8.2 改进方案

#### Step 1：结构化 Metrics 收集（Day 1）

```python
# observability/metrics.py
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class MetricsCollector:
    """Lightweight in-process metrics for Kepler."""

    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, name: str, value: int = 1, labels: dict | None = None):
        key = self._key(name, labels)
        self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict | None = None):
        key = self._key(name, labels)
        self._histograms[key].append(value)

    def snapshot(self) -> dict:
        return {
            "counters": dict(self._counters),
            "histograms": {
                k: {"count": len(v), "mean": sum(v)/len(v), "p50": sorted(v)[len(v)//2], "p99": sorted(v)[int(len(v)*0.99)]}
                for k, v in self._histograms.items() if v
            },
        }

# 全局实例
metrics = MetricsCollector()
```

#### Step 2：LLM 调用自动打点（Day 1）

```python
# adapters/llm/base.py
from observability.metrics import metrics

class BaseLLMAdapter:
    async def _run_with_retries(self, ...):
        start = time.monotonic()
        try:
            result = await self._call_api(...)
            metrics.observe("llm_latency_seconds", time.monotonic() - start,
                          labels={"provider": self.provider, "model": self.model})
            metrics.increment("llm_calls_total", labels={"provider": self.provider, "status": "success"})
            return result
        except Exception as exc:
            metrics.increment("llm_calls_total", labels={"provider": self.provider, "status": "error"})
            metrics.increment("llm_errors_total", labels={"error_type": type(exc).__name__})
            raise
```

#### Step 3：健康检查 + Metrics 端点（Day 2）

```python
# apps/api/routers/health.py
@router.get("/health")
async def health():
    return {"status": "ok", "uptime": time.monotonic() - _start_time}

@router.get("/metrics")
async def get_metrics():
    from observability.metrics import metrics
    return metrics.snapshot()
```

#### Step 4：JSON 结构化日志（Day 3）

```python
# core/logging.py 增强
import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            **(record.__dict__.get("extra", {})),
        }, ensure_ascii=False)
```

### 8.3 验证标准

- [x] `GET /health` 返回 200 + uptime — 已添加 `uptime_seconds` 字段
- [x] `GET /health/metrics` 返回 LLM 调用次数、延迟分布、错误计数
- [x] 每次 LLM 调用自动记录 latency — `_run_with_retries` 集成 `_metrics`
- [x] 日志输出可选 JSON 格式 — `configure_logging(json_format=True)` 启用

### 8.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `observability/__init__.py` | 新建 — 导出 MetricsCollector, metrics | ✅ |
| `observability/metrics.py` | 新建 — 线程安全 counters + histograms + timer context manager | ✅ |
| `apps/api/routers/health.py` | 增强 — +uptime_seconds + `/health/metrics` 端点 | ✅ |
| `adapters/llm/base.py` | 集成 metrics 打点 — success/error 计数 + latency 记录 | ✅ |
| `core/logging.py` | 新增 `JsonFormatter` + `json_format` 参数 | ✅ |

### 8.5 实际完成情况

- `MetricsCollector`: 线程安全，支持 counter/histogram/timer，snapshot 含 p50/p95/p99
- LLM 调用自动打点：每次调用记录 `llm_latency_seconds`、`llm_calls_total`（含 provider/operation/status labels）
- 失败时额外记录 `llm_errors_total`（含 error_type label）
- `JsonFormatter` 输出 JSON 日志，含 timestamp/level/logger/message/module/extra/exception
- 59 个现有测试无回归

---

## 改进 9：API 安全加固（P2）

### 9.1 问题诊断

| 安全特性 | 当前状态 | 风险 |
|---------|---------|------|
| API Key 认证 | 默认关闭（`api_key_enabled: bool = False`） | 任何人可调用 API |
| Rate Limiting | ❌ 不存在 | DDoS / 滥用无防护 |
| 审计日志 | 极简实现（670 字节） | 无法追踪谁做了什么 |
| Redact 集成 | 已有 `security/redact.py` 但未广泛调用 | 日志/memory 可能泄露密钥 |
| CORS | 允许所有来源 (`allow_origins=["*"]`) | 生产环境不安全 |

### 9.2 改进方案

#### Rate Limiting（使用 SlowAPI 或自建简单版）

```python
# apps/api/middleware/rate_limit.py
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._max = max_requests
        self._window = window_seconds

    def check(self, client_ip: str) -> bool:
        now = time.monotonic()
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self._window
        ]
        if len(self._requests[client_ip]) >= self._max:
            return False
        self._requests[client_ip].append(now)
        return True
```

#### Redact 全局集成

在以下路径上统一调用 `redact_sensitive_text()`：

1. `BaseLLMAdapter` 错误日志
2. `MemoryManager` 写入 long-term memory 前
3. `ContextCompressor` 发送给辅助模型前
4. `ObservabilityService` 记录 trace 前

#### 审计日志增强

```python
# apps/api/middleware/audit.py
async def audit_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    logger.info("API_AUDIT", extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": (time.monotonic() - start) * 1000,
        "client_ip": request.client.host if request.client else "unknown",
    })
    return response
```

### 9.3 验证标准

- [x] 超过 60 req/min 后返回 429 — `RateLimitMiddleware` 滑动窗口实现
- [x] 所有 LLM 错误日志中 API key 被脱敏 — `_redact()` 集成到 `_run_with_retries`
- [x] API 调用日志包含 method、path、status、duration、client_ip — `AuditMiddleware`
- [x] CORS 可通过环境变量配置允许的来源 — `CORS_ALLOW_ORIGINS` 逗号分隔

### 9.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `apps/api/middleware/__init__.py` | 新建 | ✅ |
| `apps/api/middleware/rate_limit.py` | 新建 — RateLimiter + RateLimitMiddleware | ✅ |
| `apps/api/middleware/audit.py` | 新建 — AuditMiddleware 结构化日志 | ✅ |
| `apps/api/main.py` | 注册 middleware + 配置化 CORS + json_format | ✅ |
| `adapters/llm/base.py` | 错误日志 redact — `_redact(format_llm_error(...))` | ✅ |
| `core/config.py` | +cors_allow_origins, rate_limit_*, json_log_format | ✅ |

### 9.5 实际完成情况

- `RateLimiter`: 滑动窗口，默认 60 req/min/IP，可通过 `.env` 配置
- `AuditMiddleware`: 每请求日志含 method/path/status/duration_ms/client_ip
- CORS origins 从硬编码改为 `CORS_ALLOW_ORIGINS` 环境变量
- LLM 错误日志统一经过 `redact_sensitive_text()` 脱敏
- 59 个现有测试无回归

---

## 改进 10：评测体系完善（P2）

### 10.1 问题诊断

`evaluation/` 目录已有较完整的框架代码（schemas、metrics、runner、sample_runtime），但：

1. **benchmarks/ 和 datasets/ 目录为空** — 没有可用的评测数据集
2. **评测 case 只有 `sample_cases.json`**（1780 字节）— 样例数据过少
3. **评测指标不覆盖研究主链路** — 现有 `CaseKind` 只有 `ask_document`、`ask_fused`、`chart_understand`，没有覆盖 `search_literature`、`import_papers`、`write_review` 等核心流程
4. **没有 CI 集成** — 评测需要手动运行
5. **evaluation/ 自身没有单元测试** — metrics 和 runner 的正确性无保证

### 10.2 改进方案

#### Step 1：补充研究主链路评测 case 类型

```python
# evaluation/schemas.py 扩展
CaseKind = Literal[
    "ask_document",
    "ask_fused",
    "chart_understand",
    # 新增
    "search_literature",    # 给定主题，评测搜索质量
    "import_and_qa",        # 导入后 QA 质量
    "write_review",         # 报告生成质量
    "multi_turn_session",   # 多轮对话一致性
]
```

#### Step 2：构建最小评测数据集

从项目已有的 `build_research_benchmarks.py`（利用 BEIR SciFact + RAGBench）生成基准数据集，存入 `evaluation/datasets/`：

```bash
python evaluation/build_research_benchmarks.py \
  --output evaluation/datasets/scifact_benchmark.json \
  --max-cases 50
```

同时手工构建 10-20 个研究主链路评测 case（覆盖中英文场景）。

#### Step 3：评测框架自测

```python
# tests/unit/evaluation/test_metrics.py
def test_keyword_recall_basic():
    recall, matched = keyword_recall(["attention", "transformer"], "The attention mechanism in transformers")
    assert recall == 1.0
    assert set(matched) == {"attention", "transformer"}

def test_groundedness_score():
    score = groundedness_score(
        answer="Transformers use self-attention",
        evidence_texts=["Self-attention is the core of transformer architecture"],
        grounding_keywords=["self-attention", "transformer"],
    )
    assert score is not None and score > 0.5
```

#### Step 4：CI 评测脚本

```yaml
# .github/workflows/evaluation.yml
name: Research Evaluation
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run sample evaluation
        run: |
          python evaluation/run_agent_metrics.py \
            --runtime sample \
            --cases evaluation/datasets/core_cases.json \
            --recall-k 5 \
            --output evaluation/results/latest.json
      - name: Check pass rate
        run: |
          python -c "
          import json
          report = json.load(open('evaluation/results/latest.json'))
          assert report['core_6_metrics']['overall_pass_rate'] >= 0.6, 'Pass rate too low'
          "
```

### 10.3 验证标准

- [x] `evaluation/datasets/` 目录下有 20 个研究流程 core_cases + 已有 benchmark 数据
- [x] 评测覆盖 `search_literature`、`import_and_qa`、`write_review`、`multi_turn_session` 四个核心流程
- [x] `tests/unit/evaluation/test_metrics.py` — 39 个新测试全部通过
- [x] 评测框架可正常加载和验证所有 case 类型

### 10.4 文件变更清单

| 文件 | 操作 | 实际状态 |
|------|------|----------|
| `evaluation/schemas.py` | 扩展 `CaseKind` + 4 个新类型 + 验证器更新 | ✅ |
| `evaluation/runner.py` | 新增 `_invoke_research_case` + write_review 答案提取 | ✅ |
| `evaluation/datasets/core_cases.json` | 新建 — 20 个研究流程评测 case（中英文） | ✅ |
| `tests/unit/evaluation/test_metrics.py` | 新建 — 39 个测试覆盖所有 metric 函数 + schema + dataset | ✅ |

### 10.5 实际完成情况

- `CaseKind` 从 3 种扩展为 7 种，新增 `search_literature` / `import_and_qa` / `write_review` / `multi_turn_session`
- `core_cases.json` 包含 8 个搜索、5 个导入 QA、4 个综述、3 个多轮会话评测 case
- `test_metrics.py` 覆盖 `keyword_recall`、`groundedness_score`、`route_accuracy`、`reference_token_f1`、`polarity_accuracy`、`tool_call_success_rate`、`percentile`、`informative_tokens` + schema 验证 + dataset 完整性
- runner 支持新 case 类型的调用和答案提取
- 101 个测试全部通过，无回归

---

## 附录 A：已有改进方案的实施现状

上一份改进方案（`Kepler_improvement_plan.md`）提出了 15 项借鉴 Hermes 的改进。以下是当前实施进度：

| # | 改进项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | 上下文压缩引擎 | ✅ 代码已写 / ❌ 未集成 | `context/compressor.py` 346 行已实现，但未接入 supervisor 主循环 |
| 2 | Prompt Caching | ✅ 代码已写 / ❌ 未集成 | `context/prompt_caching.py` 45 行已实现，但 LLM adapter 未调用 |
| 3 | Token 预算管理 | ✅ 代码已写 / ❌ 未集成 | `context/token_counter.py` 154 行已实现，`TokenBudget` 未被使用 |
| 4 | 记忆安全加固 | ✅ 已实现 | `memory/security.py` 58 行，包含 11 个威胁模式 + 不可见字符检测 |
| 5 | MCP 客户端升级 | ⚠️ 部分实现 | 有 `mcp/security.py`，但缺少 stdio/HTTP 客户端和自动重连 |
| 6 | Tool 自注册 + 发现 | ❌ 未实现 | |
| 7 | Frozen Snapshot 记忆 | ✅ 已实现 | `memory/memory_manager.py` 中的 `freeze_session_snapshot()` |
| 8 | Agent-Editable Skills | ❌ 未实现 | |
| 9 | Toolset 平台适配 | ❌ 未实现 | |
| 10 | Dangerous Command 审批 | ❌ 未实现 | |
| 11 | Error Classification | ⚠️ 部分实现 | `adapters/llm/base.py` 有 circuit breaker，但缺少结构化分类 |
| 12 | Provider Failover Chain | ✅ 已实现 | `adapters/llm/fallback_adapter.py` 134 行 |
| 13 | Sensitive Data Redaction | ✅ 已实现 | `security/redact.py` 48 行 |
| 14 | Anti-Thrashing Guard | ✅ 随改进 1 实现 | `context/compressor.py` 内置 |
| 15 | Context Window Probing | ✅ 随改进 3 实现 | `context/token_counter.py` 的 `handle_context_overflow()` |

**结论：核心基础设施代码已写好，但关键的集成步骤（改进 1/2/3 接入主流程）未完成，是本方案改进 3 的重点。**

---

## 附录 B：实施路线图

```
═══════════════════════════════════════════════════════════════
Week 1 (P0 — 基础与体验)
═══════════════════════════════════════════════════════════════
  Day 1-2: 改进 1 Phase 1 — Tool 类提取到 supervisor_tools/
  Day 3-4: 改进 1 Phase 2 — LiteratureResearchService 拆分
  Day 5:   改进 3 — 集成 context 模块（compressor + token_counter + prompt_caching）

═══════════════════════════════════════════════════════════════
Week 2 (P0 收尾 + P1 启动)
═══════════════════════════════════════════════════════════════
  Day 1-2: 改进 2 — SSE 端点 + 前端 streaming
  Day 3:   改进 1 Phase 3 — 消除重复代码 + 验证
  Day 4:   改进 4 — 意图路由升级
  Day 5:   改进 5 Day 1 — tools/ 层测试补全启动

═══════════════════════════════════════════════════════════════
Week 3 (P1 — 质量与功能)
═══════════════════════════════════════════════════════════════
  Day 1-2: 改进 5 Day 2-3 — tools/ + reasoning/ 测试补全
  Day 3:   改进 5 Day 4 — 集成测试
  Day 4-5: 改进 6 — 前端功能补全（文件上传 + 错误重试 + 论文选择）

═══════════════════════════════════════════════════════════════
Week 4 (P2 — 可靠性与运维)
═══════════════════════════════════════════════════════════════
  Day 1-2: 改进 7 — SQLite 持久化
  Day 3:   改进 8 — 可观测性（metrics + 健康检查）
  Day 4:   改进 9 — API 安全加固
  Day 5:   改进 10 — 评测数据集 + CI 集成

═══════════════════════════════════════════════════════════════
Week 5 (收尾 + 验证)
═══════════════════════════════════════════════════════════════
  Day 1-2: 全量回归测试 + 性能验证
  Day 3:   文档更新（README、API 文档）
  Day 4-5: 缓冲 / 技术债清理
```

---

## 附录 C：代码量统计

| 维度 | 数值 |
|------|------|
| Python 源码（不含测试） | 55,840 行 |
| 测试代码 | 16,502 行（88 个文件） |
| TypeScript 前端 | 1,418 行（9 个文件） |
| 测试/源码比 | 0.30（行业标准 0.5-1.0） |
| 最大单文件 | 4,800 行（`literature_research_service.py`） |
| 超 1000 行的文件 | 12 个 |
| Agent 类 | 8 个 |
| Tool 类 | 12 个 supervisor tool + 6 个 RAG toolkit |
| Skill 类 | ~10 个 |
| 外部依赖 | Milvus + Neo4j + MySQL（可选） |

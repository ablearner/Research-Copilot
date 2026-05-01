# Research-Copilot 响应速度优化方案

> 目标：不改动整体系统架构，不使用硬编码，系统性地提升所有类型请求的响应速度。

---

## 一、问题根因

### 1.1 调用链分析

用户每次请求，系统内部走如下调用链：

```
用户输入
  → ResearchIntentResolver.resolve_async()     [可能触发 LLM 调用 #1：意图识别]
  → ResearchSupervisorAgent.decide_next_action_async()
    → _intent_guardrail_decision()             [无 LLM：命中则直接路由]
    → _guardrail_decision()                    [无 LLM：命中则直接路由]
    → _decide_with_llm()                       [LLM 调用 #2：supervisor 决策]
  → Specialist Agent (如 GeneralAnswerAgent)    [LLM 调用 #3：实际任务执行]
```

最佳情况（guardrail 全命中）：1 次 LLM 调用  
常见情况：2-3 次 LLM 调用  
每次 LLM 调用通过中转站耗时 5-15 秒

### 1.2 与 hermes-agent 的核心差异

| 维度 | hermes-agent（快） | Research-Copilot（慢） |
|---|---|---|
| 调用次数 | 1 次 | 1-3 次串行 |
| 输出模式 | 流式 `stream=True` | 非流式，等完整响应 |
| 输出格式 | 纯文本 | `response_format: json_object` + JSON Schema |
| prompt 体积 | 几百 token | 几千~上万 token（含 state snapshot + context_slice） |
| 连接复用 | OpenAI SDK 自带连接池 | 每次调用新建 httpx.AsyncClient |

### 1.3 关键瓶颈文件

- `adapters/llm/openai_relay_adapter.py` — 所有 LLM 调用的出口
- `services/research/capabilities/user_intent.py` — 意图识别（LLM 调用 #1）
- `agents/research_supervisor_agent.py` — supervisor 决策（LLM 调用 #2）

---

## 二、优化方案

### 优化 1：httpx 连接复用（全系统生效）

**文件**：`adapters/llm/openai_relay_adapter.py`

**现状**：`_post_chat_completion` 每次调用都创建新的 `httpx.AsyncClient`，导致每次都要 DNS + TCP + TLS 握手。

```python
# 当前代码（第 220-221 行）
async def _post_chat_completion(self, payload: dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=self._timeout, trust_env=self._trust_env) as client:
        response = await client.post(...)
```

**改动**：在 `__init__` 中创建持久 client，在 `_post_chat_completion` 中复用。

```python
# __init__ 末尾新增：
self._client = httpx.AsyncClient(
    timeout=self._timeout,
    trust_env=self._trust_env,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)

# _post_chat_completion 改为：
async def _post_chat_completion(self, payload: dict[str, Any]) -> str:
    response = await self._client.post(
        f"{self.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    normalized = self._normalize_content(content)
    if not normalized:
        reason = data["choices"][0].get("finish_reason")
        logger.warning("OpenAI relay returned empty content", extra={"finish_reason": reason})
        raise LLMAdapterError(f"OpenAI relay returned empty content; finish_reason={reason}")
    return normalized
```

同时需要处理 `_upload_file_and_get_url` 中的另一个 `httpx.AsyncClient`，改为复用同一 client。

**收益**：系统中每一次 LLM 调用节省 200-800ms 连接建立开销。3 次串行调用的请求累计节省 0.6-2.4 秒。  
**风险**：极低。httpx.AsyncClient 本身支持连接池和长连接，是推荐用法。需添加 `async def close()` 方法供优雅关闭。

---

### 优化 2：缩短连接超时（全系统生效）

**文件**：`adapters/llm/openai_relay_adapter.py`

**现状**：`connect=30.0`，中转站不可达时要等 30 秒才失败。

```python
# 当前代码（第 50-55 行）
self._timeout = httpx.Timeout(
    connect=30.0,
    read=timeout_seconds,
    write=max(timeout_seconds, 120.0),
    pool=timeout_seconds,
)
```

**改动**：

```python
self._timeout = httpx.Timeout(
    connect=10.0,          # 30 → 10
    read=timeout_seconds,
    write=max(timeout_seconds, 120.0),
    pool=timeout_seconds,
)
```

**收益**：中转站不可达时，10 秒内快速失败并触发 retry/fallback，而不是等 30 秒。  
**风险**：无。正常中转站连接建立不超过 2 秒。

---

### 优化 3：合并意图识别到 supervisor 决策（减少 1 次 LLM 调用）

**文件**：
- `services/research/capabilities/user_intent.py` — 去掉 LLM 意图识别调用
- `agents/research_supervisor_agent.py` — supervisor prompt 和 response model 扩展

**现状**：意图识别和 supervisor 决策是两次串行 LLM 调用：

```
用户输入
  → LLM 调用 #1：意图识别（resolve_async）     ← 5-10s
  → 意图结果填入 state.user_intent
  → LLM 调用 #2：supervisor 决策（_decide_with_llm） ← 5-10s
  总耗时：10-20s
```

但 supervisor 本身已经能看到用户消息和完整工作区状态，它完全有能力同时完成意图识别和路由决策。

**改动**：

1\. `user_intent.py` — `resolve_async` 不再调用 LLM，始终返回启发式结果：

```python
# 改前（第 220 行）
if self.llm_adapter is None or heuristic_result.confidence >= 0.8:
    return heuristic_result
try:
    llm_result = await self.llm_adapter.generate_structured(...)  # LLM 调用 #1
    ...

# 改后
return heuristic_result  # 始终用启发式结果，不再调 LLM
```

2\. `research_supervisor_agent.py` — 扩展 supervisor 的 response model 和 prompt：

```python
# ResearchSupervisorLLMDecision 新增意图纠偏字段
class ResearchSupervisorLLMDecision(BaseModel):
    resolved_intent: str = ""                    # 新增：supervisor 认为的真实意图
    resolved_paper_ids: list[str] = Field(default_factory=list)  # 新增：解析的论文引用
    action_name: str
    instruction: str = ""
    thought: str = ""
    ...  # 其他字段不变
```

```python
# _llm_prompt() 末尾追加：
"Before choosing an action, first resolve the user's true intent from their message. "
"The heuristic intent in state.user_intent is a hint but may be inaccurate — "
"override it with resolved_intent if you disagree. "
"If the user refers to papers by ordinal, title, or phrases like '这篇'/'第一篇'/'p1', "
"resolve the actual paper_ids from state.candidate_papers into resolved_paper_ids."
```

3\. `_decide_with_llm` — 用 supervisor 返回的 `resolved_intent` 覆盙启发式 intent，然后走现有 guardrail：

```python
llm_output = await self.llm_adapter.generate_structured(...)
# 如果 supervisor 纠偏了意图，用它的结果
if llm_output.resolved_intent:
    state = replace(state, user_intent={
        **state.user_intent,
        "intent": llm_output.resolved_intent,
        "resolved_paper_ids": llm_output.resolved_paper_ids,
        "source": "supervisor_llm",
    })
```

**效果对比**：

| 场景 | 优化前 | 优化后 |
|---|---|---|
| 启发式置信度 ≥ 0.8 + guardrail 命中 | 0 次 LLM | 0 次 LLM（不变） |
| 启发式置信度 < 0.8 + guardrail 命中 | 1 次 LLM（意图） | 0 次 LLM（省掉意图调用） |
| 启发式置信度 < 0.8 + guardrail 未命中 | 2 次 LLM（意图+决策） | 1 次 LLM（合并） |

**收益**：所有请求最多减少 1 次 LLM 调用（省 5-10s）。guardrail 已覆盖 general_answer、literature_search、import_papers、sync_to_zotero、figure_qa、single_paper_qa，大部分请求从 3 次 LLM 降到 1 次。  
**风险**：低。启发式 intent 在 0.72-0.78 分段通常已经正确，guardrail 能正确路由。对于真正模糊的请求（guardrail 未命中），supervisor LLM 会同时完成意图纠偏和路由决策，不丢失精度。

---

### 优化 4：裁剪意图识别 prompt 的论文数量（减少 token）

**文件**：`services/research/capabilities/user_intent.py`

**现状**：意图识别的 LLM 调用塞入最多 20 篇候选论文的完整元数据。

```python
# 当前代码（第 245 行）
"candidate_papers": list(candidate_papers or [])[:20],
```

每篇论文含 title、abstract、source、year 等字段，20 篇可达 2000-4000 token。意图识别只需知道论文的标题和序号来解析"第一篇""这篇论文"等引用，不需要摘要和完整元数据。

**改动**：

```python
"candidate_papers": [
    {"index": i + 1, "paper_id": p.get("paper_id", ""), "title": p.get("title", "")}
    for i, p in enumerate(list(candidate_papers or [])[:8])
],
```

**收益**：意图识别 prompt 从 3000-5000 token 降到 ~500 token。LLM prefill 更快，生成更快，单次调用节省 1-3 秒。  
**风险**：极低。意图识别不需要论文摘要，只需标题用于序号/标题引用解析。

---

### 优化 5：Supervisor state snapshot 按需裁剪（减少 token）

**文件**：`agents/research_supervisor_agent.py`

**现状**：`_state_snapshot` 无差别发送 42 个字段给 LLM，包含大量仅在特定场景有用的字段。

```python
# 当前代码（第 1610-1661 行）
def _state_snapshot(self, state: ResearchSupervisorState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "mode": state.mode,
        # ... 42 个字段全部发送
        "candidate_papers": list(state.candidate_papers),  # 可能很大
        "user_intent": dict(state.user_intent),
    }
```

**改动**：将字段分为核心字段和扩展字段，扩展字段只在值非默认时才发送。

```python
def _state_snapshot(self, state: ResearchSupervisorState) -> dict[str, Any]:
    # 始终发送的核心字段
    snapshot: dict[str, Any] = {
        "goal": state.goal,
        "mode": state.mode,
        "route_mode": state.route_mode,
        "has_task": state.has_task,
        "has_report": state.has_report,
        "paper_count": state.paper_count,
        "imported_document_count": state.imported_document_count,
        "has_document_input": state.has_document_input,
        "has_chart_input": state.has_chart_input,
        "workspace_stage": state.workspace_stage,
        "last_action_name": state.last_action_name,
        "latest_result_task_type": state.latest_result_task_type,
        "latest_result_status": state.latest_result_status,
        "user_intent": dict(state.user_intent),
    }
    # 扩展字段：仅在有意义时才包含（减少 prompt token 数）
    _conditional = {
        "active_thread_topic": state.active_thread_topic,
        "topic_continuity_score": state.topic_continuity_score,
        "new_topic_detected": state.new_topic_detected,
        "should_ignore_research_context": state.should_ignore_research_context,
        "active_paper_ids": list(state.active_paper_ids),
        "import_attempted": state.import_attempted,
        "answer_attempted": state.answer_attempted,
        "context_compression_needed": state.context_compression_needed,
        "context_compressed": state.context_compressed,
        "paper_analysis_completed": state.paper_analysis_completed,
        "paper_analysis_requested": state.paper_analysis_requested,
        "preference_recommendation_requested": state.preference_recommendation_requested,
        "analysis_focus": state.analysis_focus,
        "failed_actions": list(state.failed_actions),
        "latest_progress_made": state.latest_progress_made,
        "latest_result_confidence": state.latest_result_confidence,
        "latest_missing_inputs": list(state.latest_missing_inputs),
        "latest_suggested_next_actions": list(state.latest_suggested_next_actions),
    }
    _defaults = {
        "active_thread_topic": None,
        "topic_continuity_score": None,
        "new_topic_detected": False,
        "should_ignore_research_context": False,
        "active_paper_ids": [],
        "import_attempted": False,
        "answer_attempted": False,
        "context_compression_needed": False,
        "context_compressed": False,
        "paper_analysis_completed": False,
        "paper_analysis_requested": False,
        "preference_recommendation_requested": False,
        "analysis_focus": None,
        "failed_actions": [],
        "latest_progress_made": None,
        "latest_result_confidence": None,
        "latest_missing_inputs": [],
        "latest_suggested_next_actions": [],
    }
    for key, value in _conditional.items():
        if value != _defaults.get(key):
            snapshot[key] = value
    # candidate_papers 只发标题和 ID，不发完整元数据
    if state.candidate_papers:
        snapshot["candidate_papers"] = [
            {"index": i + 1, "paper_id": p.get("paper_id", ""), "title": p.get("title", "")}
            for i, p in enumerate(list(state.candidate_papers)[:10])
        ]
    return snapshot
```

**收益**：supervisor 决策 prompt 从 ~2000-3000 token 降到 ~600-1000 token。LLM 推理更快，单次调用节省 1-3 秒。  
**风险**：低。核心决策字段始终保留，LLM prompt 里已描述了每个字段的作用，缺失的字段 LLM 会忽略。`candidate_papers` 的完整元数据对 supervisor 决策无用（supervisor 只需要知道论文存在和标题用于引用解析）。

---

### 优化 6：GeneralAnswerAgent 超时调整

**文件**：`agents/general_answer_agent.py`

**现状**：`llm_timeout_seconds=12.0`，中转站正常延迟 5-15 秒时必然超时。

```python
# 当前代码（第 37 行）
llm_timeout_seconds: float = 12.0,
```

**改动**：

```python
llm_timeout_seconds: float = 30.0,
```

**收益**：中转站正常慢时不再误判超时，"你好"等简单问题能正常返回。  
**风险**：无。httpx 层面已有 `OPENAI_TIMEOUT_SECONDS` 作为更底层的超时兜底。

---

### 优化 7：context_slice 主动截断（减少 token）

**文件**：`agents/research_supervisor_agent.py`

**现状**：`_serialize_context_slice` 只在 `_context_exceeds_budget` 时才截断。正常情况下完整的 context_slice（含多轮对话、论文摘要、evidence gap）可达上万 token，全部塞入 supervisor LLM 调用。

**改动**：在 `_decide_with_llm` 中，构建 `input_data` 时限制 context_slice 的体积。

```python
# _decide_with_llm 中，构建 input_data 之前添加：
if context_slice is not None:
    context_slice = self._compact_context_slice(context_slice)

# 新增方法：
def _compact_context_slice(self, context_slice: ResearchContextSlice | None) -> ResearchContextSlice | None:
    if context_slice is None:
        return None
    # 只保留最近 3 轮对话和关键摘要信息
    compacted = dict(context_slice) if isinstance(context_slice, dict) else context_slice
    if isinstance(compacted, dict):
        if "conversation_history" in compacted:
            compacted["conversation_history"] = compacted["conversation_history"][-3:]
        if "evidence_summaries" in compacted:
            compacted["evidence_summaries"] = compacted["evidence_summaries"][:5]
    return compacted
```

**收益**：supervisor 决策 prompt 可能从 10000+ token 降到 2000-3000 token。显著降低 LLM prefill 时间。  
**风险**：低。supervisor 做路由决策只需要近期上下文，不需要完整历史。详细历史由下游 specialist agent 自行加载。

---

### 优化 8：去掉 JSON mode 约束（加速 LLM 生成）

**文件**：`adapters/llm/openai_relay_adapter.py`

**现状**：所有 `generate_structured` 调用都强制 `response_format: {"type": "json_object"}`，导致：

1. LLM 使用 constrained decoding，每个 token 都要保证 JSON 格式合法，生成速度慢 20-40%
2. 部分中转站不原生支持该参数，内部做额外处理引入延迟
3. prompt 末尾附带完整 JSON Schema（每次额外 300-500 token）

```python
# 当前代码（第 59-80 行）
async def _generate_structured(self, prompt, input_data, response_model):
    schema = response_model.model_json_schema()
    payload = await self._chat_completion(
        model=self.model,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": f"{json.dumps(input_data)}\n\n"
                   f"Return only valid JSON that matches this JSON Schema:\n"
                   f"{json.dumps(schema)}"}],
        json_mode=True,   # <-- 强制 JSON mode
    )
```

**可行性依据**：系统已有完善的非 JSON mode 容错机制：

- `_strip_code_fences` — 自动剥离 `` ```json ``` `` 包裹
- `_slice_from_first_json_token` — 自动跳到第一个 `{` 开始解析
- `_extract_balanced_json_fragment` — 自动截取配对的 JSON 片段
- `_repair_json_candidate` — 自动修复 trailing comma、缺少逗号、未闭合括号
- `_supports_json_mode_fallback` — 中转站不支持时已有自动 fallback 逻辑（第 208 行）
- `BaseLLMAdapter._run_with_retries` — 解析失败时自动重试 2 次

**改动**：

```python
async def _generate_structured(self, prompt, input_data, response_model):
    schema = response_model.model_json_schema()
    payload = await self._chat_completion(
        model=self.model,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": f"{json.dumps(input_data)}\n\n"
                   f"Return only valid JSON that matches this JSON Schema:\n"
                   f"{json.dumps(schema)}"}],
        json_mode=False,  # <-- 去掉 JSON mode 约束
    )
```

prompt 中仍然保留 JSON Schema 指令（"Return only valid JSON that matches this JSON Schema"），LLM 依然知道要返回 JSON，只是不再由 API 层面强制约束。

**收益**：
- 所有 LLM 调用生成速度提升 20-40%（去掉 constrained decoding 开销）
- 中转站兼容性更好（不再依赖 `response_format` 参数）
- 每次调用节省 JSON Schema 的额外处理延迟

**风险**：低。prompt 指令 + 5 层容错管线 + 2 次重试已经覆盖。系统本身已设计了无 JSON mode 的 fallback 路径（第 208-213 行），说明开发者预期过这种场景。

---

### 优化 9：面向用户的 Agent 改为流式输出（降低首字延迟）

**文件**：
- `adapters/llm/openai_relay_adapter.py` — 新增 `_stream_chat_completion` 方法
- `adapters/llm/base.py` — 新增 `generate_streaming` 抽象方法
- `agents/general_answer_agent.py` — GeneralAnswerAgent 走流式路径
- `agents/research_qa_agent.py` — ResearchQAAgent 合成阶段走流式
- `agents/research_writer_agent.py` — ResearchWriterAgent 综述生成走流式
- `apps/api/routers/research.py` — SSE 新增 `token` 事件类型
- `apps/cli.py` — CLI 实时渲染 token

**现状**：所有 Agent 都用 `generate_structured` 返回完整 Pydantic 对象，用户必须等 LLM 生成 100% 完成 + JSON 解析 + 验证后才能看到第一个字。

**设计方案**：双模式共存——新增 `generate_streaming` 用于面向用户的文本生成，`generate_structured` 保持不变用于系统内部决策。

#### 第 1 层：LLM 适配器层

```python
# adapters/llm/base.py — 新增抽象方法
async def generate_streaming(
    self,
    prompt: str,
    input_data: dict[str, Any],
    on_token: Callable[[str], Awaitable[None]],
) -> str:
    """Stream LLM output token-by-token, return full text when done."""
    ...

# adapters/llm/openai_relay_adapter.py — 实现
async def _stream_chat_completion(self, payload: dict, on_token) -> str:
    payload["stream"] = True
    full_text = []
    async with self._client.stream(
        "POST", f"{self.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        json=payload,
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                full_text.append(token)
                await on_token(token)
    return "".join(full_text)
```

#### 第 2 层：改动的 3 个 Agent

| Agent | 改动点 | 原因 |
|---|---|---|
| **GeneralAnswerAgent** | `answer()` 新增流式分支，当上层传入 `on_token` 回调时走流式，否则保持原逻辑 | 用户问“你好”等简单问题时，全程都在等这一个 LLM 调用 |
| **ResearchQAAgent** | 合成阶段（`ReActFinalDraft.answer` 生成）走流式，ReAct 决策循环保持 `generate_structured` | 回答通常 500-2000 字，是用户等待最久的环节 |
| **ResearchWriterAgent** | 综述生成阶段走流式 | 综述文本最长，可达数千字 |

不改动的 Agent（输出是系统内部消费的结构化数据）：
- ResearchSupervisorAgent — 路由决策，必须完整结构化
- ResearchIntentResolver — 意图识别，系统内部
- LiteratureScoutAgent — 搜索查询规划，系统内部
- PreferenceMemoryAgent — 推荐源决策，系统内部
- ResearchQAAgent（ReAct 决策阶段）— 工具调用决策，系统内部

#### 第 3 层：SSE 传输层

```python
# apps/api/routers/research.py — event_stream 中新增 token 事件
async def on_progress(event: dict) -> None:
    if event.get("type") == "token":
        yield f"event: token\ndata: {json.dumps({'text': event['text']})}\n\n"
    else:
        yield f"event: progress\ndata: {json.dumps(event)}\n\n"
```

#### 第 4 层：CLI 渲染层

```python
# apps/cli.py — 实时打印 token而不是等完整响应
if event_type == "token":
    print(event["text"], end="", flush=True)
```

**收益**：
- 用户首字延迟从 3-6s（优化 1-8 后）降到 **~1-2s**
- 长回答（QA、综述）用户不再干等，边生成边阅读
- 覆盖 80%+ 用户直接感知的等待场景

**风险**：
- 中转站 SSE 兼容性：部分中转站 `stream=True` 实现有 bug（不发 `[DONE]`），需超时兆底
- JSON 片段截断：流式过程中断连时文本不完整，已有 `_repair_json_candidate` 可兆底
- 改动量约 200-300 行，涉及 7 个文件，需充分测试

---

## 三、优化效果预估

### 3.1 LLM 调用次数对比

| 请求类型 | 优化前 | 优化后 | 原因 |
|---|---|---|---|
| "你好" | 1 次 | 1 次 | 已有 guardrail（不变） |
| "帮我找 LLM agent 论文" | 3 次 | 1 次 | 阈值降低（0.78→跳过）+ guardrail 跳过 |
| "这篇论文的方法是什么" | 3 次 | 1 次 | 阈值降低（0.78→跳过）+ guardrail 跳过 |
| "导入论文1" | 2-3 次 | 1 次 | 阈值降低 + guardrail 跳过 |
| "对比第一篇和第二篇" | 2 次 | 1 次 | 已有高置信度（0.84→跳过） |
| 复杂多步研究 | 3+ 次 | 2 次 | supervisor 仍需 LLM，但 prompt 更小 |

### 3.2 单次 LLM 调用延迟对比

| 因素 | 优化前 | 优化后 |
|---|---|---|
| 连接建立 | 200-800ms/次 | ~0ms（复用） |
| JSON mode 开销 | +20-40% 生成时间 | 无（纯文本速度） |
| 意图识别 prompt | 3000-5000 token | ~500 token |
| supervisor prompt | 5000-15000 token | 1500-3000 token |
| connect timeout | 30s | 10s |
| 用户可见首字 | 等全部完成 | ~1-2s（流式） |

### 3.3 端到端响应时间预估

| 场景 | 优化前 | 优化后（完成） | 优化后（首字） |
|---|---|---|---|
| 简单问题（"你好"、闲聊） | 15-20s（或超时） | 3-6s | ~1-2s |
| 文献检索（"找论文"） | 20-40s | 6-12s | —（非流式场景） |
| 论文问答（"这篇论文讲了什么"） | 20-35s | 6-12s | ~2-4s |
| 论文导入 | 30-60s+ | 12-20s | —（非流式场景） |
| 文献综述 | 40-90s | 15-35s | ~3-5s |
| 复杂多步研究 | 40-90s | 15-35s | —（多步流程） |
| 中转站故障 | 等 60s 超时 | 10s 快速失败 + 重试 | — |

---

## 四、实施优先级

| 优先级 | 优化项 | 改动量 | 影响范围 |
|---|---|---|---|
| P0 | 优化 1：httpx 连接复用 | ~10 行 | 所有 LLM 调用 |
| P0 | 优化 2：connect timeout 缩短 | 1 行 | 所有 LLM 调用 |
| P0 | 优化 3：合并意图识别到 supervisor | ~30 行 | 所有请求减少 1 次 LLM 调用 |
| P1 | 优化 4：意图识别 prompt 裁剪 | 3 行 | 走 LLM 意图识别的请求 |
| P1 | 优化 5：supervisor state 裁剪 | ~30 行 | 走 supervisor LLM 的请求 |
| P1 | 优化 6：GeneralAnswerAgent 超时 | 1 行 | 通用问答 |
| P1 | 优化 8：去掉 JSON mode 约束 | 1 行 | 所有 LLM 调用，生成提速 20-40% |
| P2 | 优化 7：context_slice 截断 | ~15 行 | 有长会话历史的请求 |
| P2 | 优化 9：流式输出（GA/QA/Writer） | ~250 行 | 首字延迟降到 1-2s，覆盖 80%+ 用户可见场景 |

---

## 五、验证方法

优化后用以下命令回归测试：

```bash
conda run -n Research-Copilot python -m pytest tests/ -x -q
```

手动验证响应速度：

```bash
# 启动 CLI
conda run -n Research-Copilot python -m apps.cli

# 测试用例
1. "你好"                        → 预期 <10s
2. "帮我找 LLM agent 的论文"      → 预期 <15s
3. "这篇论文的方法是什么"          → 预期 <15s
4. "导入论文1"                    → 预期 <25s（含 PDF 下载）
```

---

## 六、不在本方案范围内的事项

以下优化虽然有效，但会改动系统架构，不在本方案范围内：

- 引入 LLM 响应缓存层（需要新增缓存组件）

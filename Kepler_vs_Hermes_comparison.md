# Kepler vs Hermes-Agent 架构对比分析

## 一、定位差异

| 维度 | **Kepler** (Research-Copilot) | **Hermes-Agent** |
|------|------|------|
| **领域** | 垂直科研助手（论文检索、QA、图表分析） | 通用个人 AI agent（编码、浏览器、消息、智能家居） |
| **架构风格** | 领域驱动 DDD + Pydantic schema 贯穿 | 单文件 agent loop + 自注册 tool 插件 |
| **代码规模** | ~160 个 .py（不含 web） | ~250+ 个 .py + TUI(TypeScript) |
| **运行时** | 全 async（FastAPI/MCP server） | 同步 agent loop（OpenAI chat.completions） |

---

## 二、意图识别与路由

### Kepler：显式多层 Skill 分类

Kepler 有两个专用路由层，都是 **heuristic-first + LLM fallback** 模式：

1. **`ResearchUserIntentResolverSkill`** (`skills/research/user_intent.py`, 403 行)
   - 定义了 11 种意图枚举：`literature_search`, `paper_import`, `sync_to_zotero`, `collection_qa`, `single_paper_qa`, `paper_comparison`, `figure_qa`, `document_understanding` 等
   - 用 **正则 marker 匹配**（中英文关键词表）做 heuristic 分类，输出 `confidence` + `rationale`
   - 支持 **序号引用消歧**：`"第一篇"` / `"p1"` → 从 candidate_papers 解析 paper_id
   - 如果有 `llm_adapter`，会把 heuristic 结果作为 `heuristic_hint` 传给 LLM 做 structured output 校正

2. **`ResearchQARoutingSkill`** (`skills/research/qa_routing.py`, 186 行)
   - 二级路由：`collection_qa` / `document_drilldown` / `chart_drilldown`
   - 同样 heuristic + LLM 双路径

**特点**：路由是 **结构化的 Pydantic model**，带 `confidence`/`rationale`/`source`/`needs_clarification`，可追溯、可测试。

### Hermes：无显式意图分类，LLM 原生路由

- **没有独立的意图识别层**。用户消息直接送入 `run_agent.py` 的 agent loop
- 路由完全依赖 **LLM 的 function calling 能力**：模型看到 tool schema 后自行决定调用哪个 tool
- Slash command（`/model`, `/skills`, `/background` 等）由 `hermes_cli/commands.py` 的 `CommandDef` 注册表做 **精确字符串匹配路由**，不涉及语义理解
- `clarify` tool 让 LLM 主动向用户提问，但这是 tool 级别的，不是路由层

**对比结论**：Kepler 有 **领域感知的 intent → route → skill** 三级显式管线；Hermes 把路由完全交给 LLM，靠精心设计的 tool schema description 来引导行为。

---

## 三、查询规划 (Query Planning)

### Kepler：专用查询改写 + 多源检索计划

`skills/research/query_planning.py` (411 行) 提供两个 Skill：

- **`ResearchQueryRewriteSkill`**：
  - 中文 → 英文学术查询改写（同义词展开、CJK 分词、领域术语标准化）
  - 内建 `_SYNONYM_RULES`（VLN→vision-and-language navigation, 大模型→LLM 等）
  - LLM 路径：生成 `simplified_topic` + `english_queries` + `local_queries`
  - Heuristic 路径：正则 strip + 同义词展开 + bigram

- **`TopicPlanningSkill`**：
  - 生成 `ResearchTopicPlan`：含 queries、days_back、max_papers、sources（arxiv/semantic_scholar/ieee/openalex/zotero）
  - 每个 source 有独立的 query 数量限制

### Hermes：无显式查询规划

- 没有 query rewrite/planning 模块
- `web_search` tool 直接接受自然语言 query，由 LLM 自行决定搜索词
- 如果需要多步搜索，完全依赖 LLM 在 agent loop 中多次调用 tool

**对比结论**：Kepler 的查询规划是 **学术领域深度定制**（多源 API 适配、中英改写、同义词）；Hermes 完全依赖 LLM + web_search tool 的通用能力。

---

## 四、Skill 系统

### Kepler：结构化 SkillSpec + Registry + 检索/记忆策略绑定

```
skills/
├── base.py          # SkillSpec(Pydantic): prompt_set, preferred_tools, retrieval_policy, memory_policy, output_style
├── registry.py      # SkillRegistry: register/select_skill_for_task/allowed_external_mcp_tools
├── loader.py        # 从配置加载 Skill
└── research/        # 14 个领域 Skill 实现
    ├── user_intent.py
    ├── qa_routing.py
    ├── query_planning.py
    ├── paper_analysis.py
    ├── paper_reading.py
    ├── paper_ranking.py
    ├── review_writing.py
    ├── survey_writing.py
    └── ...
```

每个 Skill 绑定：
- **preferred_tools**：控制该 Skill 下可用哪些 tool（比如 `paper_reading` 只用 `parse_document` + `answer_with_evidence`）
- **retrieval_policy**：`mode=hybrid/vector/graph`, `top_k`, `graph_query_mode`
- **memory_policy**：是否用 session/task/preference context
- **output_style**：语言、详细度、语气

`SkillRegistry.select_skill_for_task(task_type)` 自动选择最佳 Skill。

### Hermes：Markdown SKILL.md 文件 + 渐进式加载

```
~/.hermes/skills/
├── my-skill/
│   ├── SKILL.md           # YAML frontmatter + instructions
│   ├── references/        # 辅助文档
│   └── templates/         # 输出模板
```

- Skill 是 **Markdown 文档**，不是代码。包含 YAML frontmatter（name/description/platforms/prerequisites）
- **渐进式披露**（借鉴 Anthropic）：
  - Tier 1: `skills_list` 只返回 metadata（节省 token）
  - Tier 2: `skill_view` 加载完整内容
  - Tier 3: `skill_view("name", "references/api.md")` 按需加载关联文件
- Skill 内容作为 **user message 注入**（非 system prompt），保护 prompt cache
- `skill_manage` tool 让 agent 可以 **运行时创建/编辑 Skill**
- Skills Hub 支持在线搜索、安装社区 Skill

**对比结论**：

| | Kepler | Hermes |
|---|---|---|
| 表达形式 | Python 代码 + Pydantic schema | Markdown 文档 |
| 控制粒度 | 绑定 tool/retrieval/memory/output 策略 | 纯 prompt instructions |
| 运行时可编辑 | 否 | 是（agent 自主创建/修改） |
| 面向场景 | 科研任务类型分发 | 通用知识/流程文档 |

---

## 五、记忆系统

### Kepler：4 层记忆架构

| 层级 | 模块 | 存储 | 用途 |
|------|------|------|------|
| **WorkingMemory** | `memory/working_memory.py` | 内存 dict | 当前 session 的选中论文、任务计划、中间步骤（滑动窗口 10 turn） |
| **SessionMemory** | `memory/session_memory.py` | Pydantic model | session 级 QA 历史、结论、已读论文 |
| **LongTermMemory** | `memory/long_term_memory.py` | JSON 文件 / Qdrant 向量库 | 跨 session 的结论、总结，支持向量检索 |
| **UserProfileMemory** | `memory/user_profile_memory.py` | LongTermMemory 的子集 | 用户偏好主题、来源、关键词、推理风格 |

`MemoryManager` 是统一编排层：
- **hydrate_context**：session + working + long-term recall → 合并到 `ResearchContext`
- **promote_conclusion_to_long_term**：把 session 结论升级为持久记忆
- **observe_user_query**：被动学习用户偏好（主题频率、来源偏好）

关键特性：
- 向量检索：`deterministic_memory_vector()`（hash-based pseudo-embedding）或 Qdrant 真向量
- 记忆去重：基于 `(memory_type, content)` 去重
- Paper Knowledge Memory：论文级别的结构化知识存储

### Hermes：2 文件 + frozen snapshot

| 层级 | 存储 | 用途 |
|------|------|------|
| **MEMORY.md** | `~/.hermes/memories/MEMORY.md` | agent 的个人笔记（环境事实、项目约定、工具特性） |
| **USER.md** | `~/.hermes/memories/USER.md` | 用户画像（偏好、沟通风格、工作流习惯） |

核心设计：
- **Frozen snapshot 模式**：session 开始时读取快照注入 system prompt，之后不再更新 system prompt（保护 prefix cache）
- **单 `memory` tool**：add/replace/remove 三个 action，用子字符串匹配定位 entry
- **字符上限**：MEMORY.md 2200 chars，USER.md 1375 chars
- **安全扫描**：`_MEMORY_THREAT_PATTERNS` 检测 prompt injection / exfiltration
- **文件锁 + atomic write**：`fcntl.flock` + `tempfile` → `os.replace()`
- **MemoryManager** 支持 1 个 builtin provider + 至多 1 个外部 plugin provider（如 Honcho）
- **Context fencing**：`<memory-context>` 标签 + system note 防止 LLM 误读为用户指令

**对比结论**：

| | Kepler | Hermes |
|---|---|---|
| 存储层数 | 4 层（working → session → long-term → user profile） | 2 文件 flat store |
| 检索 | 向量相似度 + 词法匹配 | 无检索（全量注入 system prompt） |
| 容量 | Qdrant 5000+ records | ~3.5KB total |
| 结构化程度 | Pydantic schema（QAPair, TaskStep, SubManagerState） | 纯文本 entries，§ 分隔 |
| Agent 自主写入 | 仅通过编排层 | tool 直接 CRUD |
| 缓存策略 | 无特殊处理 | frozen snapshot 保护 prefix cache |

---

## 六、MCP (Model Context Protocol)

### Kepler：双向 MCP（client + server）

```
mcp/
├── client/
│   ├── base.py           # BaseMCPClient protocol
│   └── registry.py       # MCPClientRegistry: 发现 tools/prompts/resources，call_tool
├── server/
│   ├── app.py            # MCP server 入口
│   ├── tool_adapter.py   # 内部 ToolSpec → MCP tool 暴露
│   ├── prompt_adapter.py # 内部 prompt → MCP prompt
│   └── resource_adapter.py
├── mapping.py            # ToolSpec ↔ MCPToolSpec 转换
└── schemas.py            # MCPToolSpec, MCPToolCallResult
```

- **作为 MCP server**：把内部 tool/prompt/resource 通过 MCP 协议暴露给外部 client
- **作为 MCP client**：连接外部 MCP server（如 Zotero、学术搜索），发现并调用其 tool
- Skill 可以通过 `allowed_external_mcp_tools` 控制哪些外部 MCP tool 可用

### Hermes：纯 MCP client（生产级 2663 行）

`tools/mcp_tool.py` 是一个 **重量级 MCP 客户端**：
- **传输**：stdio（command+args）+ HTTP/StreamableHTTP（url）
- **架构**：独立后台事件循环 `_mcp_loop` + daemon thread，`run_coroutine_threadsafe()` 桥接
- **可靠性**：指数退避自动重连（5 次重试）、per-server 超时（tool call 120s / connect 60s）
- **安全**：
  - `_build_safe_env()`：只传递白名单环境变量给 stdio 子进程
  - `_sanitize_error()`：正则剥离 credential（ghp_、sk-、Bearer、token=）
  - `_MCP_INJECTION_PATTERNS`：扫描 tool description 中的 prompt injection
- **Sampling 支持**：MCP server 可以通过 `sampling/createMessage` 请求 LLM 推理
- **动态工具发现**：`tools/list_changed` notification → 自动刷新 tool schema
- **OAuth 支持**：`mcp_oauth.py` + `mcp_oauth_manager.py`

另外 Hermes 也可以 **作为 MCP server** 暴露自身能力（`mcp_serve.py`）。

**对比结论**：

| | Kepler | Hermes |
|---|---|---|
| Client | 轻量 async protocol（99 行） | 生产级实现（2663 行），自动重连、安全加固 |
| Server | 有，暴露内部能力 | 有（`mcp_serve.py`） |
| 传输 | 未明确 | stdio + HTTP/StreamableHTTP |
| 安全 | 基本 | 环境变量过滤、credential 剥离、injection 扫描 |
| Sampling | 无 | 支持 server → agent LLM 请求 |
| 配置 | 代码级注册 | `~/.hermes/config.yaml` 声明式 |

---

## 七、Tool 系统

### Kepler：Pydantic schema + ToolExecutor + 审计追踪

```
tooling/
├── registry.py               # ToolRegistry: register/filter/as_openai_functions/as_mcp_tools
├── executor.py               # ToolExecutor: validate → execute → retry → serialize → trace
├── schemas.py                # ToolSpec, ToolCall, ToolExecutionResult, ToolCallTrace
├── serializers.py            # tool_specs_to_openai_functions()
├── research_function_specs.py # 领域 tool 定义
└── research_runtime_tool_specs.py
```

```
tools/
├── answer_toolkit.py          # answer_with_evidence
├── chart_toolkit.py           # understand_chart
├── document_toolkit.py        # parse_document
├── retrieval_toolkit.py       # hybrid_retrieve, query_graph_summary
├── research/                  # arxiv_search, semantic_scholar, ieee, openalex
└── ...
```

- **ToolSpec**：Pydantic model，含 `input_schema`/`output_schema`/`handler`/`tags`/`category`/`max_retries`/`audit_metadata`
- **ToolExecutor**：输入验证 → handler 调用（支持 async） → 输出序列化 → 重试 → `ToolCallTrace` 审计
- **Skill 过滤**：`filter_tools(skill_context=...)` 根据当前 Skill 的 `preferred_tools` 限制可用 tool
- 工具偏向 **学术/RAG 领域**：检索、图表理解、证据链、知识图谱

### Hermes：自注册 singleton + handler lambda + toolset 分组

```
tools/
├── registry.py         # ToolRegistry singleton + ToolEntry + discover_builtin_tools()
├── file_tools.py       # read_file, write_file, patch, search_files
├── terminal_tool.py    # terminal（多 backend: local/docker/ssh/modal/daytona/singularity）
├── web_tools.py        # web_search, web_extract
├── browser_tool.py     # 10+ browser 操作
├── mcp_tool.py         # MCP client
├── memory_tool.py      # memory CRUD
├── skills_tool.py      # skill 管理
├── delegate_tool.py    # 子 agent 委派
├── code_execution_tool.py  # Python sandbox
├── vision_tools.py     # 图像分析
├── ...
```

- **自注册模式**：每个 tool 文件在 `import` 时调用 `registry.register(name, toolset, schema, handler, check_fn, emoji)`
- **discover_builtin_tools()**：AST 扫描 `tools/` 目录，自动 import 含 `registry.register()` 的模块
- **Toolset 系统** (`toolsets.py`)：按平台/场景分组（hermes-cli, hermes-telegram, hermes-discord, browser, debugging, safe, hermes-acp），支持组合继承
- **check_fn**：运行时检查 tool 可用性（如 `GITHUB_TOKEN` 是否存在）
- **thread-safe**：`threading.RLock` 保护注册表（MCP 动态刷新安全）
- 工具偏向 **通用 agent 操作**：文件、终端、浏览器、消息、TTS、cronjob、HomeAssistant

**对比结论**：

| | Kepler | Hermes |
|---|---|---|
| 注册方式 | 显式 `ToolSpec` + `register_many()` | 模块级 `registry.register()` 自注册 |
| Schema | Pydantic `input_schema`/`output_schema` | OpenAI function-calling dict |
| 执行器 | `ToolExecutor` with retry + trace | `handle_function_call()` in `model_tools.py` |
| 分组 | Skill `preferred_tools` 过滤 | Toolset dict（按平台/场景） |
| 审计 | `ToolCallTrace` with latency, attempts, correlation_id | 无结构化审计 |
| 工具范围 | ~15 个学术/RAG 工具 | ~50+ 通用工具（file/terminal/browser/web/tts/HA/...） |

---

## 八、上下文工程

### Kepler：无显式压缩，靠 retrieval 控制上下文

- **没有独立的 context compressor 模块**
- 上下文管理依赖：
  - `WorkingMemory` 滑动窗口（`max_turns=10`）
  - `SessionMemory` history 裁剪（`max_history_turns`）
  - `SkillMemoryPolicy` 控制是否注入 session/task/preference context
  - Retrieval `top_k` 控制检索结果数量
- 推理策略模块 (`reasoning/`)：
  - **ReAct**（596 行）：Thought → Action → Observation 循环，最多 12 步
  - **Plan-and-Solve**（124 行）：先规划再执行
  - **CoT**：chain-of-thought prompting
- 没有 prompt caching 实现

### Hermes：3 层压缩 + prompt caching + trajectory compressor

#### 1. Context Compressor (`agent/context_compressor.py`, 1276 行)

实时对话压缩，保护 head + tail，用 **auxiliary LLM** 压缩 middle：
- **Tool output pruning**：清理老旧 tool 输出（cheap pre-pass）
- **Token-budget tail protection**：不是固定 N 条消息，而是按 token 预算保护尾部
- **LLM summarization**：结构化摘要模板，含 Resolved/Pending/Active Task/Remaining Work
- **Iterative summary**：多次压缩时保留之前摘要的信息
- **CONTEXT COMPACTION prefix**：明确告知模型这是"前任 context window 的交接"，不要重复执行

#### 2. Prompt Caching (`agent/prompt_caching.py`, 73 行)

Anthropic `system_and_3` 策略：
- 4 个 `cache_control` breakpoint：system prompt + 最后 3 条非 system 消息
- 纯函数实现，deep copy messages 后注入 `{"type": "ephemeral"}`
- 支持 native Anthropic 和 OpenRouter 两种格式

#### 3. Trajectory Compressor (`trajectory_compressor.py`, 1509 行)

**离线**训练数据压缩（非实时）：
- 用真实 tokenizer（如 `moonshotai/Kimi-K2-Thinking`）计算 token
- 保护 first system/human/gpt/tool + last N turns，只压缩 middle
- 用 LLM summarization API 生成摘要替换被压缩区域
- 批量处理 JSONL 轨迹文件，生成详细压缩 metrics

#### 4. Context Engine (`agent/context_engine.py`)

上下文引用追踪和管理。

#### 5. Model Metadata (`agent/model_metadata.py`, 1220 行)

模型上下文窗口大小表 + token 粗估（`estimate_messages_tokens_rough`）。

**对比结论**：

| | Kepler | Hermes |
|---|---|---|
| 实时压缩 | 无（靠滑动窗口 + retrieval top_k） | 3 层（tool pruning → tail budget → LLM summary） |
| Prompt caching | 无 | Anthropic system_and_3（4 breakpoints） |
| Token 计算 | 无 | 粗估 + models.dev 查询 |
| 离线压缩 | 无 | trajectory_compressor（训练数据专用） |
| 推理策略 | ReAct / Plan-and-Solve / CoT | 纯 agent loop（无显式推理框架） |

---

## 九、总结与互补性

### Kepler 的优势（可借鉴给 Hermes）
1. **显式意图分类 + QA 路由**：减少 LLM 误判，提高可控性
2. **结构化 Skill → Tool 映射**：按任务类型限制 tool scope，减少 token 和误调用
3. **4 层记忆 + 向量检索**：跨 session 知识积累与召回
4. **ReAct 推理框架**：结构化的 Thought-Action-Observation 循环
5. **ToolExecutor 审计链**：每次 tool call 有完整 trace（latency、attempts、correlation_id）

### Hermes 的优势（可借鉴给 Kepler）
1. **生产级 MCP 客户端**：自动重连、credential 安全、sampling、OAuth
2. **实时 context compression**：3 层压缩保障长对话不溢出
3. **Prompt caching**：75% input token 成本节省
4. **Frozen snapshot 记忆模式**：简单但高效，完美保护 prefix cache
5. **Toolset 分组 + 平台适配**：同一 tool 注册表适配 CLI/Telegram/Discord/Slack/API 等 15+ 平台
6. **自注册发现机制**：AST 扫描自动加载 tool，零配置

### 融合建议
- Kepler 可引入 Hermes 的 **context compressor** 和 **prompt caching**，解决长 session QA 的 token 溢出
- Hermes 可引入 Kepler 的 **intent classification** 层，在 function calling 前做轻量预分类，减少大模型 tool 选择错误
- 两者的 **Skill 系统可互补**：Kepler 的代码级 SkillSpec 控制 retrieval/tool 策略，Hermes 的 Markdown Skill 提供 agent 自主编辑能力

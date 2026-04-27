# Kepler 向 Hermes 学习的改进方案

> 基于 Kepler（Research-Copilot）与 Hermes-Agent 的深度对比分析，
> 提取 Hermes 中经过生产验证的模式，适配到 Kepler 的领域架构中。

---

## 改进总览

| # | 改进项 | 优先级 | 预估工作量 | 核心收益 |
|---|--------|--------|-----------|---------|
| 1 | 上下文压缩引擎 | **P0** | 3-4 天 | 长对话不溢出，token 成本降 40-60% |
| 2 | Prompt Caching | **P0** | 0.5 天 | Anthropic 模型 input token 成本降 75% |
| 3 | Token 预算管理 | **P0** | 1 天 | 动态感知剩余空间，自适应裁剪 |
| 4 | 记忆安全加固 | **P1** | 1 天 | 防 prompt injection 进入 long-term memory |
| 5 | MCP 客户端升级 | **P1** | 2-3 天 | 自动重连、credential 安全、sampling |
| 6 | Tool 自注册 + 发现 | **P2** | 1-2 天 | 零配置加载新 tool，减少维护 |
| 7 | Frozen Snapshot 记忆 | **P2** | 1 天 | 保护 prompt cache 前缀 |
| 8 | Agent-Editable Skills | **P3** | 2 天 | agent 自主积累领域知识 |
| 9 | Toolset 平台适配 | **P3** | 1 天 | 同一注册表适配 CLI/API/MCP |
| 10 | Dangerous Command 审批 | **P3** | 0.5 天 | 外部 tool 安全守门 |
| 11 | Error Classification & Smart Recovery | **P1** | 1 天 | 结构化错误分类 → 精准恢复策略 |
| 12 | Provider Failover Chain | **P1** | 1.5 天 | 主 provider 挂掉自动切备用，科研不中断 |
| 13 | Sensitive Data Redaction | **P1** | 0.5 天 | 压缩/记忆/日志中密钥不泄露 |
| 14 | Anti-Thrashing Compression Guard | **P0** | 随改进 1 | 防止低效压缩死循环 + 保护用户最新请求 |
| 15 | Context Window Probing | **P2** | 0.5 天 | 运行时探测实际 context 上限，自动降级 |

---

## 改进 1：上下文压缩引擎（P0）

### 问题

Kepler 当前靠 `WorkingMemory.max_turns=10` 滑动窗口和 `max_history_turns` 裁剪控制上下文长度。
对于深度科研对话（多轮 QA + 大量检索结果 + 图表分析），很容易超出模型上下文窗口，
且粗暴截断会丢失关键的中间推理步骤和证据。

### 借鉴 Hermes

Hermes 的 `agent/context_compressor.py`（1276 行）实现了 3 层压缩：
1. **Tool output pruning**：清理老旧 tool 返回（最便宜的 pre-pass）
2. **Token-budget tail protection**：按 token 预算保护尾部（非固定 N 条）
3. **LLM summarization**：用 auxiliary model 生成结构化摘要替换中间 turns

关键设计：
- `CONTEXT COMPACTION` prefix 明确告知模型这是"前任 context 的交接"
- 迭代压缩：多次压缩时保留之前摘要信息
- Summary 模板含 Resolved/Pending/Active Task/Remaining Work

### 实现方案

新建 `context/compressor.py`：

```python
# context/compressor.py
class ContextCompressor:
    """3-layer context compression for Kepler conversations."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        target_budget_ratio: float = 0.75,  # 目标占模型 context 的比例
    ): ...

    def compress_messages(
        self,
        messages: list[dict],
        model_context_length: int,
        *,
        protected_tool_names: set[str] | None = None,
    ) -> list[dict]:
        """
        Layer 1: prune old tool results (replace with one-line summary)
        Layer 2: snip middle messages if still over budget
        Layer 3: LLM summarize compressed region (if llm_adapter available)
        """
        budget = int(model_context_length * self.target_budget_ratio)
        current_tokens = self._estimate_tokens(messages)
        if current_tokens <= budget:
            return messages

        # Layer 1: tool output pruning
        messages = self._prune_tool_outputs(messages, protected_tool_names or set())
        if self._estimate_tokens(messages) <= budget:
            return messages

        # Layer 2: identify compressible region (protect head + tail)
        head, middle, tail = self._split_protected(messages, tail_budget=budget // 3)
        if not middle:
            return messages

        # Layer 3: LLM summarize
        if self.llm_adapter:
            summary = await self._summarize_region(middle)
            return head + [self._make_summary_message(summary)] + tail
        else:
            # Fallback: keep only head + tail
            return head + [self._make_snip_notice(len(middle))] + tail
```

与 Kepler 的集成点：
- `MemoryManager.hydrate_context()` 之后、送入 LLM 之前调用
- `protected_tool_names` 可从当前 `SkillSpec.preferred_tools` 获取
- summary prompt 放到 `prompts/context/compress_summary.txt`，复用 Kepler 的 prompt resolver

### 文件变更

| 文件 | 操作 |
|------|------|
| `context/__init__.py` | 新建 |
| `context/compressor.py` | 新建，~250 行 |
| `context/token_counter.py` | 新建，~60 行 |
| `prompts/context/compress_summary.txt` | 新建 |
| `reasoning/react.py` | 在 `reason()` 入口处插入压缩调用 |

---

## 改进 2：Prompt Caching（P0）

### 问题

Kepler 每轮对话都完整发送 system prompt + 全部 history，
使用 Anthropic 模型时浪费大量 input token 费用。

### 借鉴 Hermes

`agent/prompt_caching.py`（73 行）实现了 `system_and_3` 策略：
- 4 个 `cache_control` breakpoint：system prompt + 最后 3 条非 system 消息
- 纯函数、零状态、deep copy 安全

### 实现方案

```python
# context/prompt_caching.py
import copy
from typing import Any

def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    cache_ttl: str = "5m",
) -> list[dict[str, Any]]:
    """Place up to 4 cache_control breakpoints: system + last 3 non-system."""
    messages = copy.deepcopy(messages)
    marker = {"type": "ephemeral"}
    breakpoints = 0

    if messages and messages[0].get("role") == "system":
        _mark(messages[0], marker)
        breakpoints += 1

    remaining = 4 - breakpoints
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _mark(messages[idx], marker)

    return messages

def _mark(msg: dict, marker: dict) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": marker}]
    elif isinstance(content, list) and content:
        content[-1]["cache_control"] = marker
    else:
        msg["cache_control"] = marker
```

集成点：在 LLM adapter 层，当 `provider == "anthropic"` 时自动应用。

### 文件变更

| 文件 | 操作 |
|------|------|
| `context/prompt_caching.py` | 新建，~50 行 |
| `adapters/llm/openai_adapter.py` | 在 `_build_messages()` 后判断是否 Anthropic，调用 caching |

---

## 改进 3：Token 预算管理（P0）

### 问题

Kepler 没有模型 context length 的全局感知，
各模块（history、retrieval、evidence）独立裁剪，可能累加后仍然超限。

### 借鉴 Hermes

`agent/model_metadata.py`（1220 行）维护了所有主流模型的 context length 表，
并提供 `estimate_messages_tokens_rough()` 粗估函数。
`context_compressor.py` 在每轮开始时计算总 token 并决定是否触发压缩。

### 实现方案

```python
# context/token_counter.py
MODEL_CONTEXT_LENGTHS = {
    "qwen-plus": 131072,
    "qwen-turbo": 131072,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "gemini-2.0-flash": 1048576,
    "deepseek-chat": 65536,
    # ... 按需补充
}

def get_context_length(model: str) -> int:
    return MODEL_CONTEXT_LENGTHS.get(model, 32768)

def estimate_tokens_rough(messages: list[dict]) -> int:
    """Fast char/4 estimation, no external dependency."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // 4 + len(messages) * 4  # per-message overhead

class TokenBudget:
    """Tracks remaining token budget across a single agent turn."""

    def __init__(self, model: str, reserve_for_output: int = 4096):
        self.total = get_context_length(model)
        self.reserved_output = reserve_for_output
        self.available = self.total - reserve_for_output
        self.used = 0

    def consume(self, tokens: int, label: str = "") -> int:
        self.used += tokens
        return self.remaining

    @property
    def remaining(self) -> int:
        return max(0, self.available - self.used)

    def should_compress(self, threshold: float = 0.85) -> bool:
        return self.used / self.available > threshold
```

集成点：
- `Settings` 新增 `llm_context_length: int | None = None`（手动覆盖）
- `MemoryManager.hydrate_context()` 接受 `TokenBudget` 参数，按剩余预算裁剪 recall 数量
- `ReActReasoningAgent.reason()` 每步后检查 `budget.should_compress()`

### 文件变更

| 文件 | 操作 |
|------|------|
| `context/token_counter.py` | 新建，~80 行 |
| `core/config.py` | 新增 `llm_context_length` 字段 |

---

## 改进 4：记忆安全加固（P1）

### 问题

Kepler 的 `LongTermMemory` 接受任意 content 写入（`upsert`），
如果 agent 或外部 MCP tool 写入恶意内容（prompt injection payload），
这些内容会在后续 session 通过 `search()` 召回并注入到 prompt 中。

### 借鉴 Hermes

`tools/memory_tool.py` 实现了两层防护：
1. **Threat pattern scanning**：11 个正则匹配 prompt injection / exfiltration
2. **Invisible unicode detection**：检测 `\u200b`, `\u202e` 等不可见字符

### 实现方案

```python
# memory/security.py
import re
from typing import Optional

_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules)', "disregard_rules"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD)', "exfil_curl"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc)', "read_secrets"),
]

_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}

def scan_memory_content(content: str) -> Optional[str]:
    """Return error string if content is unsafe for memory storage."""
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: invisible unicode U+{ord(char):04X}"
    for pattern, pid in _THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: threat pattern '{pid}'"
    return None
```

集成点：
- `LongTermMemory.upsert()` 在写入前调用 `scan_memory_content(record.content)`
- `MemoryManager.promote_conclusion_to_long_term()` 同样调用
- 扫描失败时 log warning + 拒绝写入

### 文件变更

| 文件 | 操作 |
|------|------|
| `memory/security.py` | 新建，~50 行 |
| `memory/long_term_memory.py` | `upsert()` 加 `scan_memory_content()` 前置检查 |
| `memory/memory_manager.py` | `promote_conclusion_to_long_term()` 加安全检查 |

---

## 改进 5：MCP 客户端升级（P1）

### 问题

Kepler 的 `BaseMCPClient` 是一个 87 行的 abstract protocol，
只有 `InProcessMCPClient` 一个实现（进程内直连）。
缺少：
- 外部进程 MCP server 连接（stdio/HTTP）
- 自动重连 / 超时控制
- Credential 安全（环境变量过滤、错误信息脱敏）
- 连接池管理

### 借鉴 Hermes

`tools/mcp_tool.py`（2663 行）的核心设计：
- 独立 background event loop + daemon thread
- 自动重连（指数退避，5 次重试）
- `_build_safe_env()`：白名单环境变量
- `_sanitize_error()`：正则剥离 credential
- `_MCP_INJECTION_PATTERNS`：扫描 tool description

### 实现方案

```python
# mcp/client/stdio_client.py
class StdioMCPClient(BaseMCPClient):
    """Connect to external MCP servers via stdio transport."""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        tool_timeout: int = 120,
        connect_timeout: int = 60,
        max_retries: int = 5,
    ): ...

    async def connect(self) -> None:
        """Start subprocess, establish MCP session, discover tools."""
        safe_env = _build_safe_env(self._user_env)
        # ... mcp.client.stdio.stdio_client(...)

    async def call_tool(self, tool_name, arguments, call_id) -> MCPToolCallResult:
        """Call with timeout + auto-reconnect on failure."""
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._session.call_tool(tool_name, arguments),
                    timeout=self._tool_timeout,
                )
            except (ConnectionError, asyncio.TimeoutError):
                if attempt < self._max_retries:
                    await self._reconnect(attempt)
                else:
                    raise

# mcp/client/http_client.py
class HttpMCPClient(BaseMCPClient):
    """Connect to external MCP servers via HTTP/StreamableHTTP."""
    ...

# mcp/security.py
SAFE_ENV_KEYS = frozenset({"PATH", "HOME", "USER", "LANG", "TERM", "SHELL", "TMPDIR"})
CREDENTIAL_PATTERN = re.compile(r"(?:ghp_\S+|sk-\S+|Bearer\s+\S+|token=\S+)", re.I)

def build_safe_env(user_env: dict | None) -> dict: ...
def sanitize_error(text: str) -> str: ...
def scan_tool_description(desc: str) -> list[str]: ...  # injection warnings
```

配置方式（`core/config.py`）：
```python
# Settings 新增
mcp_servers: dict[str, dict] = {}
# 格式同 Hermes: {name: {command, args, env, timeout, ...}}
```

### 文件变更

| 文件 | 操作 |
|------|------|
| `mcp/client/stdio_client.py` | 新建，~200 行 |
| `mcp/client/http_client.py` | 新建，~150 行 |
| `mcp/security.py` | 新建，~80 行 |
| `mcp/client/registry.py` | 扩展 `register_from_config()` 方法 |
| `core/config.py` | 新增 `mcp_servers` 字段 |
| `pyproject.toml` | 新增 optional dep: `mcp>=1.0` |

---

## 改进 6：Tool 自注册 + 发现（P2）

### 问题

Kepler 的 tool 注册是 **手动调用** `registry.register(tool_spec)` 或 `register_many()`，
在 `research_function_specs.py` / `research_runtime_tool_specs.py` 中集中构造。
新增 tool 需要修改多个文件。

### 借鉴 Hermes

Hermes 的 `discover_builtin_tools()`：
1. AST 扫描 `tools/` 目录下所有 `.py` 文件
2. 检测是否包含 `registry.register()` 顶层调用
3. 自动 import → 触发自注册

### 实现方案

```python
# tooling/discovery.py
import ast
import importlib
from pathlib import Path

def _module_registers_tools(path: Path) -> bool:
    """Check if module has top-level registry.register() call."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "register"
        for stmt in tree.body
    )

def discover_tools(tools_dir: Path | None = None) -> list[str]:
    """Auto-import self-registering tool modules."""
    tools_path = tools_dir or Path(__file__).parent.parent / "tools"
    imported = []
    for path in sorted(tools_path.glob("**/*.py")):
        if path.name.startswith("_") or not _module_registers_tools(path):
            continue
        module_name = str(path.relative_to(tools_path.parent)).replace("/", ".").removesuffix(".py")
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as e:
            logger.warning("Failed to import tool module %s: %s", module_name, e)
    return imported
```

迁移策略：
1. 现有 `research_function_specs.py` 中的 tool 声明改为在各 tool 文件内自注册
2. 保留 `register_many()` 作为兼容接口
3. 启动时调用 `discover_tools()` 自动加载

### 文件变更

| 文件 | 操作 |
|------|------|
| `tooling/discovery.py` | 新建，~60 行 |
| `tools/*.py` | 各 tool 文件末尾添加 `registry.register(...)` |
| 启动入口 | 调用 `discover_tools()` |

---

## 改进 7：Frozen Snapshot 记忆模式（P2）

### 问题

Kepler 的 `hydrate_context()` 每轮都从 long-term memory 检索并注入 prompt。
如果使用 Anthropic prompt caching，每轮 system prompt 变化会导致 cache miss。

### 借鉴 Hermes

Hermes `MemoryStore` 在 session 开始时做一次 `load_from_disk()`，
捕获 `_system_prompt_snapshot`。整个 session 内 system prompt 注入的内容不变。
mid-session 的 memory 写入只更新磁盘文件，不影响 prompt → 前缀缓存命中率 100%。

### 实现方案

```python
# memory/memory_manager.py 新增
class MemoryManager:
    def __init__(self, ...):
        ...
        self._session_snapshots: dict[str, dict] = {}  # session_id → frozen data

    def freeze_session_snapshot(self, session_id: str, context: ResearchContext) -> None:
        """Capture a frozen memory snapshot for prompt injection.

        Called once at session start. Subsequent hydrate_context() calls
        will use this snapshot for the system prompt portion, while
        live context is still available in working memory for tool use.
        """
        recall = self.long_term_memory.search(
            LongTermMemoryQuery(
                query=context.research_topic or "",
                keywords=context.research_goals[:5],
                top_k=3,
            )
        )
        self._session_snapshots[session_id] = {
            "recalled_memories": [r.model_dump(mode="json") for r in recall.records],
            "user_profile": self.load_user_profile().model_dump(mode="json"),
        }

    def get_frozen_prompt_block(self, session_id: str) -> dict | None:
        """Return the frozen snapshot for system prompt injection."""
        return self._session_snapshots.get(session_id)
```

集成点：session 初始化时调用 `freeze_session_snapshot()`，构建 system prompt 时用 frozen 数据。

### 文件变更

| 文件 | 操作 |
|------|------|
| `memory/memory_manager.py` | 新增 `freeze_session_snapshot()` + `get_frozen_prompt_block()` |
| session 初始化逻辑 | 在 `hydrate_context()` 首次调用时自动 freeze |

---

## 改进 8：Agent-Editable Skills（P3）

### 问题

Kepler 的 Skill 是 Python 代码（`SkillSpec`），只能开发者修改。
Agent 在科研过程中学到的领域特定知识无法自主保存为可复用 Skill。

### 借鉴 Hermes

Hermes Skill 是 Markdown 文档 + YAML frontmatter，agent 通过 `skill_manage` tool 直接 CRUD。

### 实现方案

在 Kepler 现有的 **代码 Skill 体系不变**的前提下，增加一层 **知识 Skill**：

```
~/.kepler/skills/           # 或 .data/skills/
├── molecular-dynamics/
│   ├── SKILL.md            # agent 编写的领域知识
│   └── references/
│       └── force-fields.md
└── meta-analysis/
    └── SKILL.md
```

新增 3 个 tool：
- `skills_list`：列出所有知识 Skill 的 metadata
- `skill_view`：加载完整内容（渐进式披露）
- `skill_manage`：create / update / delete

知识 Skill 作为 **额外 context** 注入到 prompt 中（不替换 `SkillSpec`），
类似 Hermes 的 user message 注入方式。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tools/skill_tools.py` | 新建，~300 行 |
| `skills/knowledge_loader.py` | 新建，~100 行（frontmatter 解析 + 渐进式加载） |
| `prompts/skills/inject_knowledge.txt` | 新建 |
| `core/config.py` | 新增 `skills_dir` 配置 |

---

## 改进 9：Toolset 平台适配（P3）

### 问题

Kepler 的 `ToolRegistry.filter_tools()` 只支持 tags / names / skill_context 过滤。
如果未来要支持多种前端（CLI、Web、MCP server、API），
不同前端可用的 tool 集合可能不同（比如 CLI 不需要 `understand_chart`）。

### 借鉴 Hermes

Hermes 的 `toolsets.py` 定义了 30+ toolset：
- 按能力分组（web, browser, file, terminal, vision）
- 按平台分组（hermes-cli, hermes-telegram, hermes-api-server）
- 支持组合继承（`includes` 引用其他 toolset）

### 实现方案

```python
# tooling/toolsets.py
TOOLSETS = {
    "research-core": {
        "description": "Core research tools",
        "tools": ["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
    },
    "document": {
        "description": "Document processing tools",
        "tools": ["parse_document", "understand_chart"],
    },
    "academic-search": {
        "description": "Academic search APIs",
        "tools": ["arxiv_search", "semantic_scholar_search", "openalex_search"],
    },
    "kepler-cli": {
        "description": "Full CLI toolset",
        "tools": [],
        "includes": ["research-core", "document", "academic-search"],
    },
    "kepler-api": {
        "description": "API server toolset (no vision)",
        "tools": [],
        "includes": ["research-core", "academic-search"],
    },
}

def resolve_toolset(name: str) -> list[str]:
    """Recursively resolve toolset to flat tool name list."""
    ...
```

`ToolRegistry.filter_tools()` 新增 `toolset: str | None` 参数。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tooling/toolsets.py` | 新建，~100 行 |
| `tooling/registry.py` | `filter_tools()` 新增 `toolset` 参数 |

---

## 改进 10：Dangerous Command 审批（P3）

### 问题

Kepler 通过 MCP 或外部 tool 可能执行有副作用的操作（写入 Zotero、删除文档等），
目前没有安全审批机制。

### 借鉴 Hermes

`tools/approval.py` 实现了 dangerous command detection：
- 正则匹配危险命令模式（rm -rf, sudo, chmod, etc.）
- 执行前回调用户确认
- 支持 auto-approve 白名单

### 实现方案

```python
# tooling/approval.py
from typing import Callable, Awaitable

DANGEROUS_PATTERNS = [
    r'\brm\s+(-[rf]+\s+|.*\s+/).*',
    r'\bsudo\b',
    r'\bchmod\s+777\b',
    r'\bdrop\s+table\b',
    r'\bdelete\s+from\b',
    r'\btruncate\b',
]

class ApprovalGate:
    def __init__(
        self,
        callback: Callable[[str, str, dict], Awaitable[bool]] | None = None,
        auto_approve_tools: set[str] | None = None,
    ): ...

    async def check(self, tool_name: str, tool_input: dict) -> bool:
        """Return True if approved, False if rejected."""
        if tool_name in (self.auto_approve_tools or set()):
            return True
        if self._is_dangerous(tool_input):
            if self.callback:
                return await self.callback(tool_name, "Dangerous operation detected", tool_input)
            return False  # default reject without callback
        return True
```

集成到 `ToolExecutor.execute_tool_call()` 在执行前调用 `approval_gate.check()`。

### 文件变更

| 文件 | 操作 |
|------|------|
| `tooling/approval.py` | 新建，~80 行 |
| `tooling/executor.py` | 执行前插入 `approval_gate.check()` |

---

## 改进 11：Error Classification & Smart Recovery（P1）

### 问题

Kepler 的 `BaseLLMAdapter._run_with_retries()` 采用简单的 circuit breaker + 全量 retry 策略：
- 只区分"应该打开熔断"和"应该重试"两种情况
- 不区分 rate-limit vs billing vs auth vs context-overflow 等不同错误类型
- 没有 context-overflow 时自动压缩的恢复路径
- 没有 rate-limit 后的 credential rotation 或 provider fallback

当 API 返回 429 时，Kepler 只是指数退避重试，浪费宝贵的 timeout 窗口。

### 借鉴 Hermes

`agent/error_classifier.py`（183 行）实现了结构化错误分类：

```python
class FailoverReason(Enum):
    rate_limit = "rate_limit"
    billing = "billing"
    auth = "auth"
    context_overflow = "context_overflow"
    thinking_signature = "thinking_signature"
    long_context_tier = "long_context_tier"
    server_error = "server_error"
    timeout = "timeout"
    connection = "connection"
    content_filter = "content_filter"
    unknown = "unknown"

@dataclass
class ClassifiedError:
    reason: FailoverReason
    status_code: int | None
    retryable: bool
    should_compress: bool       # context_overflow → 触发压缩
    should_rotate_credential: bool  # rate_limit/billing → 换 key
    should_fallback: bool       # 持续失败 → 换 provider
```

每次 API 错误后，`classify_api_error()` 基于 HTTP status + error body + 模型 context length 判断失败原因，
`run_conversation()` 根据 `ClassifiedError` 的 flag 组合决定恢复策略（压缩/换 key/换 provider/abort）。

### 实现方案

```python
# adapters/llm/error_classifier.py
from enum import Enum
from dataclasses import dataclass

class FailureReason(Enum):
    rate_limit = "rate_limit"
    billing = "billing"
    auth = "auth"
    context_overflow = "context_overflow"
    content_filter = "content_filter"
    server_error = "server_error"
    timeout = "timeout"
    connection = "connection"
    unknown = "unknown"

@dataclass
class ClassifiedError:
    reason: FailureReason
    status_code: int | None
    retryable: bool
    should_compress: bool
    should_fallback: bool

def classify_llm_error(
    exc: Exception,
    *,
    provider: str = "",
    approx_tokens: int = 0,
    context_length: int = 0,
) -> ClassifiedError:
    """Classify an LLM API error into a structured recovery hint."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()

    # Context overflow — most actionable
    if status == 400 and any(k in msg for k in ("context_length", "max_tokens", "too long", "token limit")):
        return ClassifiedError(FailureReason.context_overflow, status, retryable=False,
                               should_compress=True, should_fallback=False)
    if status == 413:
        return ClassifiedError(FailureReason.context_overflow, status, retryable=False,
                               should_compress=True, should_fallback=False)

    # Rate limit
    if status == 429:
        return ClassifiedError(FailureReason.rate_limit, status, retryable=True,
                               should_compress=False, should_fallback=True)

    # Billing / quota
    if status == 402 or "quota" in msg or "billing" in msg:
        return ClassifiedError(FailureReason.billing, status, retryable=False,
                               should_compress=False, should_fallback=True)

    # Auth
    if status in (401, 403):
        return ClassifiedError(FailureReason.auth, status, retryable=False,
                               should_compress=False, should_fallback=True)

    # Content filter
    if "content_filter" in msg or "safety" in msg:
        return ClassifiedError(FailureReason.content_filter, status, retryable=False,
                               should_compress=False, should_fallback=False)

    # Server error
    if status and 500 <= status < 600:
        return ClassifiedError(FailureReason.server_error, status, retryable=True,
                               should_compress=False, should_fallback=True)

    # Timeout / connection
    if isinstance(exc, (TimeoutError, ConnectionError)):
        reason = FailureReason.timeout if isinstance(exc, TimeoutError) else FailureReason.connection
        return ClassifiedError(reason, None, retryable=True,
                               should_compress=False, should_fallback=True)

    return ClassifiedError(FailureReason.unknown, status, retryable=True,
                           should_compress=False, should_fallback=False)
```

集成到 `BaseLLMAdapter._run_with_retries()` 替代现有的 `should_open_provider_circuit()` + `_should_retry_exception()` 逻辑：

```python
# 在 _run_with_retries 的 except 块中：
classified = classify_llm_error(exc, provider=self.provider, ...)
if classified.should_compress and self._compressor:
    # 触发上下文压缩后重试（与改进 1 联动）
    ...
if classified.should_fallback and self._fallback_adapter:
    # 委托给 fallback adapter（与改进 12 联动）
    ...
if not classified.retryable:
    break
```

### 文件变更

| 文件 | 操作 |
|------|------|
| `adapters/llm/error_classifier.py` | 新建，~100 行 |
| `adapters/llm/base.py` | `_run_with_retries()` 使用 `classify_llm_error()` 替代现有判断逻辑 |

---

## 改进 12：Provider Failover Chain（P1）

### 问题

Kepler 的 `ProviderBinding` 只绑定单个 provider。
当主 provider 出错（rate-limit、billing exhausted、服务宕机）时，整个研究流程中断。
对于长时间运行的科研 agent（多轮检索 + 多步推理），可用性至关重要。

### 借鉴 Hermes

Hermes 实现了完整的 provider failover：

1. **Fallback chain**（`_fallback_chain`）：有序列表 `[{provider, model, base_url?, api_key?}, ...]`
2. **Eager fallback**：rate-limit 时不浪费 retry 窗口，直接切换
3. **Per-turn primary restoration**（`_restore_primary_runtime()`）：每轮开始恢复主 provider，
   避免一次性故障永久钉在 fallback 上
4. **Credential pool rotation**（`CredentialPool`）：同 provider 下多 API key 轮换，
   支持 4 种策略（fill_first / round_robin / random / least_used）

### 实现方案

Kepler 不需要 Hermes 的全部 credential pool 复杂度，但需要 **provider fallback chain**：

```python
# adapters/llm/fallback_adapter.py
class FallbackLLMAdapter(BaseLLMAdapter):
    """Wraps a primary adapter with an ordered fallback chain."""

    def __init__(
        self,
        primary: BaseLLMAdapter,
        fallbacks: list[BaseLLMAdapter],
        *,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ):
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self._primary = primary
        self._fallbacks = fallbacks
        self._active: BaseLLMAdapter = primary
        self._is_fallback_active = False

    async def _generate_structured(self, prompt, input_data, response_model):
        try:
            return await self._active._generate_structured(prompt, input_data, response_model)
        except Exception as exc:
            classified = classify_llm_error(exc)
            if classified.should_fallback:
                next_adapter = self._activate_next_fallback()
                if next_adapter:
                    return await next_adapter._generate_structured(prompt, input_data, response_model)
            raise

    def _activate_next_fallback(self) -> BaseLLMAdapter | None:
        """Switch to the next available fallback adapter."""
        for fb in self._fallbacks:
            if not fb._is_provider_circuit_open():
                self._active = fb
                self._is_fallback_active = True
                logger.info("Activated fallback provider: %s", getattr(fb, 'model', 'unknown'))
                return fb
        return None

    def restore_primary(self) -> None:
        """Restore primary adapter at the start of a new turn."""
        if self._is_fallback_active:
            self._active = self._primary
            self._is_fallback_active = False
```

配置方式（`core/config.py`）：
```python
# Settings 新增
llm_fallback_providers: list[dict] = []
# 格式: [{"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}]
```

集成点：
- `_build_llm_adapter()` 构建 primary + fallbacks，包装为 `FallbackLLMAdapter`
- `ReActReasoningAgent.reason()` 每轮开始调用 `adapter.restore_primary()`

### 文件变更

| 文件 | 操作 |
|------|------|
| `adapters/llm/fallback_adapter.py` | 新建，~150 行 |
| `core/config.py` | 新增 `llm_fallback_providers` |
| `apps/api/runtime.py` | `_build_llm_adapter()` 支持构建 fallback chain |

---

## 改进 13：Sensitive Data Redaction（P1）

### 问题

Kepler 在以下场景可能泄露敏感数据：
1. **上下文压缩**（改进 1）将对话发送给 auxiliary model 做摘要时，可能包含 API key、密码
2. **long-term memory** 存储的内容可能包含用户的 credential
3. **日志输出** 可能包含 token、密钥

### 借鉴 Hermes

`agent/redact.py` 实现了 `redact_sensitive_text()`：
- 在所有发往 auxiliary model 的内容上应用（summarization、background review）
- 在 compression summary 输出上也再次应用（LLM 可能无视 prompt 指令原样回显密钥）
- 模式覆盖：AWS keys、GitHub tokens、API keys（`sk-`、`ghp_`）、Bearer tokens、
  Base64 encoded secrets、connection strings、`.env` 文件内容

### 实现方案

```python
# security/redact.py
import re
from typing import Any

_REDACT_PATTERNS = [
    # API keys
    (r'(sk-[a-zA-Z0-9]{20,})', "[REDACTED:api_key]"),
    (r'(ghp_[a-zA-Z0-9]{36,})', "[REDACTED:github_token]"),
    (r'(gho_[a-zA-Z0-9]{36,})', "[REDACTED:github_oauth]"),
    # AWS
    (r'(AKIA[0-9A-Z]{16})', "[REDACTED:aws_key]"),
    # Bearer tokens
    (r'(Bearer\s+[a-zA-Z0-9\-._~+/]+=*)', "[REDACTED:bearer]"),
    # Generic secrets in key=value patterns
    (r'(?i)((?:api[_-]?key|secret|token|password|passwd|credential)\s*[=:]\s*)[^\s,;]{8,}',
     r'\1[REDACTED]'),
    # Connection strings
    (r'(?i)((?:mysql|postgres|mongodb|redis)://[^\s]+)', "[REDACTED:connection_string]"),
]

_COMPILED = [(re.compile(p), r) for p, r in _REDACT_PATTERNS]

def redact_sensitive_text(text: str) -> str:
    """Strip secrets from text before sending to auxiliary models or storage."""
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
```

集成点：
- `context/compressor.py`：`_serialize_for_summary()` 和 summary 输出都调用 `redact_sensitive_text()`
- `memory/security.py`：与改进 4 合并，`scan_memory_content()` 前先 redact
- logging：`BaseLLMAdapter` 的错误日志调用 `redact_sensitive_text(str(exc))`

### 文件变更

| 文件 | 操作 |
|------|------|
| `security/redact.py` | 新建，~60 行 |
| `context/compressor.py` | 序列化和输出时调用 redact |
| `memory/security.py` | import 并复用 redact 模式 |

---

## 改进 14：Anti-Thrashing Compression Guard（P0 补充）

### 问题

改进 1 的上下文压缩引擎如果不加保护，可能出现 **压缩死循环**：
当对话中每条消息都很大（如科研 agent 的长 evidence bundle）时，
压缩只能移除 1-2 条消息，token 几乎不下降，下轮又触发压缩。

### 借鉴 Hermes

`context_compressor.py` 的 anti-thrashing 设计：
1. 跟踪每次压缩的 `savings_pct`（节省百分比）
2. 连续 2 次 `savings_pct < 10%` → 跳过后续压缩
3. LLM summary 失败后进入冷却期（60s transient / 600s permanent）
4. summary model 不可用时自动降级到 main model

同时 Hermes 的 `_find_tail_cut_by_tokens()` 确保最后一条用户消息永远在尾部保护区内（防止 active task 丢失）。

### 实现方案

直接作为改进 1 `ContextCompressor` 的内置逻辑，不需要独立文件：

```python
# context/compressor.py 中 ContextCompressor 新增
class ContextCompressor:
    def __init__(self, ...):
        ...
        self._last_savings_pct: float = 100.0
        self._ineffective_count: int = 0
        self._summary_cooldown_until: float = 0.0

    def should_compress(self, current_tokens: int, threshold: int) -> bool:
        if current_tokens < threshold:
            return False
        # Anti-thrashing: back off after 2 consecutive ineffective compressions
        if self._ineffective_count >= 2:
            logger.warning(
                "Compression skipped — last %d compressions saved <10%% each",
                self._ineffective_count,
            )
            return False
        # Summary model cooldown
        if time.monotonic() < self._summary_cooldown_until:
            return False
        return True

    def compress_messages(self, messages, ...) -> list[dict]:
        before_tokens = self._estimate_tokens(messages)
        compressed = self._do_compress(messages, ...)
        after_tokens = self._estimate_tokens(compressed)

        savings_pct = (before_tokens - after_tokens) / before_tokens * 100
        self._last_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_count += 1
        else:
            self._ineffective_count = 0
        return compressed

    def _ensure_last_user_in_tail(self, messages, cut_idx, head_end) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Prevents active task from being summarized away — the LLM would
        lose track of what the user just asked.
        """
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                if i < cut_idx:
                    return max(i, head_end + 1)
                return cut_idx
        return cut_idx
```

### 文件变更

| 文件 | 操作 |
|------|------|
| `context/compressor.py` | 改进 1 实现中内置 anti-thrashing 逻辑 + `_ensure_last_user_in_tail()` |

> 注：此改进是改进 1 的必要补充，应一起实现。

---

## 改进 15：Per-Turn Context Window Probing（P2）

### 问题

不同模型 / 不同 provider 的实际 context window 可能与文档值不同：
- OpenRouter 代理的某些 provider 只支持 32K，虽然模型标称 128K
- Anthropic 长 context tier 需要付费开通，否则上限 200K 变 200K 但可能被 429
- 自建模型的实际 VRAM 可能限制 context

Kepler 的 `MODEL_CONTEXT_LENGTHS` 静态表无法覆盖这些运行时差异。

### 借鉴 Hermes

Hermes 的 context probing 策略：
1. 首次遇到 context-overflow 错误时，记录实际可用窗口（`_context_probed = True`）
2. 动态下调 `context_length` 和 `threshold_tokens`
3. Anthropic `long_context_tier` 错误专门处理：降到 200K
4. 不持久化 probe 结果（订阅升级后应自动恢复）

### 实现方案

```python
# context/token_counter.py 扩展 TokenBudget
class TokenBudget:
    def __init__(self, model: str, ...):
        ...
        self._probed = False

    def handle_context_overflow(self, error_msg: str) -> bool:
        """React to a context-overflow error by probing actual limits.

        Returns True if the budget was adjusted (caller should retry after compress).
        """
        if self._probed:
            return False  # already adjusted, don't loop

        # Try to extract actual limit from error message
        import re
        match = re.search(r'maximum.*?(\d{4,})\s*token', error_msg, re.I)
        if match:
            actual_limit = int(match.group(1))
            if actual_limit < self.total:
                logger.warning(
                    "Context probe: model reports %d tokens (was %d), adjusting",
                    actual_limit, self.total,
                )
                self.total = actual_limit
                self.available = self.total - self.reserved_output
                self._probed = True
                return True

        # Fallback: reduce by 25%
        reduced = int(self.total * 0.75)
        logger.warning("Context probe: reducing budget from %d to %d", self.total, reduced)
        self.total = reduced
        self.available = self.total - self.reserved_output
        self._probed = True
        return True
```

集成点：
- 当改进 11 的 `ClassifiedError.should_compress == True` 时调用
- `TokenBudget.handle_context_overflow()` → 压缩 → 重试
- 与改进 1、改进 3 协同工作

### 文件变更

| 文件 | 操作 |
|------|------|
| `context/token_counter.py` | `TokenBudget` 新增 `handle_context_overflow()` |

---

## 实施路线图

```
Week 1 (P0):
  ├── Day 1-2: Token 预算管理 (改进 3)
  ├── Day 3:   Prompt Caching (改进 2)
  └── Day 4-5: 上下文压缩引擎 + Anti-Thrashing (改进 1 + 14)

Week 2 (P1):
  ├── Day 1:   Error Classification (改进 11)
  ├── Day 2:   Provider Failover Chain (改进 12)
  ├── Day 3:   Sensitive Data Redaction (改进 13)
  ├── Day 4:   记忆安全加固 (改进 4)
  └── Day 5:   MCP 客户端升级启动 (改进 5)

Week 3 (P1 收尾 + P2):
  ├── Day 1-2: MCP 客户端升级完成 (改进 5)
  ├── Day 3:   Tool 自注册 (改进 6)
  ├── Day 4:   Frozen Snapshot (改进 7)
  └── Day 5:   Context Window Probing (改进 15)

Week 4 (P3, 可选):
  ├── Day 1-2: Agent-Editable Skills (改进 8)
  ├── Day 3:   Toolset 平台适配 (改进 9)
  └── Day 4:   Dangerous Command 审批 (改进 10)
```

### 验证标准

| 改进 | 可验证目标 |
|------|-----------|
| 上下文压缩 | 50 轮对话不溢出 128K context；压缩后 ReAct 仍能引用前序证据 |
| Prompt Caching | Anthropic API 日志显示 cache_read_input_tokens > 0 |
| Token 预算 | `budget.should_compress()` 在 80% 时触发，不出现 context length exceeded 错误 |
| 记忆安全 | `scan_memory_content("ignore all previous instructions")` → blocked |
| MCP 升级 | stdio client 连接 → kill server → 自动重连 → tool call 成功 |
| Tool 自注册 | 新 tool 文件放入 `tools/`，无需修改任何其他文件 |
| Frozen Snapshot | session 内 memory.upsert() 不改变 system prompt hash |
| Agent Skills | agent 执行 `skill_manage(action="create", ...)` → 文件持久化 |
| Toolset | `resolve_toolset("kepler-cli")` 返回完整 tool 列表 |
| 审批 | `ToolExecutor` 遇到 `rm -rf /` 时拒绝执行并 log |
| Error Classification | `classify_llm_error(429_exc).should_fallback == True` |
| Provider Fallback | primary 返回 429 → 自动切 fallback → 返回成功结果 |
| Redaction | `redact_sensitive_text("sk-abc123...")` → `[REDACTED:api_key]` |
| Anti-Thrashing | 连续 2 次低效压缩后 `should_compress()` 返回 False |
| Context Probing | 首次 context-overflow → budget 自动下调 → 压缩后重试成功 |

### 依赖新增

```toml
# pyproject.toml [project.optional-dependencies]
context = ["tiktoken>=0.7.0"]
mcp-client = ["mcp>=1.0"]
```

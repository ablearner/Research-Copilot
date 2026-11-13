# Research-Copilot

`Research-Copilot` 是一个面向多用户科研工作流的 Agent 平台：用户在个人工作区创建项目、导入论文、开展研究并保存对话与报告。现有 Supervisor / Specialist / RAG 负责研究能力，新增后端负责身份、权限、任务可靠性与业务持久化。MySQL 仅保存服务端业务事实，Agent 记忆仍由独立后端管理。

新平台入口是 `apps.platform.main:app`，前端为 `web/src/PlatformApp.tsx`，配套独立 Worker 与 Scheduler。先阅读[运行手册](docs/platform-runbook.md)，再查看[架构与 ER](docs/platform-architecture.md)、[演示脚本](docs/platform-demo.md)、[验证记录](docs/platform-validation.md)、[性能报告](docs/platform-performance.md)和[工程复盘](docs/platform-retrospective.md)。首期为单 Worker 的单机部署，不包含计费、团队共享或生产高可用承诺。

## 当前代码主线

当前项目最值得先记住的是四层边界：

- `services/research/`
  高层 research 业务服务层，负责 conversation、task、paper pool、import job、collection QA、workspace、report 和持久化收口。
- `runtime/research/`
  Supervisor Graph 运行时，负责构建 `ResearchAgentToolContext`、Markdown skill 候选召回、manager 决策、specialist agent 调度和最终响应聚合。
- `tools/research/`
  研究领域能力单元，负责 query planning、paper ranking、survey writing、paper import/search、QA routing、Zotero search 等可复用业务能力。
- `rag_runtime/`
  底层 tool-first 执行层，负责 `parse / index / retrieve / answer / chart understanding`。

当前活跃前端工作区（`web/` 目录）：

- 技术栈：Next.js 15 (App Router) + React 18 + TypeScript + Tailwind CSS。
- 应用入口：`web/app/layout.tsx` + `web/app/page.tsx` → `web/src/PlatformApp.tsx`。
- API 封装：`web/src/platform-api.ts`，通过同源 `/api/backend/*` 代理访问新平台 `/api/v1`。
- 产品流程：注册/验证/登录、项目、论文库、异步导入、研究对话、任务与报告、会话和个人凭证管理。
- 私有文件通过平台授权下载；旧 `web/app/api/files/[...path]/route.ts` 返回 404。

`web/src/App.tsx`、`web/src/api.ts` 与原 `ChatView`/`MessageBubble` 等组件保留作离线兼容代码，不是当前 `page.tsx` 使用的在线入口。旧组件的内联图片和轨迹展示能力不能直接算作新平台已验收功能。

## 架构概览

```text
Browser → Next.js / PlatformApp → 同源代理 → apps.platform /api/v1
                                               ↓
                           MySQL 业务事实 / Redis 限流 / 私有对象存储
                                               ↓
                       Outbox → Scheduler → RabbitMQ → 单 Research Worker
                                               ↓
                          AgentBridge（身份、文档版本、attempt 隔离）
                                               ↓
                       LiteratureResearchService → Supervisor / Specialists
                                               ↓
                         RagRuntime → scoped retrieval / tools / 独立记忆
```

上面的四层 Agent 边界继续保留。在线桥接每次执行构造受限上下文，关闭服务器个人 Zotero、全局 MCP 和本地代码执行；当前图使用 attempt 内存图。离线模式仍可使用原有工具配置，具体能力不能直接等同于在线开放权限。

## Agent 能力基础（在线开放范围见上文）

- `research discovery`
  根据研究主题搜索 arXiv、OpenAlex、Semantic Scholar、IEEE 等学术源，生成候选论文池和研究草稿。
- `paper import`
  对选中的论文执行 `download -> parse -> optional Zotero sync -> embedding index`。`ImportPapersRequest.fast_mode` 默认是 `true`，因此主导入阶段默认先完成 embedding 索引；如果 `include_graph=true`，图谱索引会标记为 `graph_backfill_pending`，后台导入 job 会继续执行 graph backfill。索引阶段有 `RESEARCH_IMPORT_INDEX_TIMEOUT_SECONDS` 超时保护；索引超时会保留已解析/已同步的论文，不再把整次导入判为失败。
- `collection QA`
  基于已导入论文集合做混合检索和 grounded answer，必要时下钻到单文档或图表链路。
- `paper question answering`
  对已导入且索引成功的论文，问答优先走 `RAG 正文证据 -> 图谱/摘要证据 -> 元数据兜底`。如果论文仍是 `index_status=timeout|failed|skipped`，系统不会假装有全文证据，只会用标题、摘要、候选元数据做弱回答并暴露证据边界。
- `advanced actions`
  做多论文对比、优先阅读推荐、上下文压缩，并写回 workspace。
- `CLI terminal agent`
  支持本地会话、profile 管理、runtime 配置和交互式研究终端。

## 记忆系统

项目里有两套相关但边界不同的“记忆”：

- research agent memory
  定义在 `memory/`，包括 `WorkingMemory`、`SessionMemory`、`LongTermMemory`、`PaperKnowledgeMemory` 和 `UserProfileMemory`。`UserProfileMemory` 构建在 `LongTermMemory` 之上，由 `MemoryManager` 统一管理。
- RAG runtime session memory
  定义在 `rag_runtime/memory.py`，核心是 `GraphSessionMemory`，用于给底层问答链路保存结构化会话快照。

以下是原 Agent/离线运行配置；在线桥接会覆盖目录、身份和图 provider，并按作用域发布记忆：

- `SESSION_MEMORY_PROVIDER=sqlite`
  底层 `GraphSessionMemory` 默认写入 `RESEARCH_SQLITE_DB_PATH` 的 `session_memory` 表；research agent 的 `SessionMemory` 默认写入同一个 DB 的 `research_session_memory` 表。
- `LONG_TERM_MEMORY_PROVIDER=sqlite`
  `MemoryManager` 默认通过 `memory.factory.build_memory_manager()` 统一构造，长期记忆写入 `long_term_memory` + `long_term_memory_fts`。
- `PAPER_KNOWLEDGE_PROVIDER=sqlite`
  论文知识卡片默认写入同一个 `kepler.db` 的 `research_paper_knowledge` 表。JSON store 仅保留为 legacy/test/migration 兼容后端。
- `VECTOR_STORE_PROVIDER=milvus`
- `GRAPH_STORE_PROVIDER=memory`

原离线环境配置可能覆盖上述代码默认值，例如使用 `RUNTIME_BACKEND=business`、OpenAI-compatible LLM/embedding、Milvus compose 暴露端口 `19531`、`GRAPH_STORE_PROVIDER=neo4j`。Milvus 同时承载向量检索和原生 Sparse-BM25 关键词检索；在这种配置下 Neo4j 是必需服务。如果保持代码默认的 `GRAPH_STORE_PROVIDER=memory`，则不需要启动 Neo4j。

## 主要目录

- `web/`
  当前活跃前端（Next.js 15 + React 18 + Tailwind），ChatGPT 风格对话式布局
- `apps/platform/`、`apps/worker/`、`apps/scheduler/`
  在线业务 API、可靠任务执行、Outbox 投递与租约恢复
- `services/platform/`、`services/identity/`、`services/projects/`、`services/task_execution/`
  多用户业务、权限、资源生命周期与任务状态机
- `apps/api/`
  原离线兼容 API 与可复用 runtime 装配；不是公网多用户入口
- `services/research/`
  research 主业务 service、paper operations、QA router 和 workspace/report/conversation 持久化
- `runtime/research/`
  高层 Supervisor Graph runtime、context builder、unified action adapters、result aggregator 和 response formatter
- `agents/`
  高层采用 1 个 Supervisor + 4 个子 Agent：LiteratureScout（检索与推荐）、Knowledge（文档与知识管理）、Analyst（分析与问答）、Writer（综述写作）。内部流程位于 `tools/research/workflows/`，不单独注册为 Agent。
- `rag_runtime/`
  底层 RAG runtime
- `tools/`
  文档、图表、检索、回答等稳定工具层
- `tools/research/`
  研究领域能力与工具：学术源搜索、Zotero search、query planning、paper ranking/reading/analysis、survey/review writing、QA routing 等
- `retrieval/`
  vector / sparse / graph / graph summary 检索与重排
- `adapters/`
  LLM、embedding、vector store、graph store 等基础设施适配
- `adapters/mcp/zotero_local.py`
  Zotero 本地 Connector 网关和进程内 MCP app
- `adapters/storage/`
  新平台使用 `adapters/storage/mysql/` 保存业务数据；原 JSON / SQLite StorageBackend 保留作离线、迁移与测试兼容
- `observability/`
  轻量级进程内指标收集（counters、histograms）、LLM 调用自动埋点
- `security/`
  敏感数据脱敏（API key、token、密码、连接串等 11 种模式）
- `memory/`
  research 侧工作记忆、会话记忆、长期记忆、用户画像记忆和论文知识记忆
- `chains/` + `rag_runtime/strategies.py`
  可复用推理策略实现与协议。`CoTReasoningAgent` 和 `PlanAndSolveReasoningAgent` 在 `chains/`，独立的底层 ReAct 工具推理 worker 在 `agents/rag_react_qa_worker.py`，策略组合定义在 `rag_runtime/strategies.py`；它不计入上述 4 个高层子 Agent。
- `tooling/`
  工具注册 / 执行框架和 research function 规范
- `mcp/`
  MCP（Model Context Protocol）客户端注册和服务端暴露。已实现标准 MCP 协议，详见下方 [MCP 标准化架构](#mcp-标准化架构) 章节
- `evaluation/`
  评测框架（7 种 case 类型）、benchmark 构建、metrics 和数据集管理
- `core/`
  共享核心模块：配置（`config.py`）、知识加载（`knowledge_loader.py`）、技能系统（`skill_registry.py`、`skill_matcher.py`、`skill_validator.py`）
- `skills/`
  标准化技能目录，`builtin/` 内置技能（git-tracked），`community/` 社区/外部技能（gitignored）。每个技能是一个含 `SKILL.md` 的文件夹
- `scripts/`
  环境检查、Milvus 重置、SciGraphQA 入库、WSL Zotero 桥接等工具脚本
- `sdk/`
  本地 CLI / SDK 调用封装和 runtime profile 管理

## 新平台与离线入口

在线运行使用 `apps.platform.main:app` 的 `/api/v1`；完整启动命令见[运行手册](docs/platform-runbook.md)。在线 CLI 位于 `apps/platform/cli.py`，使用可撤销 PAT。

### 原 Research API（仅受控离线兼容）

下面列出旧 `apps.api` 接口，不是新平台 Web 的请求入口。生产配置禁止启动旧入口，不应将其匿名路由、reset 或本地文件能力暴露到公网。

- `POST /research/agent`
  原 Agent 统一调用入口
- `POST /research/agent/stream`
  原离线前端使用的 SSE 流式入口
- `GET/POST /research/conversations`
- `GET/PATCH/DELETE /research/conversations/{conversation_id}`
- `POST /research/reset`
- `POST /research/papers/search`
- `POST /research/tasks`
- `GET /research/tasks/{task_id}`
- `POST /research/tasks/{task_id}/run`
- `GET /research/tasks/{task_id}/report`
- `POST /research/tasks/{task_id}/papers/import`
- `POST /research/tasks/{task_id}/papers/import/jobs`
- `GET /research/jobs/{job_id}`
- `GET /research/tasks/{task_id}/papers/{paper_id}/figures`
- `POST /research/tasks/{task_id}/papers/{paper_id}/figures/analyze`
- `PATCH /research/tasks/{task_id}/todos/{todo_id}`
- `POST /research/tasks/{task_id}/todos/{todo_id}/search`
- `POST /research/tasks/{task_id}/todos/{todo_id}/import`

兼容入口仍保留但已标记为 deprecated：`POST /research/agent/run`、`POST /research/agent/chat`、`POST /research/papers/import`、`POST /research/papers/import/jobs`。

### Research Document API

- `POST /research/documents/upload`
- 文档解析、索引、图表理解和 grounded QA 已收编为 Supervisor 内部 knowledge tools，不再暴露独立低层 API
- `GET /health` — 应用健康检查（含 uptime）
- `GET /health/metrics` — 运行时指标（LLM 调用延迟/计数/错误分布）

### CLI

入口文件是 `apps/cli.py`，console script 别名是：

- `research-copilot`
- `kepler`

一级命令：

- `list-conversations`
- `show-profile`
- `doctor`
- `status`
- `trajectory`
- `update-profile`
- `models show|set`
- `plugins list|enable|disable`
- `agent`

详细说明见 [docs/CLI终端使用文档.md](docs/CLI终端使用文档.md)。

## 配置与依赖

配置入口是 [core/config.py](core/config.py)，默认从项目根目录 `.env` 读取。

关键配置维度：

- LLM
  `LLM_PROVIDER`、`LLM_MODEL`、`OPENAI_API_KEY`、`DASHSCOPE_API_KEY`
- embedding / reranker
  `EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`RERANKER_MODEL`、`RERANKER_UNAVAILABLE_POLICY`
- stores
  `VECTOR_STORE_PROVIDER`、`GRAPH_STORE_PROVIDER`、`SESSION_MEMORY_PROVIDER`、`LONG_TERM_MEMORY_PROVIDER`、`PAPER_KNOWLEDGE_PROVIDER`
- external store config
  `MILVUS_URI`、`NEO4J_URI`
- research persistence
  `RESEARCH_STORAGE_ROOT`、`RESEARCH_RESET_ON_STARTUP`
- long-term memory
  `LONG_TERM_MEMORY_PROVIDER`、`MEMORY_QUALITY_GATE_ENABLED`、`MEMORY_MIN_QUALITY_SCORE`
- uploads
  `UPLOAD_DIR`、`UPLOAD_MAX_BYTES`

代码默认值中比较关键的几点：

- `APP_ENV=local` 时会暴露 `/uploads/*`
- `UPLOAD_MAX_BYTES` 默认 `25 MiB`
- `VECTOR_STORE_PROVIDER` 默认 `milvus`
- `MILVUS_URI` 代码默认 `http://localhost:19530`；使用仓库的 `docker-compose.milvus.yml` 时需要配置为 `http://localhost:19531`
- `MILVUS_DIMENSION` 代码默认自动跟随 embedding adapter；如果用 `text-embedding-3-large`，本地通常配置为 `3072`
- Milvus collection 会同时创建 dense vector 字段和 Sparse-BM25 字段；已有旧 collection 需要重建后才能使用原生 BM25
- `GRAPH_STORE_PROVIDER` 代码默认 `memory`；业务 `.env` 可配置为 `neo4j`，此时后端启动会要求 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` 且 Neo4j 可连接
- `SESSION_MEMORY_PROVIDER` 默认 `sqlite`
- `LONG_TERM_MEMORY_PROVIDER` 默认 `sqlite`
- `PAPER_KNOWLEDGE_PROVIDER` 默认 `sqlite`
- `MEMORY_QUALITY_GATE_ENABLED` 默认开启，低质量/无证据 QA 结论只保留在会话记忆，不写入长期记忆
- `RESEARCH_IMPORT_INDEX_TIMEOUT_SECONDS` 默认 `300`
- `STORAGE_PROVIDER` 默认 `sqlite`（可切换为 `json` 用于 legacy/test/migration 兼容）
- `CORS_ALLOW_ORIGINS` 默认允许 localhost:3000/3001
- `RATE_LIMIT_MAX_REQUESTS` 默认 60 次/分钟/IP
- `JSON_LOG_FORMAT` 默认 `false`，设为 `true` 启用结构化 JSON 日志
- WSL 下 `kepler agent` 会在启动时尝试从默认网关自动注入 Windows 代理环境，避免 `api.bltcy.ai` 直连 DNS/443 不稳定导致 manager 或 embedding 调用卡住。可用 `KEPLER_WSL_PROXY_AUTO_ENABLE=0` 关闭。

## MCP 标准化架构

项目的 MCP（Model Context Protocol）客户端已完成标准化，完全符合 MCP 规范：

### 协议合规

- **JSON-RPC 2.0 信封**：所有 MCP 通信（`tools/list`、`tools/call`、`initialize` 等）均使用标准 JSON-RPC 2.0 格式，包含 `jsonrpc`、`method`、`params`、`id` 字段
- **initialize 握手**：客户端在首次通信前自动执行 MCP `initialize` 握手，交换 `protocolVersion`、`capabilities` 和 `clientInfo`，并发送 `notifications/initialized` 通知
- **错误处理**：自动解析 JSON-RPC `error` 对象（含 `code` 和 `message`），并对错误消息进行凭据脱敏

### 传输方式

| 传输方式 | 客户端类 | 说明 |
|---------|---------|------|
| stdio | `StdioMCPClient` | 通过子进程 stdin/stdout 通信，支持自动重连、指数退避 |
| HTTP | `HttpMCPClient` | 通过 HTTP POST 通信，支持自定义 headers 和重试 |
| 进程内 | `InProcessMCPClient` | 直接调用本地 MCPServerApp，无网络开销 |

### 添加外部 MCP 工具

用户可以通过配置文件动态添加外部 MCP 工具服务器（如论文搜索、翻译等）：

1. 复制配置模板：`cp mcp_servers.example.json mcp_servers.json`
2. 编辑 `mcp_servers.json`，添加你的 MCP 服务器：

```json
{
  "servers": {
    "paper-search": {
      "transport": "http",
      "url": "http://localhost:3001",
      "headers": {"Authorization": "Bearer your-token"},
      "timeout": 60,
      "enabled": true
    },
    "local-tool": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@some/mcp-server"],
      "env": {"API_KEY": "xxx"},
      "enabled": true
    }
  }
}
```

3. 重启后端服务，系统自动注册并初始化配置的 MCP 服务器
4. 外部工具会自动注入到 ReAct 推理引擎的可用工具池中

**注意**：`mcp_servers.json` 已加入 `.gitignore`，不会被提交到版本控制。

### 关键文件

| 文件 | 职责 |
|-----|------|
| `mcp/client/base.py` | MCP 客户端基类、JSON-RPC 2.0 构建器、协议常量 |
| `mcp/client/stdio_client.py` | stdio 传输客户端 |
| `mcp/client/http_client.py` | HTTP 传输客户端 |
| `mcp/client/registry.py` | MCP 客户端注册中心、配置加载、工具发现 |
| `mcp/security.py` | 环境变量过滤、错误脱敏、工具描述注入检测 |
| `mcp/server/app.py` | MCP 服务端（对外暴露本系统能力） |
| `mcp_servers.example.json` | 用户配置模板 |

## Skill 系统

项目实现了标准化 Markdown skill 系统，支持内置 skill 和外部社区 skill 的即插即用。

当前边界：

- Markdown skill 是 `skills/**/SKILL.md` 中的工作流指令包，不是可执行工具，也不是 Agent。
- Python 业务能力统一放在 `tools/research/`，只保留 `PaperReadingTool`、`PaperAnalysisTool`、`PaperRankingTool`、`ResearchEvaluationTool` 等 `*Tool` 命名。
- 每次请求先召回轻量 `skill_candidates` 给 Supervisor；Supervisor 输出 `selected_skill_names`；完整 `SKILL.md` 正文只注入被调度的 specialist message。

### 目录结构

```text
skills/
├── builtin/                  # 内置技能（git-tracked）
│   ├── academic-concepts/
│   ├── chart-analysis/
│   ├── knowledge-management/
│   ├── paper-comparison/     # 论文对比分析
│   ├── paper-discovery/
│   ├── paper-recommendation/
│   ├── literature-survey/    # 文献综述写作
│   ├── paper-reading/        # 论文深度阅读
│   ├── research-evaluation/  # 研究质量评估
│   └── research-qa/
├── community/                # 社区/外部技能（gitignored）
└── .skill_config.json        # 技能启用/禁用配置（gitignored）
```

### 核心模块

| 文件 | 职责 |
|-----|------|
| `core/skill_registry.py` | SkillRegistry — 扫描、索引、三级渐进加载（L1 元数据 / L2 指令 / L3 引用文件） |
| `core/skill_matcher.py` | SkillMatcher — trigger 正则匹配 + tag/description 关键词匹配，可接 embedding/reranker |
| `core/skill_validator.py` | SkillValidator — prompt injection 检测、路径安全、大小限制 |
| `tools/research/skill_resolver.py` | ResearchSkillResolver — 先生成候选，再按 Supervisor 选择加载正文 |

### 添加外部技能

从网上下载的技能文件夹（包含 `SKILL.md`）放入 `skills/community/` 即可：

```bash
cp -r downloaded-skill/ skills/community/
# 重启后自动扫描并可用
```

### SKILL.md 格式

每个技能目录必须包含一个 `SKILL.md` 文件，使用 YAML frontmatter + Markdown body：

```markdown
---
name: my-skill
description: "技能描述"
triggers:
  - "触发正则1"
  - "触发正则2"
tags: [tag1, tag2]
trust_level: community    # builtin | trusted | community
enabled: true
requires:
  tools: [tool_name]
---

# 技能指令（Markdown）
具体的执行步骤和输出格式要求...
```

更完整的链路见 [docs/RC技能系统选取加载使用详解.md](docs/RC技能系统选取加载使用详解.md)。

## 原离线 Agent 启动（兼容开发）

新平台请使用[运行手册](docs/platform-runbook.md)启动 API、Worker、Scheduler 和 Web。以下命令仅保留原 Agent 的受控本地开发路径，不能与新平台占用同一 API 端口，也不能作为当前 Web 的完整启动方案。

### 原服务启动顺序

#### 第一步：基础设施

```bash
# 1. 启动 Milvus（Docker Compose，含 etcd + minio + milvus）
cd /home/myc/Research-Copilot
docker-compose -f docker-compose.milvus.yml up -d
# 等待端口 19531 就绪（约 15-30 秒）
# 检查：curl -s http://localhost:9092/healthz  →  应返回 "OK"
# 如果看到 orphan container `kepler-opensearch` 提示，说明旧 OpenSearch 服务已从 compose 中移除；
# 可执行 `docker-compose -f docker-compose.milvus.yml up -d --remove-orphans` 清理旧容器。

# 2. 如果 GRAPH_STORE_PROVIDER=neo4j，启动 Neo4j（Docker 容器，数据挂载在 .data/neo4j/）
docker start kepler-neo4j
# 等待端口 7687 就绪（约 10-15 秒）
# 检查：docker exec kepler-neo4j cypher-shell -u neo4j -p neo4j1234 'RETURN 1'

# 3. 如果需要访问本地 Zotero，启动 Zotero Bridge（WSL 环境，需要 Windows 上先打开 Zotero 桌面程序）
bash scripts/wsl_zotero_bridge.sh start
# 检查：ss -tlnp | grep 23119
```

#### 第二步：后端

```bash
cd /home/myc/Research-Copilot
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m uvicorn apps.api.main:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir apps --reload-dir services --reload-dir rag_runtime \
  --reload-dir agents --reload-dir tools --reload-dir adapters \
  --reload-dir memory --reload-dir retrieval --reload-dir skills --reload-dir domain
# 等待日志出现 "Application startup complete"
# 如果卡在 "Waiting for application startup"，优先检查 Milvus；若 GRAPH_STORE_PROVIDER=neo4j，再检查 Neo4j
# 注意：必须用 --reload-dir 限制监视目录，否则 watchfiles 会因 web/node_modules 文件过多而崩溃
```

原 `App.tsx` 未挂载在当前页面；启动旧 API 后可通过原 CLI/SDK 调试，不会自动切换新 Web 的业务接口。

### 端口一览

| 服务 | 端口 | 启动方式 |
|------|------|------|
| Milvus | 19531 (映射到容器内 19530) | 向量检索 + Sparse-BM25 关键词检索，`docker-compose -f docker-compose.milvus.yml up -d` |
| Neo4j | 7474/7687 | 仅 `GRAPH_STORE_PROVIDER=neo4j` 时需要，常用 `docker start kepler-neo4j` |
| Zotero Bridge | 23119 | 仅需要本地 Zotero 时启动，`bash scripts/wsl_zotero_bridge.sh start` |
| 后端 (FastAPI) | 8000 | `uvicorn apps.api.main:app` |
| 新平台前端 (Next.js) | 3000 | 需配套新平台服务，见运行手册 |

### WSL 环境注意事项

- WSL 默认只分配宿主机一半内存，建议在 `C:\Users\<用户名>\.wslconfig` 中配置 `memory=24GB`
- `wsl --shutdown` 后所有服务需要重新启动
- 如果 Docker daemon 配置了 HTTP 代理（如 `systemd` proxy），容器会继承代理设置，导致 Milvus 内部组件（etcd、minio）之间通信失败（502 Bad Gateway）。`docker-compose.milvus.yml` 中已为所有服务设置 `no_proxy="*"` 解决此问题
- Milvus v2.5.4 在某些 WSL2 内核版本下可能出现 SIGABRT 崩溃，确保使用 `docker-compose` 方式启动（含 `security_opt: seccomp:unconfined`）
- 如果旧 Milvus collection 是在 Sparse-BM25 支持前创建的，启动时会提示缺少 BM25 sparse schema；需要重置 collection 并重新索引文档：
  `/home/myc/miniconda3/envs/Research-Copilot/bin/python scripts/reindex_milvus_research_papers.py`

### CLI

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py --help
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py agent
```

### 环境自检

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python scripts/check_env.py
```

## 推荐阅读顺序

1. [apps/api/main.py](apps/api/main.py)
2. [apps/api/runtime.py](apps/api/runtime.py)
3. [apps/api/research_runtime.py](apps/api/research_runtime.py)
4. [web/src/App.tsx](web/src/App.tsx)
5. [web/src/api.ts](web/src/api.ts)
6. [web/src/components/ChatView.tsx](web/src/components/ChatView.tsx)
7. [web/src/components/MessageBubble.tsx](web/src/components/MessageBubble.tsx)
8. [runtime/research/supervisor_graph_runtime_core.py](runtime/research/supervisor_graph_runtime_core.py)
9. [runtime/research/context_builder.py](runtime/research/context_builder.py)
10. [services/research/literature_research_service.py](services/research/literature_research_service.py)
11. [tools/research/paper_search.py](tools/research/paper_search.py)
12. [rag_runtime/runtime.py](rag_runtime/runtime.py)
13. [memory/memory_manager.py](memory/memory_manager.py)
14. [retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py)

## 相关文档

- [docs/系统运行指南.md](docs/系统运行指南.md)
- [docs/Milvus学习文档.md](docs/Milvus学习文档.md)
- [docs/关键词匹配与稀疏检索详解.md](docs/关键词匹配与稀疏检索详解.md)
- [docs/rag流程详解.md](docs/rag流程详解.md)
- [docs/Zotero本地连接指南.md](docs/Zotero本地连接指南.md)
- [docs/CLI终端使用文档.md](docs/CLI终端使用文档.md)

## 多用户平台独立运行入口

新平台使用 `apps.platform.main:app`、`apps.worker` 与 `apps.scheduler`，通过 MySQL 持久化业务与任务状态。本机启动、邮箱验证、停机备份和空环境恢复步骤见 [平台运行手册](docs/platform-runbook.md)，决策见 [部署 ADR](docs/platform-adr.md)，已执行与未执行范围见 [部署验证记录](docs/platform-validation.md)。本机离线演示与脚本校验不代表公网生产或真实模型质量验证完成。

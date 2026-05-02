# Research-Copilot

`Research-Copilot`是一个面向科研调研工作区的多智能体系统。它把前端研究工作台、高层 research 编排、底层 RAG 执行、CLI 和本地持久化放在同一个仓库里。

## 当前代码主线

当前项目最值得先记住的是三层边界：

- `services/research/`
  高层 research 业务服务层，负责 conversation、task、paper pool、import job、collection QA、workspace、report 和持久化收口。
- `runtime/research/`
  Supervisor Graph 运行时，负责构建 `ResearchAgentToolContext`、技能匹配、manager 决策、specialist agent 调度和最终响应聚合。
- `tools/research/`
  研究领域能力单元，负责 query planning、paper ranking、survey writing、paper import/search、QA routing、Zotero search 等可复用业务能力。
- `rag_runtime/`
  底层 tool-first 执行层，负责 `parse / index / retrieve / answer / chart understanding`。

当前活跃前端工作区（`web/` 目录）：

- 技术栈：Next.js 15 (App Router) + React 18 + TypeScript + Tailwind CSS
- 应用入口：`web/app/layout.tsx` + `web/app/page.tsx`
- 主组件：`web/src/App.tsx`
- API 封装：`web/src/api.ts`（通过 Next.js rewrites 代理到 FastAPI）
- 图片服务：`web/app/api/files/[...path]/route.ts`（Next.js Route Handler）
- 聊天视图：`web/src/components/ChatView.tsx`
- 消息气泡：`web/src/components/MessageBubble.tsx`（含内联图片渲染）

### 前端设计

前端采用类似 ChatGPT / Claude 的对话式布局：

- **主区域**：居中的对话线程，占满屏幕高度
- **输入栏**：底部固定，支持检索/问答模式切换
- **侧边栏**：默认收起，包含会话列表
- **内联图片**：自动检测消息中的图片路径并渲染为 `<img>` 标签
- **决策轨迹**：Agent 活动和 Trace 默认展开显示

核心前端组件（`web/src/components/`）：

| 组件 | 职责 |
|---|---|
| `ChatView` | 聊天视图：消息过滤、发送、task_id 跟踪 |
| `MessageBubble` | 消息气泡：Markdown 渲染、内联图片、论文卡片、决策轨迹 |
| `InputBar` | 底部输入栏 |
| `PaperCard` | 论文卡片 |
| `Sidebar` | 侧边栏会话列表 |
| `WelcomeScreen` | 空状态欢迎页 |


## 架构概览

```text
Browser (http://localhost:3000)
-> Next.js (web/, App Router)
-> /api/backend/* rewrites -> FastAPI (http://127.0.0.1:8000)
-> /api/files/* -> Route Handler -> .data/storage/
-> LiteratureResearchService
-> ResearchSupervisorGraphRuntime
-> runtime/research context builder / dispatcher / result aggregator
-> specialist agents / research services
-> RagRuntime
-> DocumentTools / RetrievalTools / AnswerTools / ChartTools
-> Milvus (vector) / Neo4j (graph) / session memory / local persistence
```

## 当前主能力

- `research discovery`
  根据研究主题搜索 arXiv、OpenAlex、Semantic Scholar、IEEE 等学术源，生成候选论文池和研究草稿。
- `paper import`
  对选中的论文执行 `download -> parse -> Zotero sync -> embedding index -> graph index`。索引阶段有 `RESEARCH_IMPORT_INDEX_TIMEOUT_SECONDS` 超时保护；索引超时会保留已解析/已同步的论文，不再把整次导入判为失败。
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

默认配置下：

- `SESSION_MEMORY_PROVIDER=sqlite`
  底层 `GraphSessionMemory` 默认写入 `RESEARCH_SQLITE_DB_PATH`。
- `LONG_TERM_MEMORY_PROVIDER=sqlite`
  runtime profile 显示为 sqlite；research service 内部的长期记忆仍由 `MemoryManager` 和本地 JSON store 承载业务画像/论文知识。
- `VECTOR_STORE_PROVIDER=milvus`
- `GRAPH_STORE_PROVIDER=memory`

## 主要目录

- `web/`
  当前活跃前端（Next.js 15 + React 18 + Tailwind），ChatGPT 风格对话式布局
- `apps/api/`
  FastAPI 入口、依赖注入、runtime 装配和 HTTP 路由
- `services/research/`
  research 主业务 service、paper operations、QA router 和 workspace/report/conversation 持久化
- `runtime/research/`
  高层 Supervisor Graph runtime、context builder、unified action adapters、result aggregator 和 response formatter
- `agents/`
  高层 specialist agents（10 个：Supervisor、Writer、PreferenceMemory、Knowledge、LiteratureScout、ChartAnalysis、GeneralAnswer、PaperAnalysis、ResearchQA、ResearchDocument）
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
  数据持久化后端（JSON 文件 / SQLite），通过 `StorageBackend` 协议抽象
- `observability/`
  轻量级进程内指标收集（counters、histograms）、LLM 调用自动埋点
- `security/`
  敏感数据脱敏（API key、token、密码、连接串等 11 种模式）
- `memory/`
  research 侧工作记忆、会话记忆、长期记忆、用户画像记忆和论文知识记忆
- `reasoning/`
  可复用推理策略层（CoT / PlanAndSolve / ReAct）
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

## 当前入口

### Research API

- `POST /research/agent`
  当前统一主入口
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
  `VECTOR_STORE_PROVIDER`、`GRAPH_STORE_PROVIDER`、`SESSION_MEMORY_PROVIDER`
- external store config
  `MILVUS_URI`、`NEO4J_URI`
- research persistence
  `RESEARCH_STORAGE_ROOT`、`RESEARCH_RESET_ON_STARTUP`
- long-term memory
  `LONG_TERM_MEMORY_PROVIDER`
- uploads
  `UPLOAD_DIR`、`UPLOAD_MAX_BYTES`

当前默认值中比较关键的几点：

- `APP_ENV=local` 时会暴露 `/uploads/*`
- `UPLOAD_MAX_BYTES` 默认 `25 MiB`
- `VECTOR_STORE_PROVIDER` 默认 `milvus`
- `MILVUS_URI` 代码默认 `http://localhost:19530`；仓库当前 `.env` 和 compose 映射常用 `http://localhost:19531`
- `GRAPH_STORE_PROVIDER` 代码默认 `memory`，仓库当前 `.env` 中可配置为 `neo4j`
- `SESSION_MEMORY_PROVIDER` 默认 `sqlite`
- `LONG_TERM_MEMORY_PROVIDER` 默认 `sqlite`
- `RESEARCH_IMPORT_INDEX_TIMEOUT_SECONDS` 默认 `300`
- `STORAGE_PROVIDER` 默认 `json`（可切换为 `sqlite`）
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

项目实现了标准化的技能系统，支持内置技能和外部社区技能的即插即用。

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
| `core/skill_matcher.py` | SkillMatcher — trigger 正则匹配 + tag/description 关键词匹配 |
| `core/skill_validator.py` | SkillValidator — prompt injection 检测、路径安全、大小限制 |

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

## 本地启动

### 完整启动顺序（必须按顺序执行）

#### 第一步：基础设施

```bash
# 1. 启动 Milvus（Docker Compose，含 etcd + minio + milvus）
cd /home/myc/Research-Copilot
docker-compose -f docker-compose.milvus.yml up -d
# 等待端口 19531 就绪（约 15-30 秒）
# 检查：curl -s http://localhost:9092/healthz  →  应返回 "OK"

# 2. 启动 Neo4j（Docker 容器，数据挂载在 .data/neo4j/）
docker start kepler-neo4j
# 等待端口 7687 就绪（约 10-15 秒）
# 检查：docker exec kepler-neo4j cypher-shell -u neo4j -p neo4j1234 'RETURN 1'

# 3. 启动 Zotero Bridge（WSL 环境，需要 Windows 上先打开 Zotero 桌面程序）
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
  --reload-dir memory --reload-dir retrieval --reload-dir domain
# 等待日志出现 "Application startup complete"
# 如果卡在 "Waiting for application startup"，说明 Milvus 或 Neo4j 没启动
# 注意：必须用 --reload-dir 限制监视目录，否则 watchfiles 会因 web/node_modules 文件过多而崩溃
```

#### 第三步：前端

```bash
cd /home/myc/Research-Copilot/web
npm run dev
# 访问 http://localhost:3000
```

### 端口一览

| 服务 | 端口 | 启动方式 |
|------|------|------|
| Milvus | 19531 (映射到容器内 19530) | `docker-compose -f docker-compose.milvus.yml up -d`（在 Research-Copilot 目录下） |
| Neo4j | 7474/7687 | `docker start kepler-neo4j`（Docker 容器，数据在 `.data/neo4j/`） |
| Zotero Bridge | 23119 | `bash scripts/wsl_zotero_bridge.sh start` |
| 后端 (FastAPI) | 8000 | `uvicorn apps.api.main:app` |
| 前端 (Next.js) | 3000 | `cd web && npm run dev` |

### WSL 环境注意事项

- WSL 默认只分配宿主机一半内存，建议在 `C:\Users\<用户名>\.wslconfig` 中配置 `memory=24GB`
- `wsl --shutdown` 后所有服务需要重新启动
- 如果 Docker daemon 配置了 HTTP 代理（如 `systemd` proxy），容器会继承代理设置，导致 Milvus 内部组件（etcd、minio）之间通信失败（502 Bad Gateway）。`docker-compose.milvus.yml` 中已为所有服务设置 `no_proxy="*"` 解决此问题
- Milvus v2.5.4 在某些 WSL2 内核版本下可能出现 SIGABRT 崩溃，确保使用 `docker-compose` 方式启动（含 `security_opt: seccomp:unconfined`）

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
- [docs/系统完整运行流程说明.md](docs/系统完整运行流程说明.md)
- [docs/Milvus学习文档.md](docs/Milvus学习文档.md)
- [docs/rag流程详解.md](docs/rag流程详解.md)
- [docs/Zotero本地连接指南.md](docs/Zotero本地连接指南.md)
- [docs/当前项目结构图.md](docs/当前项目结构图.md)
- [docs/CLI终端使用文档.md](docs/CLI终端使用文档.md)

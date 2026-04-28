# Kepler

`Kepler`（原 Research-Copilot）是一个面向科研调研工作区的多智能体系统。它把前端研究工作台、高层 research 编排、底层 RAG 执行、CLI 和本地持久化放在同一个仓库里。

## 当前代码主线

当前项目最值得先记住的是两层运行时：

- `services/research/`
  高层 research 业务层，负责 conversation、task、paper pool、import job、collection QA、workspace、report 和 supervisor orchestration。
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
-> specialist agents / research services
-> RagRuntime
-> DocumentTools / RetrievalTools / AnswerTools / ChartTools
-> Milvus (vector) / Neo4j (graph) / session memory / local persistence
```

## 当前主能力

- `research discovery`
  根据研究主题搜索 arXiv、OpenAlex、Semantic Scholar、IEEE 等学术源，生成候选论文池和研究草稿。
- `paper import`
  对选中的论文执行 `download -> parse -> graph extraction -> embedding index -> graph index`。
- `collection QA`
  基于已导入论文集合做混合检索和 grounded answer，必要时下钻到单文档或图表链路。
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

- `SESSION_MEMORY_PROVIDER=auto`
  有 MySQL 配置时使用 MySQL，否则退回内存。
- `LONG_TERM_MEMORY_PROVIDER=json`
  CLI / research service 默认使用 JSON 文件型 long-term memory；也支持 Qdrant 和纯内存。
- `VECTOR_STORE_PROVIDER=milvus`
- `GRAPH_STORE_PROVIDER=memory`

## 主要目录

- `web/`
  当前活跃前端（Next.js 15 + React 18 + Tailwind），ChatGPT 风格对话式布局
- `app/`、`components/`、`lib/`
  旧 Next.js 前端（已不活跃）
- `apps/api/`
  FastAPI 入口、依赖注入、runtime 装配和 HTTP 路由
- `services/research/`
  research 主业务、workspace 持久化和 supervisor orchestration
- `agents/`
  高层 specialist agents（8 个：Supervisor、Writer、PreferenceMemory、Knowledge、Scout、ChartAnalysis、GeneralAnswer、PaperAnalysis）
- `rag_runtime/`
  底层 RAG runtime
- `tools/`
  文档、图表、检索、回答等稳定工具层
- `retrieval/`
  vector / sparse / graph / graph summary 检索与重排
- `adapters/`
  LLM、embedding、vector store、graph store 等基础设施适配
- `adapters/storage/`
  数据持久化后端（JSON 文件 / SQLite），通过 `StorageBackend` 协议抽象
- `observability/`
  轻量级进程内指标收集（counters、histograms）、LLM 调用自动埋点
- `security/`
  敏感数据脱敏（API key、token、密码、连接串等 11 种模式）
- `memory/`
  research 侧工作记忆、会话记忆、长期记忆、用户画像记忆和论文知识记忆
- `skills/`
  skill spec 定义、加载、注册和 research 领域技能扩展（含 16+ 研究技能类）
- `reasoning/`
  可复用推理策略层（CoT / PlanAndSolve / ReAct）
- `tooling/`
  工具注册 / 执行框架和 research function 规范
- `mcp/`
  MCP（Model Context Protocol）客户端注册和服务端暴露
- `evaluation/`
  评测框架（7 种 case 类型）、benchmark 构建、metrics 和数据集管理
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

### RAG / Document / Chart API

- `POST /documents/upload`
- `POST /documents/parse`
- `POST /documents/index`
- `POST /documents/ask`
- `POST /documents/ask/fused`
- `POST /charts/understand`
- `POST /charts/ask`
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
- `skills list|enable|disable|default`
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
  `MILVUS_URI`、`NEO4J_URI`、`MYSQL_*`、`POSTGRES_DSN`
- research persistence
  `RESEARCH_STORAGE_ROOT`、`RESEARCH_RESET_ON_STARTUP`
- long-term memory
  `LONG_TERM_MEMORY_PROVIDER`、`QDRANT_PATH`、`QDRANT_COLLECTION_NAME`
- uploads
  `UPLOAD_DIR`、`UPLOAD_MAX_BYTES`

当前默认值中比较关键的几点：

- `APP_ENV=local` 时会暴露 `/uploads/*`
- `UPLOAD_MAX_BYTES` 默认 `25 MiB`
- `VECTOR_STORE_PROVIDER` 默认 `milvus`
- `GRAPH_STORE_PROVIDER` 默认 `neo4j`
- `SESSION_MEMORY_PROVIDER` 默认 `auto`
- `LONG_TERM_MEMORY_PROVIDER` 默认 `json`
- `STORAGE_PROVIDER` 默认 `json`（可切换为 `sqlite`）
- `CORS_ALLOW_ORIGINS` 默认允许 localhost:3000/3001
- `RATE_LIMIT_MAX_REQUESTS` 默认 60 次/分钟/IP
- `JSON_LOG_FORMAT` 默认 `false`，设为 `true` 启用结构化 JSON 日志

## 本地启动

### 完整启动顺序（必须按顺序执行）

#### 第一步：基础设施

```bash
# 1. 启动 Milvus（Docker）
docker start research-copilot-milvus
# 等待端口 19530 就绪（约 10-15 秒）
# 检查：ss -tlnp | grep 19530

# 2. 启动 Neo4j
export JAVA_HOME=/home/myc/miniconda3/envs/Research-Copilot/lib/jvm
/home/myc/neo4j/bin/neo4j start
# 等待端口 7687 就绪
# 检查：ss -tlnp | grep 7687

# 3. 启动 Zotero Bridge（WSL 环境，需要 Windows 上先打开 Zotero 桌面程序）
bash scripts/wsl_zotero_bridge.sh start
# 检查：ss -tlnp | grep 23119
```

#### 第二步：后端

```bash
cd /home/myc/Kepler
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m uvicorn apps.api.main:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir apps --reload-dir services --reload-dir rag_runtime \
  --reload-dir agents --reload-dir tools --reload-dir adapters \
  --reload-dir memory --reload-dir retrieval --reload-dir skills --reload-dir domain
# 等待日志出现 "Application startup complete"
# 如果卡在 "Waiting for application startup"，说明 Milvus 或 Neo4j 没启动
# 注意：必须用 --reload-dir 限制监视目录，否则 watchfiles 会因 web/node_modules 文件过多而崩溃
```

#### 第三步：前端

```bash
cd /home/myc/Kepler/web
npm run dev
# 访问 http://localhost:3000
```

### 端口一览

| 服务 | 端口 | 启动方式 |
|------|------|------|
| Milvus | 19530 | `docker start research-copilot-milvus` |
| Neo4j | 7474/7687 | `/home/myc/neo4j/bin/neo4j start` |
| Zotero Bridge | 23119 | `bash scripts/wsl_zotero_bridge.sh start` |
| 后端 (FastAPI) | 8000 | `uvicorn apps.api.main:app` |
| 前端 (Next.js) | 3000 | `cd web && npm run dev` |

### WSL 环境注意事项

- WSL 默认只分配宿主机一半内存，建议在 `C:\Users\<用户名>\.wslconfig` 中配置 `memory=24GB`
- `wsl --shutdown` 后所有服务需要重新启动
- Milvus 在内存不足时会因 etcd 连接丢失反复崩溃

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
8. [services/research/research_supervisor_graph_runtime_core.py](services/research/research_supervisor_graph_runtime_core.py)
9. [services/research/literature_research_service.py](services/research/literature_research_service.py)
10. [services/research/research_function_service.py](services/research/research_function_service.py)
11. [rag_runtime/runtime.py](rag_runtime/runtime.py)
12. [memory/memory_manager.py](memory/memory_manager.py)
13. [retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py)
14. [skills/research/__init__.py](skills/research/__init__.py)

## 相关文档

- [系统运行指南.md](系统运行指南.md)
- [系统完整运行流程说明.md](系统完整运行流程说明.md)
- [项目学习和使用文档.md](项目学习和使用文档.md)
- [系统学习与面试指南.md](系统学习与面试指南.md)
- [Milvus学习文档.md](Milvus学习文档.md)
- [MySQL数据库使用说明.md](MySQL数据库使用说明.md)
- [docs/当前项目结构图.md](docs/当前项目结构图.md)
- [docs/CLI终端使用文档.md](docs/CLI终端使用文档.md)

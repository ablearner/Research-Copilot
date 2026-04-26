# Research-Copilot

`Research-Copilot` 是一个面向科研调研工作区的多智能体系统。它把前端研究工作台、高层 research 编排、底层 RAG 执行、CLI 和本地持久化放在同一个仓库里。

## 当前代码主线

当前项目最值得先记住的是两层运行时：

- `services/research/`
  高层 research 业务层，负责 conversation、task、paper pool、import job、collection QA、workspace、report 和 supervisor orchestration。
- `rag_runtime/`
  底层 tool-first 执行层，负责 `parse / index / retrieve / answer / chart understanding`。

前端工作区对应：

- 页面入口：`app/page.tsx`
- FastAPI 代理：`app/api/backend/[...path]/route.ts`
- 状态控制：`lib/use-literature-research-controller.ts`
- controller 拆分：`lib/literature-research-controller/{session,runtime,import-flow,qa-figure-flow,todo-flow,derived}.ts`

### 前端设计

前端采用类似 ChatGPT / Claude 的对话式布局：

- **主区域**：居中的对话线程（`max-w-3xl`），占满屏幕高度
- **输入栏**：底部固定，自适应高度的 textarea + 模式切换（检索/问答）
- **侧边栏**：默认收起，点击左上角图标展开，包含会话列表、检索设置、论文导入、待办等面板
- **视觉风格**：灰色中性色调，蓝色点缀，`rounded-lg` 圆角，无渐变/毛玻璃/装饰阴影

核心前端组件：

| 组件 | 职责 |
|---|---|
| `LiteratureResearchPanel` | 主布局 — 侧边栏 + 对话区 + 输入栏 |
| `ResearchComposer` | 底部输入栏，支持检索/问答模式切换 |
| `ResearchConversationMessage` | 对话气泡、论文卡片、trace 展示 |
| `ResearchThreadPreamble` | 空状态欢迎页 + 建议卡片 |
| `ResearchWorkspaceSidebar` | 右侧工作区面板（任务摘要、论文导入、分析） |
| `ResearchSidebarSections` | 左侧会话列表和检索设置 |
| `ResearchWorkspaceResults` | 综述、对比表、推荐结果展示 |
| `ResearchThreadArtifacts` | 导入结果、候选论文池、QA 回答 |

## 架构概览

```text
Browser
-> Next.js workspace
-> /api/backend/*
-> FastAPI routers
-> LiteratureResearchService
-> ResearchSupervisorGraphRuntime
-> specialist agents / research services
-> RagRuntime
-> DocumentTools / RetrievalTools / AnswerTools / ChartTools
-> vector store / graph store / session memory / local persistence
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

- `app/`、`components/`、`lib/`
  前端研究工作区（ChatGPT 风格对话式布局）、状态控制和后端代理
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
  评测脚本、benchmark 构建、metrics 和数据集管理
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
- `GET /health`

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
- `GRAPH_STORE_PROVIDER` 默认 `memory`
- `SESSION_MEMORY_PROVIDER` 默认 `auto`
- `LONG_TERM_MEMORY_PROVIDER` 默认 `json`

## 本地启动

### 1. 后端

```bash
/bin/bash scripts/run_api_dev.sh
```

实际启动：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. 前端

```bash
npm run dev
```

### 3. CLI

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py --help
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py agent
```

### 4. 环境自检

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python scripts/check_env.py
```

## 推荐阅读顺序

1. [apps/api/main.py](apps/api/main.py)
2. [apps/api/runtime.py](apps/api/runtime.py)
3. [apps/api/research_runtime.py](apps/api/research_runtime.py)
4. [app/api/backend/[...path]/route.ts](app/api/backend/[...path]/route.ts)
5. [components/LiteratureResearchPanel.tsx](components/LiteratureResearchPanel.tsx)
6. [lib/use-literature-research-controller.ts](lib/use-literature-research-controller.ts)
7. [services/research/research_supervisor_graph_runtime_core.py](services/research/research_supervisor_graph_runtime_core.py)
8. [services/research/literature_research_service.py](services/research/literature_research_service.py)
9. [services/research/research_function_service.py](services/research/research_function_service.py)
10. [rag_runtime/runtime.py](rag_runtime/runtime.py)
11. [memory/memory_manager.py](memory/memory_manager.py)
12. [retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py)
13. [skills/research/__init__.py](skills/research/__init__.py)

## 相关文档

- [系统运行指南.md](系统运行指南.md)
- [系统完整运行流程说明.md](系统完整运行流程说明.md)
- [项目学习和使用文档.md](项目学习和使用文档.md)
- [系统学习与面试指南.md](系统学习与面试指南.md)
- [Milvus学习文档.md](Milvus学习文档.md)
- [MySQL数据库使用说明.md](MySQL数据库使用说明.md)
- [docs/当前项目结构图.md](docs/当前项目结构图.md)
- [docs/CLI终端使用文档.md](docs/CLI终端使用文档.md)

# CLI终端使用文档

本文档基于当前代码实现整理，覆盖 [apps/cli.py](../apps/cli.py)、[sdk/client.py](../sdk/client.py) 和 [sdk/runtime_profile.py](../sdk/runtime_profile.py) 中已经存在的 CLI 能力。

## 1. 入口与安装

当前最稳妥的调用方式是直接运行 Python 入口：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py --help
```

如果你已经把项目安装成 console script，则还有两个别名：

- `kepler`
- `research-copilot`

它们都映射到：

- `apps.cli:main`

## 2. CLI 能做什么

当前 CLI 不是单一调试脚本，而是一个“本地研究终端 + 运行时管理工具”，主要覆盖四类能力：

- 会话管理
- 用户画像管理
- runtime 配置管理
- 交互式研究终端

## 3. 一级命令

运行：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py --help
```

当前一级命令为：

- `list-conversations`
- `show-profile`
- `doctor`
- `status`
- `trajectory`
- `update-profile`
- `models`
- `skills`
- `plugins`
- `agent`

## 4. 常用非交互命令

### 4.1 查看已有研究会话

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py list-conversations
```

### 4.2 查看用户长期研究偏好

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py show-profile
```

这里展示的是 research 侧 long-term memory 中的 `user_profile`，不是底层 runtime session memory。

### 4.3 更新用户画像

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py update-profile \
  --topic "GraphRAG" \
  --source arxiv \
  --source openalex \
  --keyword groundedness \
  --reasoning-style rigorous \
  --note "prefer evidence-backed answers"
```

当前支持：

- `--topic`
- `--source`
- `--keyword`
- `--reasoning-style`
- `--note`

### 4.4 运行环境诊断

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py doctor
```

它会输出一份 JSON 检查结果，主要用于快速确认：

- 本地存储目录
- Milvus / Neo4j / MySQL 连通性
- API key 是否存在

### 4.5 查看运行时状态

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py status
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py status --conversation-id <conversation_id>
```

不带 `--conversation-id` 时，主要查看 runtime 模型、skill 和 plugin 状态。

带 `--conversation-id` 时，还会额外展示当前会话摘要。

### 4.6 查看会话轨迹

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py trajectory <conversation_id>
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py trajectory <conversation_id> --messages 20 --events 20
```

这个命令适合排查：

- 最近消息
- 最近事件
- 压缩后的上下文轨迹

## 5. 运行时配置命令

### 5.1 查看当前模型配置

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py models show
```

### 5.2 设置模型覆盖

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py models set \
  --llm-provider openai \
  --llm-model gpt-5.4-mini \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large
```

还支持：

- `--chart-vision-provider`
- `--chart-vision-model`

这些配置会写到本地 runtime profile，而不是直接改 `.env`。

### 5.3 查看和切换 skill

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py skills list
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py skills enable research_report
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py skills disable financial_report
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py skills default research_report
```

### 5.4 查看和切换 plugin

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py plugins list
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py plugins enable zotero_local_mcp
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py plugins disable terminal_agent
```

当前内置 plugin 名称来自 `apps/cli.py` 与 `sdk/client.py`：

- `academic_search`
- `zotero_local_mcp`
- `local_code_execution`
- `trajectory_inspector`
- `terminal_agent`

## 6. 交互式终端：agent

进入交互终端：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py agent
```

常用启动参数：

```bash
/home/myc/miniconda3/envs/Research-Copilot/bin/python apps/cli.py agent \
  --topic "graph rag for literature review" \
  --mode auto \
  --days-back 180 \
  --max-papers 12 \
  --source arxiv \
  --source openalex
```

参数说明：

- `--conversation-id`
- `--topic`
- `--mode`
  - `auto`
  - `research`
  - `qa`
  - `import`
  - `document`
  - `chart`
- `--days-back`
- `--max-papers`
- `--source`

如果没有传 `--conversation-id`，CLI 会先创建一个新会话。

## 7. agent 模式下的 slash commands

当前实现支持：

- `/help`
- `/status`
- `/profile` — 查看长期偏好画像
- `/profile remove <topic>` — 移除某个兴趣主题
- `/profile clear` — 重置偏好画像
- `/preferences` — `/profile` 的别名
- `/events`
- `/trajectory`
- `/sources`
- `/sources set <sources...>`
- `/papers`
- `/papers filter <keyword>`
- `/papers show <index|paper_id|title>`
- `/papers abstract <index|paper_id|title>`
- `/select <ids|mustread|ingest>`
- `/select clear`
- `/import selected|all|mustread|ingest`
- `/zotero selected [collection]`
- `/figure`
- `/open-figure`
- `/new [topic]`
- `/clear` — 清除当前 session memory，开始新会话
- `/use <conversation_id>`
- `/exit`

### 7.1 论文池查看与筛选

```text
/papers
/papers filter graph
/papers show 3
/papers abstract 3
```

### 7.2 选中论文并导入

```text
/select 1 2 3
/import selected
```

### 7.3 切换搜索源

```text
/sources
/sources set arxiv openalex semantic_scholar ieee zotero
```

### 7.4 图表锚点

如果本轮回答关联到了 figure，CLI 会缓存最近一次 figure 信息。你可以用：

```text
/figure
/open-figure
```

### 7.5 偏好画像管理

```text
/profile
/profile remove GraphRAG
/profile clear
```

`/profile` 展示从历史研究对话中学到的长期兴趣主题、权重和偏好。
`/profile remove <topic>` 可以移除噪声主题。
`/profile clear` 重置整个偏好画像。

### 7.6 Zotero 同步

```text
/zotero selected
/zotero selected My Collection
```

这依赖本地 Zotero MCP 配置可用。

## 8. 数据存放位置

CLI 不会把状态只放在内存里，几个关键文件会落到本地存储目录：

- runtime profile
  `<research_storage_root>/cli/runtime_profile.json`
- research conversation、report、task、message
  由 `ResearchReportService` 写入 `research_storage_root`
- paper knowledge / session memory
  写入 `.data/research` 下的对应目录
- observability
  `.data/research/observability/`

## 9. 与 HTTP API 的关系

这套 CLI 不是简单 `curl` 包装，它主要直接调用本地 SDK 和 service：

- CLI 入口：[apps/cli.py](../apps/cli.py)
- SDK 封装：[sdk/client.py](../sdk/client.py)
- 研究服务：[services/research/literature_research_service.py](../services/research/literature_research_service.py)

所以它适合：

- 本机开发
- 调试和演示
- 不开前端时快速跑 research 工作流

## 10. 先看哪些文件

如果你准备继续扩展 CLI，建议按这个顺序读：

1. [apps/cli.py](../apps/cli.py)
2. [sdk/client.py](../sdk/client.py)
3. [sdk/runtime_profile.py](../sdk/runtime_profile.py)
4. [memory/memory_manager.py](../memory/memory_manager.py)
5. [services/research/literature_research_service.py](../services/research/literature_research_service.py)

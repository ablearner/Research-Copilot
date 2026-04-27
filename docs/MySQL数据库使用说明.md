# Research-Copilot MySQL Session Memory 使用说明

MySQL 在当前项目里不是主数据库，而是底层 `GraphSessionMemory` 的可选持久化后端。

## 1. 当前角色

MySQL 只负责保存 `rag_runtime/memory.py` 里的 `SessionMemorySnapshot`，也就是：

- `current_document_id`
- `last_retrieval_summary`
- `last_answer_summary`
- `current_task_intent`
- `metadata_json`
- `updated_at`

它不负责：

- research conversation / task / report / job 持久化
- 向量检索
- 图谱检索
- research 侧 `WorkingMemory / SessionMemory / LongTermMemory`

## 2. 什么场景需要它

如果你只是本地调试，完全可以直接使用内存型 session memory。

建议启用 MySQL 的场景：

- 想在后端重启后保留底层问答链路的 session snapshot
- 想验证 `SESSION_MEMORY_PROVIDER=auto/mysql` 的真实回写路径
- 想观察图表问答和文档问答的结构化 session state

## 3. 关键配置

```env
SESSION_MEMORY_PROVIDER=auto
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=...
MYSQL_DATABASE=ecs
MYSQL_CHARSET=utf8mb4
MYSQL_SESSION_MEMORY_TABLE=session_memory
```

实际装配逻辑在 [apps/api/runtime.py](apps/api/runtime.py)：

- `SESSION_MEMORY_PROVIDER=memory|local|inmemory` 时使用内存
- `SESSION_MEMORY_PROVIDER=auto` 时，如果 `MYSQL_HOST`、`MYSQL_USER`、`MYSQL_DATABASE` 都存在，就使用 MySQL
- `SESSION_MEMORY_PROVIDER=mysql` 时强制要求 MySQL 配置齐全

## 4. 当前系统里的位置

后端会构建：

```text
Settings
-> _build_session_memory()
-> GraphSessionMemory(
     MySQLSessionMemoryStore(...) | InMemorySessionMemoryStore(...)
   )
-> RagRuntime.session_memory
```

它主要用于：

- 给下一轮 prompt 注入结构化 session context
- 保存最近一次 retrieval/answer 的摘要
- 保存图表问答和 research turn 的简化历史

## 5. 表结构与初始化

`MySQLSessionMemoryStore.ensure_schema()` 会在第一次真正建立连接时自动建表，通常不需要手工建表。

表字段包括：

- `session_id`
- `current_document_id`
- `last_retrieval_summary`
- `last_answer_summary`
- `current_task_intent`
- `metadata_json`
- `updated_at`

## 6. 验证方式

先确认 MySQL 可连：

```bash
mysql -h 127.0.0.1 -P 3306 -u root -p
```

然后：

1. 启动后端
2. 走一轮 research、document 或 chart QA
3. 检查 `MYSQL_SESSION_MEMORY_TABLE` 中是否有写入

## 7. 当前结论

MySQL 在 `Research-Copilot` 里只是底层 session memory 的增强项，不是主链路必需品。

如果你想要最小运行环境，可以直接使用：

- `SESSION_MEMORY_PROVIDER=memory`
- `VECTOR_STORE_PROVIDER=memory`
- `GRAPH_STORE_PROVIDER=memory`

如果你想验证更接近真实部署的链路，再逐步切到：

- MySQL session memory
- Milvus vector store
- Neo4j graph store

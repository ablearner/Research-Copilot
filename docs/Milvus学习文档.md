# Research-Copilot Milvus 使用说明

这份文档只讲当前项目里 Milvus 的角色、装配方式和排障重点。

## 1. Milvus 当前负责什么

在 `Research-Copilot` 里，Milvus 负责底层向量检索。

它保存的主要是多模态 embedding 记录，例如：

- 文本块 embedding
- 页面级 embedding
- 图表相关 embedding

它不负责：

- 图谱关系查询
- 会话记忆
- research task / report / conversation 持久化

## 2. 当前整体分工

- `Milvus`
  向量检索
- `Neo4j`
  图谱检索
- `MySQL`
  可选 runtime session memory
- 本地文件 / `.data/research`
  research conversation、report、job、paper knowledge 等持久化

## 3. 代码里的装配位置

Milvus 的装配在 [apps/api/runtime.py](apps/api/runtime.py)：

- `_build_vector_store()`
- 当 `VECTOR_STORE_PROVIDER=milvus|zilliz` 时，构建 `MilvusVectorStore`

默认配置来自 [core/config.py](core/config.py)：

- `VECTOR_STORE_PROVIDER=milvus`
- `MILVUS_URI=http://localhost:19530`
- `MILVUS_COLLECTION_NAME=multimodal_embeddings`

## 4. 关键配置

```env
VECTOR_STORE_PROVIDER=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_NAME=multimodal_embeddings
MILVUS_DIMENSION=
MILVUS_METRIC_TYPE=COSINE
MILVUS_INDEX_TYPE=HNSW
```

说明：

- `MILVUS_DIMENSION` 可以为空；运行时会按 embedding adapter 的向量维度处理
- 如果你显式配置了维度，就要和当前 embedding 模型输出一致

## 5. 当前系统里什么时候会写 Milvus

主要发生在导入链路：

```text
download PDF
-> parse document
-> embedding_index_service
-> vector_store.upsert_embeddings()
-> Milvus
```

也就是说，只有在文档被导入和建索引时，Milvus 才会真正写入。

## 6. 启动方式

如果你本地使用仓库里的 compose：

```bash
cd <repo-root>
docker compose -f docker-compose.milvus.yml up -d
```

之后确认 `MILVUS_URI` 指向的地址可达。

## 7. 常见排障

### 7.1 连不上 Milvus

先检查：

- Milvus 容器是否真的起来了
- `MILVUS_URI` 是否和实际端口一致
- 后端日志里是不是在启动阶段就卡在 vector store 初始化

### 7.2 维度不匹配

重点检查：

- 当前 embedding model 的实际输出维度
- `MILVUS_DIMENSION`

### 7.3 导入很慢

优先看：

- `EMBEDDING_TEXT_BATCH_SIZE`
- `RESEARCH_IMPORT_CONCURRENCY`
- embedding provider 本身的吞吐和网络

## 8. 当前结论

Milvus 在当前项目里的定位非常明确：只做向量检索，不承担图谱能力，也不承担任务状态或记忆系统。

如果你只是本地最小运行，也可以切到：

```env
VECTOR_STORE_PROVIDER=memory
```

这样底层会退回 [adapters/local_runtime.py](adapters/local_runtime.py) 里的 `InMemoryVectorStore`。

# 后端服务与 MySQL 性能报告

测量日期：2026-11-10（Asia/Shanghai）。原始数据：[优化前](../benchmarks/backend/before-index.json)、[优化后](../benchmarks/backend/latest.json)；脚本：[run.py](../benchmarks/backend/run.py)。两个文件保留每次延迟样本、错误、连接获取时间、进程资源和 EXPLAIN。

## 范围与环境

这是一组 Python Service + 真实 MySQL 的合成数据基准，**没有 HTTP、队列投递或模型执行**。延迟包括业务授权、事务、数据库查询、结果物化和连接获取，排除造数与预热。配置中模型调用为 0；没有使用 Mock 数据库。脚本创建的 LocalObjectStore 没有读写实际 PDF。

| 项目 | 实测配置 |
| --- | --- |
| 主机 | WSL2 / Linux 6.6.87.2，Intel i5-12600KF，16 个可见逻辑 CPU |
| 可见内存 | 24608944 KiB（约 23.47 GiB），不是独占硬件 |
| Python / SQLAlchemy / asyncmy | 3.11.15 / 2.0.49 / 0.2.18 |
| MySQL | 8.4.11，READ-COMMITTED，buffer pool 128 MiB，max_connections 151 |
| 客户端连接池 | size 5，max_overflow 10，timeout 30 秒，pre_ping 开启 |
| 合成数据 | 100000 条消息、10000 条文档及版本、10000 条终态任务 |
| 文件与队列 | 文件实际字节 0，数据集 Outbox 行数 0；不会触发真实导入 |
| 测量 | 每场景/并发档 200 次操作，预热 20 次；并发 1、10、30 |
| 对比方式 | 优化后 `--reuse` 优化前数据集，工作区和项目相同 |

场景为消息游标分页（最多 50 条）、项目文档列表（最多 100 条）与任务状态读取。此批结果没有测量任务提交吞吐。并发指有界协程数，不是注册用户或日活人数。每档仅 200 个样本、单轮实验，没有重复实验置信区间；短时间波动不能归因于某一优化。

## 文档列表优化前后

| 并发 | 前 P50 ms | 后 P50 ms | 前 P95 ms | 后 P95 ms | 前 P99 ms | 后 P99 ms | 前 ops/s | 后 ops/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 85.62 | 4.62 | 122.88 | 6.82 | 142.12 | 12.11 | 11.01 | 196.14 |
| 10 | 133.70 | 21.31 | 155.47 | 40.14 | 170.90 | 41.74 | 73.64 | 424.80 |
| 30 | 335.95 | 59.32 | 427.92 | 126.60 | 497.08 | 163.33 | 85.48 | 411.17 |

三档优化前后均为 0 错误。并发 1 的文档列表 P95 从 122.88 ms 降到 6.82 ms；并发 30 时仍有 126.60 ms 的 P95，说明增大并发没有线性提高吞吐。这里只报告本次服务调用结果，不换算为互联网在线容量。

### 从 EXPLAIN 定位问题

优化前的项目关联条件没有显式提供 `pp.workspace_id=:w`。虽然业务文档已限定工作区，优化器仍从 `project_papers` 的联合索引全扫描开始：`type=index`、估算 rows=10007，并出现 `Using temporary; Using filesort`。

改动包括显式工作区条件，以及迁移 `0006_document_indexes` 的两个索引：

```sql
CREATE INDEX ix_document_listing
    ON documents (workspace_id, deleted_at, created_at, id);
CREATE INDEX ix_document_version_latest
    ON document_versions (workspace_id, document_id, created_at, id);
```

优化后的已记录计划先以 `ix_document_listing` 对文档做 range 读取，再通过 `(workspace_id, project_id, paper_id)` 对关联表做 `eq_ref`。主查询不再出现临时表和 filesort；索引同时匹配作用域、删除标记与列表顺序。

这里不能把两个索引都描述成已命中：最新版本子查询在该次 EXPLAIN 中仍选择 `uq_document_version_owner`，仍有 `Using filesort`，估算 rows 从 1429 变为 1。数据集每文档仅一个版本且有 active_version，没有充分覆盖大量历史版本回退查询；`ix_document_version_latest` 的独立收益需要后续专门测量。两个源码/DDL 变化一起测试，也没有消融证明各自贡献。

## 优化后完整场景结果

| 场景 | 并发 | P50 ms | P95 ms | P99 ms | ops/s | 获取连接 P95 ms | 错误数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| message_cursor | 1 | 5.48 | 7.77 | 10.14 | 182.94 | 0.34 | 0 |
| message_cursor | 10 | 14.72 | 17.76 | 52.25 | 644.93 | 5.38 | 0 |
| message_cursor | 30 | 43.41 | 91.73 | 102.22 | 598.64 | 69.51 | 0 |
| document_list | 1 | 4.62 | 6.82 | 12.11 | 196.14 | 0.39 | 0 |
| document_list | 10 | 21.31 | 40.14 | 41.74 | 424.80 | 5.12 | 0 |
| document_list | 30 | 59.32 | 126.60 | 163.33 | 411.17 | 94.75 | 0 |
| task_status | 1 | 3.33 | 4.26 | 4.81 | 297.06 | 0.34 | 0 |
| task_status | 10 | 19.14 | 22.57 | 24.47 | 506.29 | 1.63 | 0 |
| task_status | 30 | 59.52 | 64.68 | 68.24 | 483.66 | 34.34 | 0 |

连接获取时间包含建连/pre_ping，并非纯连接池排队等待。文档列表并发 30 的连接获取 P95 为 94.75 ms；池最多 15 个连接，在 30 协程下出现等待是合理候选原因，不能只依据这一项就断言服务器 CPU 饱和。

优化后基准客户端累计峰值 RSS 为 83528 KiB（约 81.57 MiB）；文档列表并发 30 的客户端 CPU 为 0.508 秒，约单核 104.4%。这是 Python 客户端进程的统计，不是 MySQL 容器资源或整机利用率。该批数据没有独立数据库 CPU/RSS 采样，也没有模型超时与端到端研究延迟。

## 复现

先按运行手册启动专用 MySQL 并迁移到 head。不要覆盖原始结果文件。以下命令从仓库根目录运行，解释器使用已安装项目依赖的 Python 3.11：

```bash
python -m benchmarks.backend.run   --messages 100000 --documents 10000 --tasks 10000   --requests 200 --warmup 20 --concurrency 1 10 30   --output /tmp/platform-benchmark-current.json
```

已有原数据集且未删除时可以复用：

```bash
python -m benchmarks.backend.run   --reuse benchmarks/backend/before-index.json   --messages 100000 --documents 10000 --tasks 10000   --requests 200 --warmup 20 --concurrency 1 10 30   --output /tmp/platform-benchmark-reused.json
```

新的运行只重测当前实现，不能通过 `--reuse` 自动回到旧 SQL/索引。重建优化前条件应使用隔离测试库与对应代码/迁移状态，不要降级正在使用的业务库。数据生成会持久写入合成业务记录，应为结果保留数据集 ID，避免反复造数混淆规模。

后续容量评估需要增加 HTTP 认证/代理成本、提交与恢复场景、重复采样、长期负载、真实对象/向量 provider 和少量真实模型调用。目前这些结果仅支持“发现并修复文档列表 SQL 查询瓶颈”的结论，不支持生产 SLA 或真实用户规模承诺。

## 补充：实际 HTTP 路径测量（2026-11-14）

原始记录：`benchmarks/backend/http-latest.json`；脚本：`python -m benchmarks.backend.http_run --help`。真实网络请求直连本机 Uvicorn `http://127.0.0.1:8000/api/v1`，经过 PAT 鉴权、服务层和 MySQL；没有 Next.js 代理、TLS 或模型调用。Worker/Scheduler 停止，提交后立即取消，临时 PAT 测后撤销。本次专用小 fixture 初始只有一个取消任务/用户消息，文档列表为空，提交场景随后增加40条消息及取消任务；**不是前述10万消息数据集，也不是容量压测**。

每场景每并发20次，无预热，并发1/2，共160个逻辑操作（提交取消各含两个HTTP请求，共200请求），错误0。API采用本地配置，限流关闭；限流正确性另有Redis集成测试。模型执行和固定延迟调度承载尚未测量。

| 场景 | 并发 | P50 ms | P95 ms | P99 ms | 成功操作/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| message_cursor | 1 | 7.39 | 9.38 | 21.66 | 121.08 |
| message_cursor | 2 | 9.27 | 10.66 | 11.97 | 207.42 |
| document_list | 1 | 7.96 | 14.67 | 14.69 | 111.53 |
| document_list | 2 | 9.84 | 10.81 | 11.03 | 202.18 |
| task_status | 1 | 7.67 | 8.90 | 9.35 | 126.48 |
| task_status | 2 | 9.58 | 10.85 | 11.00 | 202.22 |
| submit_cancel | 1 | 75.31 | 107.06 | 116.93 | 12.51 |
| submit_cancel | 2 | 119.17 | 138.64 | 159.37 | 16.37 |

`submit_cancel` 延迟包含取消请求；原始 JSON 中的 `submit_only` 单独记录提交响应时间。运行时需设置专用 fixture 的 `PLATFORM_BENCH_TOKEN`、`PLATFORM_BENCH_WORKSPACE_ID`、`PLATFORM_BENCH_PROJECT_ID`、`PLATFORM_BENCH_CONVERSATION_ID`、`PLATFORM_BENCH_TASK_ID`，停止 Worker/Scheduler 后传 `--workers-stopped`；不要把真实用户会话用作压测输入。默认5次可降低触发限流的概率，测试仍应在独立实例进行。

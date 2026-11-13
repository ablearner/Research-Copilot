# 平台验收记录

本文件区分实现、可复现验证与尚未完成的外部验收。当前交付是单机多用户科研 Agent 平台；MySQL 保存服务端业务事实，Agent 记忆保持独立 SQLite，检索与图谱为隔离的派生数据。不能将本机测试解释成公网商业上线或真实用户规模。

## 验证环境与命令

Python 3.11.15、Next.js 15.5.29、Node.js 20、MySQL 8.4.11、RabbitMQ 4.2、Redis 7.4、MinIO；专用 Compose project 为 `research-copilot-platform`。Python 环境位于 `/home/myc/miniconda3/envs/Research-Copilot`。依赖与启动见[运行手册](platform-runbook.md)。测试仅在专用库运行，先停止应用 Worker/Scheduler，避免它们消费故障用例的 Outbox。

```bash
python scripts/platform-local.py infra
python scripts/platform-local.py migrate
RUN_PLATFORM_INTEGRATION=1 python -m pytest -q -o faulthandler_timeout=60
python -m ruff check apps/platform apps/worker apps/scheduler services/platform \
  services/identity services/projects services/task_execution adapters/storage/mysql \
  adapters/object_store adapters/queue tests/unit/platform tests/integration/backend
PLATFORM_API_ORIGIN=http://127.0.0.1:8000 npm --prefix web run build
```

真实依赖集成测试在 `tests/integration/backend/`；默认不设开关会跳过这些测试。单元测试使用 SQLite 是契约验证，不替代 MySQL 锁、隔离级别和事务测试。PDF 使用真实解析子进程；研究链路使用现有 Supervisor 和确定性 local 模型、memory vector/graph，不使用付费模型。账号切换使用完整页面导航丢弃敏感状态；损坏记忆清单隔离并继续恢复，相关回归与图谱嵌套来源测试均保留。公开论文下载测试替换外部 HTTP 数据源，不能代表 arXiv 实时可用性。

最终回归与浏览器结果见下方“最终执行结果”；本轮执行日志保存在 `/tmp/research-platform-final-*.log`。`/tmp` 日志和本机数据不属于可移植交付，复现时应自行归档。

## 方案验收矩阵

| 要求 | 实现和证据 | 验证边界 |
| --- | --- | --- |
| 身份、工作区、项目及私有文件 | identity/business/papers API 测试，真实上传和浏览器双用户文件拒绝 | 个人工作区，无团队邀请 UI |
| 多用户检索隔离 | `test_scope.py`：越界命中、嵌套图谱来源、检索期间撤销、attempt 独立集合 | 真实 Milvus 单文档链路已验收；混合用户召回仍需专项验证 |
| 幂等提交、Outbox 确认间隙 | `test_task_mysql.py`、`test_queue_resilience.py` | 真实 MySQL/RabbitMQ；确认间隙以异常注入复现 |
| Worker 被终止后恢复、旧 epoch 拒绝 | 独立领取进程 SIGKILL、过期租约回收与迟到提交断言 | 测试缩短租约，不是生产恢复 SLA |
| 部分失败与阶段复用 | 实际 PDF/Supervisor 集成、成功下载阶段复用、失败项重试 | 任意 LangGraph 节点续跑不在当前实现内 |
| 取消、配额、项目删除 | MySQL 并发配额、取消终态、删除项目取消 queued/请求取消 running | 单 Worker 进程；禁止直接多主机扩容记忆写入 |
| 上传与发布一致性 | MinIO 签名 POST、大小/hash/PDF 校验、重复完成、活动版本发布 | 对象先写、MySQL 后发布；不可见孤儿通过对账清理 |
| 对话、报告、SSE 恢复 | 消息游标/最近消息、报告版本、`test_sse.py` 的 Last-Event-ID 补发/非法游标 | 浏览器刷新不等于完整网络故障测试；SSE 独立 API 用例验证 |
| Redis 故障 | 真实限流、不可用时写入关闭测试 | Redis 仅限流；SSE 直接读取 MySQL，不依赖通知 |
| 邮箱、凭证、管理入口 | Mailpit 验证/重置、PAT撤销、admin动作审计及指标测试 | 公网 SMTP/TLS 部署另验收 |
| 大项目资源选择 | 第 101 条论文/文档定向查询回归，API/Worker 共用过滤路径 | 默认列表最多100条；未提供大规模论文库完整浏览 UI |
| 历史数据迁移 | business/files migration 单元测试：dry run、重复执行、hash冲突、缺失文件 | 只对测试 fixture 演练，未迁移用户原始业务库 |
| 备份恢复 | 空卷恢复与全部业务表/实际文件指纹比较 | 同机隔离 Compose，非全新主机或公网灾备 |
| 性能 | 合成10万消息/1万文档/1万任务，优化前后原始结果 | Python service + MySQL；不代表HTTP/模型吞吐，见[报告](platform-performance.md) |

## 备份恢复的实测证据

2026-11-10 停止专用应用写入和依赖容器，备份 MySQL、MinIO、Worker 工件到 `.data/backup-rehearsal-20261110/`，恢复到独立 `research-copilot-restore` 空卷和 `.data/restore-worker`。恢复容器随后停止，原服务重新启动。没有覆盖原数据卷。

`platform-fingerprint.py` 逐表主键排序计算行内容摘要，并读取实际文件计算 SHA256；排除明确的 benchmark 虚拟文件元数据。`.data/original-fingerprint.json` 与 `.data/restored-fingerprint.json` 完全一致：**24 表、361089 行、5 个实际文件**。归档包含 SHA256 清单，`platform-backup.py verify` 再校验通过。该证据验证静态快照内容，不证明所有多模态绝对路径在换机器后仍有效。

备份为当时 schema 0005；当前数据库升级到 `0006_document_indexes`。恢复必须使用对应版本并显式升级。未测量端到端 RTO/RPO，不填写估算值充当指标。

## 仍需外部环境或人工完成的验收

- 邀请真实使用者试用并记录反馈；本轮未发送邀请，也没有真实用户规模数据。
- 全新主机依赖安装、公网 TLS/邮件/反向代理、真实论文集模型与检索效果验证。真实模型与 Milvus 的合成 PDF 链路已通过，见[真实 API 验收](platform-real-api-validation.md)。
- HTTP 容量和固定延迟模型调度压测；已补充小 fixture 的实际 HTTP 测量，但样本不能支持容量结论。
- 迁移到不同 Worker 路径时，解析 JSON 中页面图片的本地绝对路径需要重解析；目前未实现派生图片的私有下载代理。
- 本地旧 attempt 工件与异常退出遗留向量集合尚无自动保留期清理；对象存储孤儿对账由运维 CLI 执行，默认 dry run。

因此，方案 M1–M3 的本机核心闭环有工程实现和测试证据，M4 的上线/真实试用验收不能标为全部完成。求职演示可使用[演示脚本](platform-demo.md)，并说明上述边界。

## 最终执行结果

2026-11-14 最终执行：

- `RUN_PLATFORM_INTEGRATION=1 python -m pytest -q -o faulthandler_timeout=60`：**1029 passed，2 条上游弃用警告，160.60秒，退出0**。没有跳过真实后端依赖测试。
- 损坏记忆清单隔离文件名调整后，`tests/unit/platform/test_memory_commit.py` 再次 **2 passed**；新图谱来源用例先复现越界未拒绝，修复后 **4 passed**。
- 新平台 Python Ruff 检查、HTTP benchmark Ruff 检查、`git diff --check`：退出0。
- `NEXT_TELEMETRY_DISABLED=1 PLATFORM_API_ORIGIN=http://127.0.0.1:8000 npm --prefix web run build`：退出0，TypeScript/生产构建通过。一次未禁用遥测的构建已输出产物但进程等待外部连接，主动结束后使用此命令重新完成，不把前一次当作成功退出。
- 四进程启动器 `platform-local.py run --offline` 启动成功，真实浏览器 **2 passed（43.9秒）**：签名上传/解析/研究/刷新/双用户文件拒绝/项目重命名删除；无项目账号入口/创建PAT/同标签页切换用户后旧PAT不可见。退出启动器后所有四个子进程停止。
- 浏览器测试专用 `PLATFORM_REQUIRE_EMAIL_VERIFICATION=false`；邮箱验证由真实 Mailpit 集成用例覆盖。执行 `E2E_BASE_URL=http://localhost:3000 npx playwright test`；本机额外通过 `LD_LIBRARY_PATH=/tmp/research-playwright-libs/extracted/usr/lib/x86_64-linux-gnu` 提供 Chromium 所需 NSS/NSPR 库。
- HTTP benchmark：160个逻辑操作、200个实际请求，错误0；Worker/Scheduler停止，临时凭证已撤销。真实测量与限制见[性能报告](platform-performance.md)。
- 备份归档 SHA256 验证及恢复前后业务指纹重新比较通过；`platform-local.py migrate`、`doctor` 退出0，迁移 head为0006。

新增回归确实暴露了三项缺陷：第101条资源被列表截断误拒绝、嵌套图谱外部来源未校验、账号切换残留明文PAT；另有损坏清单阻断恢复。已修复并保留用例。浏览器失败基线包含测试专用凭证，未作为公开仓库附件提交；不得复用这些本机测试账号。

后续真实模型验收：DeepSeek、Embedding、VLM 共 23 次远程请求全部 HTTP 200，合成 PDF 的导入、文本/图片问答、报告持久化、异用户文件拒绝和应用重启恢复通过。修复后的完整回归 **1031 passed，2 warnings，160.47 秒，退出 0**。具体失败记录、修复、复现命令及边界见[真实 API 验收](platform-real-api-validation.md)。

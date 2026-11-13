# 多用户平台本机运行与恢复手册

此入口运行新平台 API、Worker、Scheduler 与平台前端；旧匿名 API 不在启动命令中。下面是单机本地演示配置，不能直接作为公网部署配置。使用仓库当前提交与 lock 文件保存部署版本；实际模型能力取决于 `.env` 中的 provider 配置。

## 前置条件与安装

需要 Python 3.11、Node.js 20+、npm、Docker 与 Docker Compose（v2 或 `docker-compose` v1），以及足够存放 PDF、MySQL、模型缓存和 Worker 工件的磁盘空间。依赖解析仍可能随未锁定 Python 版本范围变化；保存安装后的 `pip freeze` 以支持环境重建。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
cp .env.platform.example .env.platform
# 按 .env.example 配置 .env；不要覆盖现有真实凭据文件。
cd web
npm ci
npm run build
cd ..
.venv/bin/python scripts/platform-local.py infra
.venv/bin/python scripts/platform-local.py doctor
```

`infra` 复用 `compose.platform.yml` 并增加 Mailpit。Compose project 默认是 `research-copilot-platform`；可用 `PLATFORM_COMPOSE_PROJECT` 更改，但各主机端口固定，不能同时启动两套。基础设施端口只绑定回环地址。MySQL/RabbitMQ 初始化需要时间，`doctor` 展示健康状态但不自动宣称依赖 ready。

数据库准备好后执行**显式迁移**，不要用测试 `initialize()` 创建生产数据库：

```bash
.venv/bin/python scripts/platform-local.py migrate
.venv/bin/python scripts/platform-local.py run --offline
```

`migrate` 从 `.env.platform` 读取数据库配置并执行 Alembic `upgrade head`，包含当前全部迁移，而不是固定到某个历史版本。`run` 在前台监管四个进程：API `127.0.0.1:8000`、Worker、Scheduler、Next.js `localhost:3000`。任一子进程退出会结束整组；Ctrl-C 向所有子进程组发送 TERM，最多等 40 秒后终止，未完成任务由持久租约恢复。它不是 systemd 的替代品，也不自动无限重启。

访问 `http://localhost:3000`，邮箱验证邮件在 `http://localhost:58025`；注册、打开验证链接后登录。前端通过同源 `/api/backend` 代理到 API，浏览器主机名必须与 `PLATFORM_PUBLIC_ORIGIN` 一致。使用 `127.0.0.1:3000` 时也应修改 public origin。`--offline` 强制本地模型/embedding、内存向量、禁止 reranker 自动下载，仅用于可运行性演示，不能作为真实模型或 RAG 质量证据。使用真实模型时去掉 `--offline`，并配置 `.env` 及需要的向量存储。

初步检查：`curl --fail http://127.0.0.1:8000/health`；完成一次注册验证、PDF 上传导入、问答、取消/重试，再刷新页面确认任务与会话从后端恢复。API health 仅证明数据库连接，不代表 RabbitMQ、对象存储或模型可用。应用日志输出到前台，任务错误和事件保存在 MySQL。

## 停机一致性备份

跨 MySQL/对象存储/SQLite 没有原子在线快照。本机脚本使用停机备份：停止所有 API、Worker、Scheduler 和外部写入者，待进程退出后才传 `--quiesced`。脚本无法识别由其他进程管理器启动的写入者。默认 Worker 根目录为 `.data/platform-worker`；脚本读取 `PlatformSettings.worker_root`，支持 `.env.platform` 或 `PLATFORM_WORKER_ROOT` 指定的自定义路径。恢复时使用目标环境配置的根目录，必须为空；不要指向正在使用的源目录。

```bash
# 先在 run 的终端 Ctrl-C，确认所有应用进程结束。
.venv/bin/python scripts/platform-backup.py backup /secure/backups/platform-2026-11-09 --quiesced
.venv/bin/python scripts/platform-backup.py verify /secure/backups/platform-2026-11-09
```

脚本停止 Compose 基础设施后，归档 MySQL 数据卷、MinIO 数据卷与 Worker 文件/SQLite，生成 SHA256 清单；完成后保持停机。它复用本地 mysql:8.4 镜像中的 tar，不需要额外备份容器镜像。备份目录权限 0700，包含私人论文、消息和会话摘要，需由运维加密离线保管。另行安全托管 `.env`、`.env.platform`、代码提交、镜像 digest、`pip freeze` 和前端 lock；脚本不复制密钥文件。备份成功后可运行 `infra`、`run` 恢复服务。

Redis 限流状态无需恢复；RabbitMQ 不作物理复制，恢复后根据 MySQL outbox 重放，避免 Rabbit 节点身份与物理卷绑定问题。任务 generation/epoch 与清理幂等性是重放的前提。外部 Milvus 等持久派生存储不包含在本机备份中；本配置用 memory provider 从授权 parsed 工件重新构建索引。自定义外部存储必须制定对应备份/重建流程后才能声称完整恢复。

## 恢复到全新环境

先在隔离主机或空 Compose project 中演练；**不要对原实例执行覆盖恢复**。目标使用相同 MySQL 8.4 镜像/架构、同一代码版本和迁移集。先恢复密钥配置，安装依赖和构建前端，但不要运行 `infra`（启动 MySQL 会初始化非空卷）。恢复命令只创建停止的容器，并拒绝非空 MySQL/MinIO 卷和非空 Worker 目录。

```bash
.venv/bin/python scripts/platform-backup.py verify /secure/backups/platform-2026-11-09
.venv/bin/python scripts/platform-backup.py restore /secure/backups/platform-2026-11-09 --quiesced
.venv/bin/python scripts/platform-local.py infra
.venv/bin/python scripts/platform-local.py migrate
```

启动应用前，在恢复的 MySQL 中执行 `deploy/platform-replay-outbox.sql`。例如本地 profile（命令行不展开密码）：

```bash
docker-compose -p research-copilot-platform -f compose.platform.yml exec -T mysql \
  sh -c 'MYSQL_PWD="$MYSQL_PASSWORD" mysql -u "$MYSQL_USER" "$MYSQL_DATABASE"' \
  < deploy/platform-replay-outbox.sql
.venv/bin/python scripts/platform-local.py run --offline
```

使用 Compose v2 时把 `docker-compose` 改为 `docker compose`。恢复后检查 Alembic 版本、用户/项目/文档/消息/报告数量、文件 SHA256 与下载、到期 running task 的恢复及 queued 任务最终状态。重新登录验证会话策略；旧会话摘要也在快照中，公网恢复时应按事故策略撤销。不得仅因 tar 校验通过就宣称业务恢复成功；记录实测恢复时间/RPO/RTO。

## 公网部署边界

本地 Compose 固定开发密码，未提供公网生产认证/TLS，不能直接暴露端口。上线需单独凭据与网络隔离、TLS 反向代理、正确 Origin、生产 Secure Cookie、邮件服务、分布式限流、请求体上限、持久日志监控和自动进程重启。保留单机 Worker/SQLite 约束，不能未经并发验证扩到多主机。设置 `PLATFORM_ENVIRONMENT=production` 会检查部分配置，但不替代部署评审。

2026-11-10 已完成一次隔离恢复演练：停写后备份到 `.data/backup-rehearsal-20261110`，恢复到独立 Compose project `research-copilot-restore` 的空 MySQL/MinIO 卷，并将 Worker 目录恢复到 `.data/restore-worker`。源实例与恢复实例没有同时占用相同端口。原库与恢复库逻辑指纹核对为 **24 张表、361089 行与 5 个实际文件 SHA256 一致**；本机证据为 `.data/original-fingerprint.json` 和 `.data/restored-fingerprint.json`。该数量包含测试/合成业务数据，不是用户规模；校验的 5 个文件不包括性能数据集中的无实体文件元数据。

本次演练验证单机停写备份与空卷恢复，没有覆盖原实例，也没有证明全新主机依赖安装、外部向量库灾备或所有多模态路径恢复。没有据此声明 RPO/RTO 数值；发布前仍需部署环境专项验证，见[验证记录](platform-validation.md)。

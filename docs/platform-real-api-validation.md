# 真实模型 API 全链路验收

2026-11-14（Asia/Shanghai）在本机专用平台环境执行。成功轮次为 00:59:49–01:04:15，服务重启后于 01:05:34 完成持久化复验。基础提交为 `cf64b097b32d2286829eba594ed2905c44c72512`，本记录依赖同批提交的修复与脚本。

## 实际通过的链路

HTTP 注册 → Mailpit 邮箱验证 → Cookie/CSRF 登录 → 项目 → MinIO 签名上传 → 完成上传 → MySQL 任务/Outbox → RabbitMQ → Worker → PDF 解析 → 远程图谱抽取及 Embedding → Milvus 索引/检索 → Supervisor → 文本/图片答案 → MySQL 消息与报告 → HTTP 读取。

测试 PDF 由脚本生成，正文含 Orion 在 TestSet 上准确率为 92%；Alpha=37、Beta=82 只绘制在嵌入图片中，不写入 PDF 文本层或图片提问。文本回答正确返回 92%，图片回答正确返回 37、82，比较结果为 Beta 更大。Worker 观察器记录到实际携带 `image_url` 的 VLM 请求，未替换模型响应。

| 任务 | ID | 终态 | HTTP 轮询测得耗时 |
| --- | --- | --- | --- |
| 导入 | `9eaa788b-043f-4886-820e-cfcde8fd2658` | succeeded | 72.46 秒 |
| 文本问答 | `5c0b47d3-42bb-4582-8c22-73ffb255cfbf` | succeeded | 87.48 秒 |
| 图片问答 | `36b37c03-44b3-48c9-9fd4-727533e94a0e` | succeeded | 97.10 秒 |

成功轮次的远程调用统计如下，不含此前失败轮次及独立单调用 smoke：

| 模型 | 请求数 | HTTP 结果 | 服务商返回的 total_tokens 合计 |
| --- | --- | --- | --- |
| `deepseek-flash` | 11 | 全部 200 | 149369 |
| `qwen3.7-text-embedding-flash` | 11 | 全部 200 | 2137 |
| `qwen3.8-flash` | 1，含图片 | 200 | 2868 |

DeepSeek 通过 `api.deepseek.com`，Embedding/VLM 通过当前配置的 `maas.qianwenaiapi.com`。Token 是返回 usage 的机械求和，不是计费金额；不推断端点背后的模型实现。全部请求的 HTTP 成功还须结合上面的业务答案断言，不能单独证明链路成功。

同时验证：新 HTTP 客户端读取相同历史；原 PDF 下载字节一致；另一个经过邮箱验证的用户读取该文件返回 404。随后完整停止并重启 API、Worker、Scheduler，通过测试账号的真实密码重置邮件重新登录，确认文本与图片回答完全一致、1 份含正文结果的研究报告可读、PDF 字节一致。该报告由问答流程产生，未另测长篇独立综述生成。

## 暴露并修复的问题

1. 业务模式拒绝平台配置的 memory graph，导致真实模型 Worker 初始化失败。现在由 AgentBridge 显式注入本 attempt 独立的 InMemoryGraphStore，仍使用远程模型和 Milvus；旧业务配置校验保持原约束。
2. 当前 Embedding 实际返回 1024 维，本机 `.env` 原配置为 3072，导致 Milvus 导入失败。已将本机维度修正为 1024，没有删除既有集合；每次执行使用独立集合。
3. 私有 PDF 的 PaperCandidate 缺少 `storage_uri`，图片工具跳过输入。现在 Worker 从已鉴权的对象存储读取原 PDF，写入本 attempt 的 uploads 目录再交给工具，并保留越权文档拒绝测试。此前任务虽然标记 succeeded，但答案未包含图中数值，验收脚本正确判失败。
4. 图谱抽取原固定 12 秒，真实推理请求曾因此降级。新增 `GRAPH_TEXT_TIMEOUT_SECONDS`，默认仍为 12，本次脚本设为 60；成功轮次中可观测到约 39 秒和 52 秒的抽取调用。

## 复现

按[运行手册](platform-runbook.md)启动专用 MySQL/RabbitMQ/Redis/MinIO/Mailpit 并迁移。配置 `.env` 中真实 LLM、Embedding、VLM，以及可连接的 Milvus；维度必须匹配实际 Embedding 返回值。本机使用 Milvus 2.5.4、端口 19531。已有仓库 Milvus 容器可通过 `docker-compose -p research-copilot -f docker-compose.milvus.yml start` 恢复。使用已有本地 reranker 缓存。

```bash
# 终端一；读取现有凭据，启动 API 58000、真实 Worker、Scheduler
python scripts/platform-real-e2e.py serve

# 终端二；实际消耗 API 配额，创建合成 PDF 和独立测试账号
python scripts/platform-real-e2e.py test

# 在终端一 Ctrl-C，待子进程全部退出，再重新运行 serve
# 终端二；经邮件重置本脚本账号密码，再通过 API 检查恢复结果
python scripts/platform-real-e2e.py verify-restored
```

不要并行启动其他平台 Worker/Scheduler；真实依赖回归前先停止上述服务，避免消费故障测试的任务。测试保留合成账号/任务与本机工件，不清理其他业务数据；仅对隔离开发环境执行。

原始证据位于 `.data/real-e2e/result.json`、`provider-calls.jsonl`、`accepted-worker.log`、各 task ID 的 `*-events.txt`。`result.json` 含 `success: true` 与 `restart_verification.success: true`；这些本机文件被 Git 忽略。观察器只记录端点 host/path、模型、状态、耗时、图片数量和 usage，不记录密钥、请求头或提示词。重复运行会覆盖 result 和服务日志，应先归档。

本次恢复验证后的 `result.json` SHA256 为 `75ba715623a639d5a1c68031db3d53b056926cde193903d354081de4bfefb4a3`。

## 验证边界

- 本轮通过 HTTP 调用后端，没有重新跑连接真实模型的浏览器交互；此前浏览器验收使用 local 模型。
- 真实 Milvus 的单文档导入/检索已通过；图谱为 attempt 内存隔离实现，本轮不使用 Neo4j。跨用户文件拒绝已验证，真实 Milvus 多用户混合召回攻击仍需专项覆盖。
- 验证同机应用进程重启，不等于 MySQL/MinIO 灾备、跨主机恢复或所有多模态路径可迁移。
- 合成单页 PDF 无法代表真实论文集质量、长任务稳定性或容量。真实调用取消/重试、公共论文检索下载、长篇综述、公网部署不属于本次成功断言。
- 文本问答单次调用出现 76532 prompt tokens；每次研究任务仍会重建派生索引并抽取图谱。当前耗时和 token 数表明还需要性能优化，不能据此宣称低延迟或低成本。
- 图片答案目前附带服务器本地导出路径；本次证明图片被实际识别，没有证明浏览器能以私有鉴权 URL 展示导出图。

## 修复后的回归

停止真实测试 API/Worker/Scheduler 后执行 `RUN_PLATFORM_INTEGRATION=1 python -m pytest -q -o faulthandler_timeout=60`：**1031 passed，2 条上游弃用警告，160.47 秒，退出 0**。日志在 `/tmp/research-real-regression-final.log`。该回归中的 Agent 使用测试模型，真实模型证据以本文件前面的独立 HTTP 验收为准。

首次完整回归暴露两个既有 Embedding 适配器测试读取本机 `.env` 的问题：新增的真实 Embedding 地址覆盖了其预期备用地址。给这两个测试显式传 `_env_file=None`，不改变产品的配置优先级；针对性复验 9 passed，随后上述全量回归通过。新增的两个 scope 用例覆盖业务模式图谱构造和授权 PDF 落盘。全部改动 Python 文件 Ruff 检查与 `git diff --check` 通过。

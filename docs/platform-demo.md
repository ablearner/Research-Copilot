# 5–8 分钟演示脚本

目标是展示一个完整科研 Agent 产品的业务闭环，以及可以解释的后端故障行为。准备和安装不计入演示时长；首次启动步骤见[运行手册](platform-runbook.md)。现场只使用自己的专用演示数据，不对共享开发库停服务或改权限。

## 演示前准备

安装项目 Python 依赖与 Web 依赖，将 `.env.platform.example` 复制到 `.env.platform` 并核对端口。保留邮箱验证和限流；Mailpit 用于接收本地验证邮件。已有配置不要覆盖。以下从仓库根目录运行，`python` 应是已安装项目依赖的 Python 3.11 环境：

```bash
python scripts/platform-local.py infra
python scripts/platform-local.py migrate
python scripts/platform-local.py doctor
npm --prefix web ci
npm --prefix web run build
python scripts/platform-local.py run --offline
```

使用默认 `http://localhost:3000` 打开 Web，邮件箱为 `http://localhost:58025`。Next.js 构建时默认代理 `127.0.0.1:8000`；若使用其他 API 端口，构建前设置 `PLATFORM_API_ORIGIN`。不要把 `localhost` 和 `127.0.0.1` 混用，Cookie 与 CSRF origin 应匹配。

`--offline` 使用本地模型、embedding 和 memory 检索 provider，仍运行真实业务服务和 Supervisor；开场明确说明它用来稳定演示控制流程，不代表真实大模型回答质量。提前准备 `web/tests/fixtures/evidence.pdf`，内容包含方法 Alpha 的准确率证据。另开浏览器无痕窗口作为用户 B，避免共用 Cookie。

故障演示采用已有自动化用例，另准备终端。全局 Outbox/租约测试必须与后台 Worker/Scheduler 错开：完成 UI 演示后在启动终端按 Ctrl-C，等待子进程退出，再执行测试。中间件继续运行。若要展示真实模型效果，可另录一次真实模型调用并标记配置；不要用离线替身的结果冒充。

## 现场顺序

| 时间 | 操作 | 展示与解释 |
| --- | --- | --- |
| 0:00–0:40 | 简述产品，A 注册；在 Mailpit 完成验证、登录 | 自动创建个人工作区；身份来自后端会话 |
| 0:40–1:30 | 新建项目，论文库上传示例 PDF | 浏览器拿到受限上传授权；服务端完成时重新校验，解析/索引作为异步任务 |
| 1:30–2:10 | 查看任务与论文状态，解析/索引均完成后进入对话 | “上传成功”和“可研究”是两个状态；MySQL 保留任务事件 |
| 2:10–3:10 | 提问 `What accuracy does method Alpha achieve?`；提交后刷新页面，再打开该会话 | 用户消息、任务和最终助手消息从后端恢复；不是组件内假进度 |
| 3:10–4:00 | B 在无痕窗口注册验证，复制 A 的 PDF 下载链接到 B 窗口 | 返回资源不可见；不能靠知道文件 ID 绕过权限 |
| 4:00–4:40 | A 查看报告入口/历史结果；若本次 QA 未产出报告，就展示已保存消息；进入任务面板 | 不承诺每个 QA 都生成报告。说明报告只在 Agent 有相应输出时保存版本 |
| 4:40–6:20 | 停止应用进程，运行下面两个故障测试 | 展示发布确认与数据库标记间断点，以及领取后进程被杀的恢复 |
| 6:20–7:30 | 展示测试断言、架构图和限制 | 总结幂等、epoch fencing、单 Worker 记忆边界，不宣称恰好一次或生产高可用 |

现场回答较慢时可提前准备已完成的任务，仍现场展示一条新任务的提交与刷新恢复。不要为了赶时长把准备好的结果说成刚刚运行。

## 可复现故障证据

应用 Worker/Scheduler 停止后，在仓库根目录运行：

```bash
RUN_PLATFORM_INTEGRATION=1 python -m pytest -q \
  tests/integration/backend/test_queue_resilience.py::test_confirm_mark_gap_retains_outbox_and_permits_duplicate_delivery \
  tests/integration/backend/test_task_mysql.py::test_killed_claiming_process_recovers_without_duplicate_commit
```

第一个测试模拟 RabbitMQ 已确认、Outbox 尚未标记时中断，检查记录仍可重投；第二个启动独立领取进程并终止它，缩短测试租约，检查过期恢复及旧 epoch 无法提交。测试用的短租约不能作为真实环境恢复耗时。现场以实际命令输出为准，不预先写成功数量或恢复秒数。

补充的部分导入与 SSE 证据可运行已有集成/API 用例，不需要现场注入随机网络故障。浏览器刷新展示的是历史恢复；SSE 断线按 `Last-Event-ID` 补发需结合 API 测试断言说明，不能把刷新本身当作完整网络断线测试。

## 浏览器自动验收

`web/tests/e2e/platform.spec.ts` 覆盖注册、项目、私有上传、导入、跨用户文件拒绝、Agent 回答、刷新和旧文件入口关闭。该用例当前直接注册后进入工作区，测试专用 API 需设置 `PLATFORM_REQUIRE_EMAIL_VERIFICATION=false`；邮箱验证另由真实 Mailpit 集成测试覆盖。不要在公网环境关闭验证。

在独立测试应用运行且 `PLATFORM_PUBLIC_ORIGIN` 与测试 URL 一致时：

```bash
cd web
E2E_BASE_URL=http://localhost:3000 npx playwright test
```

浏览器及系统依赖需预先安装。用例产出 `web/test-results/research-workspace.png`，失败时保留 trace；实际测试结果、配置与原始日志见[验证记录](platform-validation.md)。本脚本是复现流程，不表示本文件编辑时重跑过所有命令。

## 面试讲述边界

可以展示：Agent 研究决策、多用户资源建模、事务 Outbox、重复投递防护、租约恢复、私有文件、删除语义和版本发布。必须主动解释：目前单 Worker、本地确定性模型/memory provider 测试范围、业务阶段重试而非任意推理节点续跑、没有真实用户规模或生产 SLA。

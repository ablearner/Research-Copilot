# Research-Copilot 改造方案清单

这份文档的目标不是讨论抽象愿景，而是给 `Research-Copilot` 提供一份可以逐步落地的改造清单。

改造原则只有一条：

- 保留 `Research-Copilot` 当前“科研工作流系统”的主线
- 借鉴 `nanobot` 在 agent runtime、tool 治理、memory 和产品化方面的长处
- 不把系统改造成一个泛聊天机器人

可以把整体方向概括为：

```text
先补稳定性和运行时边界
-> 再补 skill 化、tool 治理、长期 memory
-> 最后补多入口、hook/plugin、深化多代理
```

---

## 一、改造优先级总览

### 高优先级

优先解决“系统能否长期稳定运行、恢复、追踪”的问题。

包括：

- 统一任务与会话状态机
- 补强上下文压缩与恢复机制
- 统一 runtime 事件模型
- 工具治理标准化
- 核心科研能力 skill 化

### 中优先级

在系统已经稳定可控的前提下，补“长期助手能力”和“产品可用性”。

包括：

- 引入长期用户记忆层
- 增加 CLI / SDK 入口
- 配置与部署体验升级
- 可观测性与评测打通

### 后优先级

在主流程稳、边界清晰之后，再补平台能力和复杂协作能力。

包括：

- Hook / Plugin 机制
- 多入口轻集成
- 深化多代理协作
- 对外生态与平台化

---

## 二、高优先级改造清单

### 1. 统一任务与会话状态机 ✅ 已实现

目标：
让 `Research-Copilot` 从“多个流程能跑”变成“整个研究线程可恢复、可追踪、可重试”。

改造清单：

- 统一定义 `conversation / task / import_job / qa_job / advanced_action_job` 的状态枚举
- 增加标准生命周期：`queued -> running -> waiting_input -> completed / failed / cancelled`
- 为每类任务增加统一元数据：
  - `started_at`
  - `updated_at`
  - `finished_at`
  - `error_code`
  - `error_message`
  - `retry_count`
- 前端和后端统一使用同一套状态字段，避免各自做隐式推断
- 所有长任务支持幂等重试和任务恢复
- 为跨模块调用增加 `task_id / conversation_id / correlation_id`

预期收益：

- 导入、QA、分析这些链路会更容易排错
- 系统刷新、重连、重试时更稳定
- 为后续 event stream、观察性和多代理打基础

### 2. 补强上下文压缩与恢复机制 ✅ 已实现

目标：
避免研究线程越跑越重，同时保留关键研究上下文。

改造清单：

- 区分“原始消息历史”和“压缩后的研究摘要”
- 定义自动压缩触发条件：
  - 超过轮次阈值
  - 超过 token 阈值
  - 进入关键阶段切换时触发
- 规定压缩摘要必须包含：
  - 当前研究目标
  - 已导入论文集合
  - 当前关键结论
  - 当前证据缺口
  - 下一步建议动作
- 前端恢复会话时优先拉取研究摘要，再按需展开详细历史
- 给摘要增加版本号和更新时间，避免旧摘要污染新任务
- 将压缩结果和 workspace snapshot 对齐，避免两份状态表达冲突

预期收益：

- 长研究线程不会迅速失控
- 恢复会话的速度和一致性会更好
- agent 在多轮对话中的稳定性更高

### 3. 统一 runtime 事件模型 ✅ 已实现

目标：
让 agent、tool、日志、前端实时展示使用同一套运行事件语言。

改造清单：

- 定义统一事件类型：
  - `agent_started`
  - `agent_routed`
  - `tool_called`
  - `tool_succeeded`
  - `tool_failed`
  - `memory_updated`
  - `task_completed`
  - `task_failed`
- `services/research/` 和 `rag_runtime/` 都通过统一事件结构输出运行过程
- 为事件增加统一字段：
  - `event_id`
  - `event_type`
  - `task_id`
  - `conversation_id`
  - `correlation_id`
  - `timestamp`
  - `payload`
- 前端研究线程直接消费这些事件做实时状态更新
- 保留事件落盘或追踪接口，方便后续回放和调试

预期收益：

- 运行过程更透明
- 前端展示不再依赖零散状态拼接
- 后续接 observability 和 replay 成本更低

### 4. 工具治理标准化 ⚠️ 部分实现

目标：
把“很多内部能力”变成“受控、可观测、可维护的工具系统”。

改造清单：

- 统一所有 tool 的输入输出 schema
- 明确区分三类工具：
  - `runtime tools`
  - `research business tools`
  - `external MCP tools`
- 为每个 tool 定义：
  - 超时策略
  - 重试策略
  - 错误分类
  - 审计字段
- 记录 tool 调用日志，包括：
  - 调用时间
  - 调用耗时
  - 输入摘要
  - 输出摘要
  - 错误原因
- 对高风险工具增加权限控制和环境隔离
- 给上层 supervisor 明确 tool 能力边界，减少隐式耦合

预期收益：

- tool-first runtime 会真正变成可治理平台
- 工具异常更容易追踪
- 后续接外部能力时风险更低

### 5. 核心科研能力 skill 化 ✅ 已实现

目标：
减少 service 层硬编码编排，提高复用性、可测性和可扩展性。

改造清单：

- 优先抽取 3 到 5 个核心科研 skill：
  - `literature_review`
  - `paper_compare`
  - `chart_interpretation`
  - `research_gap_discovery`
  - `paper_deep_analysis`
- 每个 skill 固定四类内容：
  - 输入结构
  - 使用的工具依赖
  - 输出结构
  - 评价指标
- Supervisor 主要负责路由和组合，不直接承载过多业务细节
- skill 既能服务后端 agent，也能服务前端固定动作入口
- 为每个 skill 增加单测和少量真实链路测试

预期收益：

- 高层编排会更清晰
- 核心科研能力更容易复用
- 后续做 specialist agent 时拆分成本更低

---

## 三、中优先级改造清单

### 1. 引入长期用户记忆层 ✅ 已实现

目标：
让系统从“当前任务助手”进化为“长期研究助手”。

改造清单：

- 新增用户层 memory，和任务层 workspace 分离
- 保存长期偏好信息：
  - 研究兴趣
  - 常用主题
  - 偏好作者 / 期刊 / 会议
  - 常见检索策略
- 增加长期记忆写入策略，避免所有消息都写入长期层
- 让 discovery、QA、analysis 能读取用户长期偏好
- 前端提供记忆查看、编辑、清理入口

预期收益：

- 用户体验更连续
- 系统对个人研究习惯的适配能力更强

### 2. 增加 CLI / SDK 入口 ✅ 已实现

目标：
降低使用门槛，让系统不只依赖 Web 工作区。

改造清单：

- 增加 CLI 命令，覆盖：
  - 检索论文
  - 导入论文
  - 发起研究问答
  - 查询任务状态
- 提供 Python SDK 或轻量 API client
- 将当前 Web 工作流中的 research action 下沉为稳定接口
- 补齐无 UI 场景下的错误信息和进度反馈

预期收益：

- 自动化接入更容易
- 外部系统复用 `Research-Copilot` 能力的门槛更低

### 3. 配置与部署体验升级

目标：
把仓库从“开发者能跑”提升到“其他人更容易部署和维护”。

改造清单：

- 增加配置分层：`local / dev / prod`
- 提供环境检查脚本，验证：
  - `Milvus`
  - `Neo4j`
  - `MySQL`
  - `API key`
- 区分最小运行模式和完整运行模式
- 补齐标准 Docker / compose 启动方案
- 把当前文档中的隐含依赖显式化

预期收益：

- 新接手者更容易上手
- 本地、测试、部署环境差异更容易控制

### 4. 可观测性与评测打通 ⚠️ 部分实现

目标：
让系统优化从“感觉更好”转为“指标更好”。

改造清单：

- 统一记录关键指标：
  - task 成功率
  - tool 成功率
  - retrieval 命中情况
  - rerank 效果
  - 回答耗时
- 建立失败案例归档
- 扩展 `evaluation/`，覆盖：
  - discovery
  - collection QA
  - chart drilldown
  - paper compare
- 建立真实科研任务的回归测试集

预期收益：

- 优化方向更明确
- 回归问题更容易被及时发现

---

## 四、后优先级改造清单

### 1. Hook / Plugin 机制

目标：
给后续团队扩展留标准接口，而不是所有逻辑都改核心代码。

改造清单：

- 增加 lifecycle hooks：
  - 任务开始
  - 任务完成
  - 任务失败
  - tool 调用前后
- 增加 tool middleware
- 支持输出后处理插件，例如：
  - 引用格式化
  - 结果落库
  - 通知发送
- 支持 provider 层 hook，便于审计和实验

预期收益：

- 扩展能力不再严重依赖改核心模块
- 外围系统更容易接入

### 2. 多入口轻集成

目标：
在不破坏主工作区的前提下，补充轻量交互入口。

改造清单：

- 增加通知型入口，例如飞书、企业微信
- 提供只读结果推送和任务提醒
- 保持复杂操作仍在 Web 工作区完成
- 不把系统改成泛聊天机器人

预期收益：

- 更适合真实日常使用
- 保持主产品聚焦，不被多渠道复杂度拖垮

### 3. 深化多代理协作 ✅ 已实现（基础版）

目标：
在基础稳定后，再追求更高并行和复杂协作能力。

当前已拆分的 specialist agents：

- LiteratureScoutAgent — 文献检索
- ResearchKnowledgeAgent — 导入、QA、压缩
- ResearchWriterAgent — 综述写作
- PaperAnalysisAgent — 论文分析/对比/推荐
- ChartAnalysisAgent — 图表分析
- GeneralAnswerAgent — 通用问答
- PreferenceMemoryAgent — 偏好推荐
- 增加并行执行和结果聚合策略
- 控制代理职责，避免 orchestration 过重
- 为多代理失败保留单代理回退路径

预期收益：

- 在复杂研究任务里提升吞吐和分工清晰度
- 但前提是单代理主路径已经足够稳

### 4. 对外生态与平台化

目标：
如果后续从科研助手走向科研平台，可以逐步开放系统能力。

改造清单：

- 增加插件市场或内部 skill registry
- 开放标准化 API 给外部系统
- 支持第三方研究数据源和机构工具接入
- 增加权限、租户、配额等平台能力

预期收益：

- 为平台化扩张预留空间
- 但不应在系统主路径尚未稳定时优先推进

---

## 五、建议落地顺序

建议按下面顺序推进：

1. 重构状态模型和任务生命周期 ✅
2. 加入上下文压缩与恢复 ✅
3. 统一 runtime 事件模型 ✅
4. 标准化 tool schema 和 tool 日志 ⚠️ 部分完成
5. 抽出首批核心 research skills ✅
6. 引入长期用户记忆 ✅
7. 增加 CLI / SDK ✅
8. 补强 observability 和 evaluation ⚠️ 部分完成
9. 再考虑 hook/plugin 和多代理深化 ✅ 基础版已完成

---

## 六、落地原则

整个改造过程中，建议始终坚持下面几条边界：

- 不为了“更像通用 agent”而削弱科研工作流主线
- 不为了“多代理”而过早引入复杂编排
- 不为了“平台化”而打散当前已经成型的 research 业务模型
- 优先让系统更稳定、更透明、更可恢复，而不是先堆新能力

一句话总结：

`Research-Copilot` 最值得从 `nanobot` 学的，不是“通用聊天能力”，而是“把一个能做事的系统，升级成一个可持续运行、可扩展、可产品化的 agent 系统”。 

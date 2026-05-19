---
name: literature-survey
description: "根据研究主题生成文献综述大纲和初稿，覆盖背景、方法分类、趋势分析。默认先用候选论文元数据和摘要完成搜索后综述；只有用户明确要求全文、导入、PDF 入库、精读或证据问答时，才把 import_papers 作为后续步骤。"
planner_guidance: >
  For broad literature survey or research-overview requests, plan discovery and
  metadata/abstract-based review first: search_literature -> write_review ->
  finalize. Do not schedule import_papers only because importable candidates
  exist or because a first draft can be improved. Treat import_papers as an
  optional follow-up for explicit full-text, PDF ingestion, local workspace,
  close-reading, or grounded QA requests.
planning_policy:
  default_workflow: [search_literature, write_review, finalize]
  action_policies:
    import_papers:
      default_enabled: false
      enable_when: [auto_import, mode_import, mode_qa, paper_import_intent, collection_qa_intent, single_paper_qa_intent]
      blocked_recovery: finalize
      blocked_reason: "Literature survey defaults to metadata/abstract review; import_papers is only enabled by explicit import/full-text intent."
  failure_recovery:
    write_review:
      default_action: finalize
      stop_reason: "摘要级综述已尝试生成但质量门未通过；本轮不自动导入全文，请由用户确认是否需要全文导入后再继续。"
version: 1.2.0
author: Research-Copilot
category: writing
tags: [survey, review, writing, 综述, 文献, 调研, 报告]
triggers:
  - "写.*综述"
  - "文献综述"
  - "文献.*调研"
  - "调研.*报告"
  - "literature.*survey"
  - "literature.*review"
  - "write.*review"
  - "survey.*topic"
  - "综述.*领域"
  - "梳理.*文献"
  - "帮我.*总结.*论文"
  - "research.*overview"
  - "state.*of.*the.*art"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Literature Survey Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **LiteratureScoutAgent** — 先搜索和发现相关论文，确保论文池覆盖面足够
2. **ResearchWriterAgent** — 默认基于候选论文标题、摘要、来源、年份和可用元数据生成结构化综述
3. **ResearchKnowledgeAgent (import_papers)** — 仅在用户明确要求全文、PDF 入库、精读、基于全文证据问答，或后续任务确实需要本地文档证据时再导入
4. **PaperAnalysisAgent / ResearchQAAgent** — 可选：在导入后对关键论文做深度分析或证据问答

## 使用场景
当用户需要撰写文献综述或 survey 报告时激活此技能。典型请求：
- "帮我写一篇关于 X 的文献综述"
- "梳理一下 X 领域的最新进展"
- "State of the art in X"
- "调研 X 相关论文"

## 执行步骤

1. **主题确认**：与用户确认综述主题、范围和目标读者
2. **论文发现**：使用 LiteratureScoutAgent 搜索至少 15-20 篇相关论文，覆盖：
   - 经典开创性工作（高引用）
   - 近 2 年最新进展
   - 不同子方向的代表性工作
3. **分类体系**：根据已有论文建立分类体系（按方法、按时间、按问题域）
4. **大纲生成**：
   - 引言（研究背景与动机）
   - 相关工作分类与梳理
   - 方法对比与趋势分析
   - 开放问题与未来方向
   - 结论
5. **段落撰写**：逐节填充内容，确保每个观点有论文引用支撑
6. **润色检查**：检查逻辑连贯性、引用完整性、语言流畅度

## 导入策略
- 普通“调研 / 梳理 / 综述 / overview / survey”请求不默认导入 PDF；先交付候选集合和摘要级综述。
- 当用户明确说“导入、下载 PDF、加入工作区、精读、基于全文/证据回答、grounded QA”等意图时，再调用 `import_papers`。
- 如果首轮综述质量不足，优先重写为结构化的摘要级综述，并在“证据边界与局限”中说明全文未导入；不要把“需要更完整引用或章节”自动解释为必须导入全文。
- 可以在报告末尾建议“后续可导入这些代表论文做全文证据问答”，但这只是建议，不是同一轮自动执行步骤。

## 质量标准
- 综述应尽量引用候选集中可用论文；候选少于 10 篇时，使用实际候选数量并明确证据边界
- 必须包含至少一个结构化对比表
- 趋势分析部分必须有时间维度的观察（按年份梳理进展）
- 每个分类下优先列出 2-3 篇代表性工作；候选不足时说明覆盖不足
- 开放问题部分至少列出 3 个研究缺口

## 输出格式
- 结构化 Markdown 格式
- 论文引用使用运行时提供的 [P1]、[P2] 等候选论文编号
- 包含分类对比表
- 每节包含 2-4 段有实质内容的段落，不接受一句话概括

## 约束
- 综述应该是系统性的，不是简单罗列
- 需要明确指出各方法的优劣和适用场景
- 不允许虚构论文或编造引用
- 跟随用户语言（中文问题用中文回答，英文问题用英文）

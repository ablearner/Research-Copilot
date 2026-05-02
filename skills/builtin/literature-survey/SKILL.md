---
name: literature-survey
description: "根据研究主题生成文献综述大纲和初稿，覆盖背景、方法分类、趋势分析。Supervisor 应优先调度 LiteratureScoutAgent 进行论文发现，再调度 ResearchWriterAgent 生成综述。"
version: 1.1.0
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
2. **ResearchKnowledgeAgent (import_papers)** — 导入关键论文以获取全文
3. **ResearchWriterAgent** — 基于导入的论文生成结构化综述报告
4. **PaperAnalysisAgent** — 可选：对关键论文做深度分析补充细节

## 使用场景
当用户需要撰写文献综述或 survey 报告时激活此技能。典型请求：
- "帮我写一篇关于 X 的文献综述"
- "梳理一下 X 领域的最新进展"
- "State of the art in X"

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

## 质量标准
- 综述必须引用至少 10 篇论文，覆盖至少 2 个子方向
- 必须包含至少一个结构化对比表
- 趋势分析部分必须有时间维度的观察（按年份梳理进展）
- 每个分类下至少列出 2-3 篇代表性工作
- 开放问题部分至少列出 3 个研究缺口

## 输出格式
- 结构化 Markdown 格式
- 论文引用使用 [Author, Year] 格式
- 包含分类对比表
- 每节包含 2-4 段有实质内容的段落，不接受一句话概括

## 约束
- 综述应该是系统性的，不是简单罗列
- 需要明确指出各方法的优劣和适用场景
- 不允许虚构论文或编造引用
- 跟随用户语言（中文问题用中文回答，英文问题用英文）

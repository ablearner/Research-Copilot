---
name: research-evaluation
description: "评估研究工作的质量、创新性、实验严谨性和影响力，输出结构化审稿意见。Supervisor 应先确保论文已导入，再调度 PaperAnalysisAgent 和 ResearchQAAgent 执行评估。"
version: 1.1.0
author: Research-Copilot
category: research
tags: [evaluation, quality, review, 评估, 审稿, 打分, 质量]
triggers:
  - "评估.*论文"
  - "评价.*质量"
  - "审稿.*意见"
  - "evaluate.*paper"
  - "review.*quality"
  - "论文.*怎么样"
  - "这篇.*好不好"
  - "值不值得.*读"
  - "worth.*reading"
  - "给.*打.*分"
  - "peer.*review"
  - "写.*review"
  - "审.*这篇"
  - "论文.*水平"
  - "创新.*程度"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Research Evaluation Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **ResearchKnowledgeAgent (import_papers)** — 确保论文已导入并解析
2. **PaperAnalysisAgent** — 提取论文结构化信息
3. **ResearchQAAgent** — 针对评估维度进行 grounded 证据检索

## 使用场景
当用户要求评估论文质量、给出审稿意见或判断创新性时激活此技能。典型请求：
- "帮我评估一下这篇论文的质量"
- "给这篇写个 peer review"
- "这篇论文值不值得精读"

## 评估维度（必须逐项给分）

1. **创新性** (Novelty) [1-5 分]
   - 问题是否有新意？是增量改进还是开创性工作？
   - 方法是否有实质性创新？核心技术贡献是什么？
   - 与最相关的 2-3 篇现有工作的区分度
2. **技术严谨性** (Rigor) [1-5 分]
   - 方法描述是否清晰、可复现？
   - 数学推导是否正确、假设是否合理？
   - 基线对比是否充分、是否选择了最新 SOTA？
3. **实验质量** (Experiments) [1-5 分]
   - 数据集选择是否合适、规模是否足够？
   - 消融实验是否覆盖关键设计选择？
   - 是否有统计显著性检验或多次运行的方差报告？
4. **写作质量** (Presentation) [1-5 分]
   - 结构是否清晰、逻辑是否连贯？
   - 图表是否信息丰富、排版是否专业？
   - 相关工作覆盖是否全面？
5. **影响力** (Impact) [1-5 分]
   - 对领域的潜在推动作用（short-term / long-term）
   - 实际应用前景和可落地性
   - 开源代码/数据集的可用性

## 质量标准
- 每个维度必须有 1-5 分评分 + 至少 2 句具体说明
- Strengths 和 Weaknesses 各至少列出 3 条
- 每条 weakness 必须附带改进建议
- 评估必须有原文证据支撑，不能凭印象打分

## 输出格式
- 整体评分：X/5（五个维度的加权平均，创新性和实验质量权重更高）
- 推荐等级：Strong Accept / Accept / Borderline / Reject
- 每个维度：评分 + 说明
- Strengths（至少 3 条）
- Weaknesses（至少 3 条，每条附改进建议）
- Questions for Authors（可选）
- 跟随用户语言

## 约束
- 评估必须基于论文原文内容，不依赖论文外部信息（如引用量、作者声誉）
- 保持客观公正，不因论文来源或作者偏向
- 如果论文信息不足以评估某个维度，明确说明而非猜测

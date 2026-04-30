---
name: research-evaluation
description: "评估研究工作的质量、创新性、实验严谨性和影响力"
version: 1.0.0
author: Research-Copilot
category: research
tags: [evaluation, quality, review, 评估, 审稿]
triggers:
  - "评估.*论文"
  - "评价.*质量"
  - "审稿.*意见"
  - "evaluate.*paper"
  - "review.*quality"
  - "论文.*怎么样"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Research Evaluation Skill

## 使用场景
当用户要求评估论文质量、给出审稿意见或判断创新性时激活此技能。

## 评估维度

1. **创新性** (Novelty)
   - 问题是否有新意？
   - 方法是否有实质性创新？
   - 与现有工作的区分度
2. **技术严谨性** (Rigor)
   - 方法描述是否清晰可复现？
   - 实验设计是否合理？
   - 基线对比是否充分？
3. **实验质量** (Experiments)
   - 数据集选择是否合适？
   - 消融实验是否充分？
   - 统计显著性检验？
4. **写作质量** (Presentation)
   - 结构是否清晰？
   - 图表是否信息丰富？
   - 相关工作覆盖是否全面？
5. **影响力** (Impact)
   - 对领域的潜在推动作用
   - 实际应用前景

## 输出格式
- 每个维度给出 1-5 分评分和简要说明
- 列出主要优点（Strengths）
- 列出主要不足（Weaknesses）
- 给出改进建议

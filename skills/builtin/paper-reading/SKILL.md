---
name: paper-reading
description: "深度阅读单篇论文，提取核心贡献、方法、实验、局限性等结构化知识卡片。Supervisor 应先导入论文，再调度 PaperAnalysisAgent 分析，必要时用 ChartAnalysisAgent 解读图表。"
version: 1.1.0
author: Research-Copilot
category: research
tags: [paper, reading, analysis, 阅读, 精读, 讲解, 理解, 总结]
triggers:
  - "精读.*论文"
  - "读.*这篇"
  - "讲解.*论文"
  - "explain.*paper"
  - "深度.*阅读"
  - "read.*paper"
  - "帮我.*看.*这篇"
  - "这篇.*讲了什么"
  - "summarize.*paper"
  - "总结.*论文"
  - "这篇.*的.*贡献"
  - "这篇.*的.*方法"
  - "paper.*summary"
  - "论文.*笔记"
  - "reading.*notes"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Paper Reading Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **ResearchKnowledgeAgent (import_papers)** — 确保论文已导入并解析
2. **PaperAnalysisAgent** — 提取结构化分析
3. **ChartAnalysisAgent** — 可选：解读论文中的关键图表
4. **ResearchQAAgent** — 可选：回答用户关于论文的追问

## 使用场景
当用户要求深度阅读或讲解某篇论文时激活此技能。典型请求：
- "帮我精读一下这篇论文"
- "这篇 paper 讲了什么"
- "总结一下这篇的核心贡献和方法"

## 执行步骤

1. **获取论文信息**：从 paper pool 或用户指定获取论文元数据和全文
2. **结构化提取**：
   - **核心贡献**：一句话概括论文做了什么（Problem → Solution → Result）
   - **方法/框架**：技术路线和关键创新点，与前人工作的区别
   - **实验设计**：数据集、基线方法、评估指标
   - **主要结果**：定量结果（含具体数字）和定性发现
   - **局限性**：作者承认的 + 你观察到的潜在问题
   - **一句话总结**：面向非专业读者的极简摘要
3. **图表解读**：识别论文中最关键的 2-3 张图表，提供解读
4. **扩展思考**：与相关工作的联系、可能的改进方向、对用户研究的启发

## 质量标准
- 核心贡献必须精准概括，不是复述 abstract
- 方法描述必须说清楚"为什么这样做"而不只是"做了什么"
- 实验结果必须包含具体数字（如 accuracy 提升了 X%）
- 局限性分析不能只说"数据集有限"，要具体指出什么局限、为什么重要
- 一句话总结必须让非领域专家也能理解

## 输出格式
- 知识卡片格式（结构化字段）
- 关键概念用加粗标注
- 技术术语首次出现时给出简要解释
- 跟随用户语言

## 约束
- 分析必须基于论文原文，不允许凭记忆补充论文中没有的内容
- 如果论文未导入，先提示导入
- 保持论文标题的原始语言（英文标题不翻译）

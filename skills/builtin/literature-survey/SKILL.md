---
name: literature-survey
description: "根据研究主题生成文献综述大纲和初稿，覆盖背景、方法分类、趋势分析"
version: 1.0.0
author: Research-Copilot
category: writing
tags: [survey, review, writing, 综述, 文献]
triggers:
  - "写.*综述"
  - "文献综述"
  - "literature.*survey"
  - "literature.*review"
  - "write.*review"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Literature Survey Skill

## 使用场景
当用户需要撰写文献综述或 survey 报告时激活此技能。

## 执行步骤

1. **主题确认**：与用户确认综述主题、范围和目标读者
2. **分类体系**：根据已有论文建立分类体系（按方法、按时间、按问题域）
3. **大纲生成**：
   - 引言（研究背景与动机）
   - 相关工作分类与梳理
   - 方法对比与趋势分析
   - 开放问题与未来方向
   - 结论
4. **段落撰写**：逐节填充内容，确保每个观点有论文引用支撑
5. **润色检查**：检查逻辑连贯性、引用完整性、语言流畅度

## 输出格式
- 结构化 Markdown 格式
- 论文引用使用 [Author, Year] 格式
- 包含分类对比表

## 注意事项
- 综述应该是系统性的，不是简单罗列
- 需要明确指出各方法的优劣和适用场景
- 趋势分析部分应有时间维度的观察

---
name: paper-reading
description: "深度阅读单篇论文，提取核心贡献、方法、实验、局限性等结构化知识卡片"
version: 1.0.0
author: Research-Copilot
category: research
tags: [paper, reading, analysis, 阅读, 精读]
triggers:
  - "精读.*论文"
  - "读.*这篇"
  - "讲解.*论文"
  - "explain.*paper"
  - "深度.*阅读"
  - "read.*paper"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Paper Reading Skill

## 使用场景
当用户要求深度阅读或讲解某篇论文时激活此技能。

## 执行步骤

1. **获取论文信息**：从 paper pool 或用户指定获取论文元数据和全文
2. **结构化提取**：
   - **核心贡献**：一句话概括论文做了什么
   - **方法/框架**：技术路线和关键创新点
   - **实验设计**：数据集、基线、评估指标
   - **主要结果**：定量结果和定性发现
   - **局限性**：作者承认的和你观察到的
   - **一句话总结**：面向非专业读者的极简摘要
3. **图表解读**：如果有关键图表，提供解读
4. **扩展思考**：与相关工作的联系、可能的改进方向

## 输出格式
- 知识卡片格式（结构化字段）
- 关键概念用加粗标注
- 技术术语首次出现时给出简要解释

---
name: paper-comparison
description: "对多篇论文进行系统性对比分析，生成方法差异、实验对比、适用场景的结构化报告"
version: 1.0.0
author: Research-Copilot
category: research
tags: [paper, comparison, analysis, 对比]
triggers:
  - "对比.*论文"
  - "compare.*paper"
  - "方法.*差异"
  - "论文.*区别"
  - "difference.*between.*paper"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Paper Comparison Skill

## 使用场景
当用户要求对比多篇论文时激活此技能。

## 执行步骤

1. **确认论文范围**：确认用户想对比哪些论文，如果未指定则从当前 paper pool 中选取
2. **提取关键维度**：从论文中提取以下对比维度：
   - 研究问题与动机
   - 方法/框架设计
   - 实验设置与数据集
   - 主要结果与指标
   - 局限性与未来方向
3. **生成对比表**：用表格形式呈现各维度对比
4. **撰写分析段落**：针对每个维度写 2-3 句对比分析
5. **给出建议**：根据用户的研究方向推荐最相关的论文

## 输出格式
- 必须包含结构化对比表
- 每个结论需要引用具体论文的证据
- 中英文根据用户语言自动切换

## 注意事项
- 如果论文数量 >5，先让用户缩小范围
- 实验结果对比必须注明数据集和指标，不要跨数据集对比

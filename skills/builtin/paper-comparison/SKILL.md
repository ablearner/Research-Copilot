---
name: paper-comparison
description: "对多篇论文进行系统性对比分析，生成方法差异、实验对比、适用场景的结构化报告。Supervisor 应调度 PaperAnalysisAgent 或 ResearchQAAgent 执行对比。"
version: 1.1.0
author: Research-Copilot
category: research
tags: [paper, comparison, analysis, 对比, 差异, versus]
triggers:
  - "对比.*论文"
  - "compare.*paper"
  - "方法.*差异"
  - "论文.*区别"
  - "difference.*between.*paper"
  - "哪个.*方法.*更好"
  - "哪篇.*更.*适合"
  - "versus"
  - "vs\\.?"
  - "A.*和.*B.*有什么.*不同"
  - "比较.*方法"
  - "pros.*cons"
  - "优缺点"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Paper Comparison Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **ResearchKnowledgeAgent (import_papers)** — 确保待对比论文已导入
2. **PaperAnalysisAgent** — 对每篇论文提取结构化信息
3. **ResearchQAAgent** — 针对具体对比维度进行 grounded QA

## 使用场景
当用户要求对比多篇论文时激活此技能。典型请求：
- "对比一下 ViT 和 Swin Transformer"
- "这三篇论文的方法有什么区别"
- "哪个方法更适合小数据集场景"

## 执行步骤

1. **确认论文范围**：确认用户想对比哪些论文，如果未指定则从当前 paper pool 中选取
2. **提取关键维度**：从论文中提取以下对比维度：
   - 研究问题与动机
   - 方法/框架设计（核心创新点、技术路线）
   - 实验设置与数据集
   - 主要结果与定量指标
   - 计算成本与效率
   - 局限性与未来方向
3. **生成对比表**：用表格形式呈现各维度对比
4. **撰写分析段落**：针对每个维度写 2-3 句深入对比分析
5. **给出建议**：根据用户的研究方向和场景推荐最合适的方法

## 质量标准
- 对比表至少覆盖 5 个维度
- 定量结果对比必须标注数据集和评估指标
- 每个对比维度必须有来自原文的证据支撑
- 结论不能含糊（如"各有优劣"），必须明确在什么场景下推荐哪个

## 输出格式
- 必须包含结构化对比表（Markdown 表格）
- 每个结论需要引用具体论文的证据
- 跟随用户语言

## 约束
- 如果论文数量 >5，先让用户缩小范围
- 实验结果对比必须注明数据集和指标，不要跨数据集对比
- 不允许在缺乏原文数据时编造定量结果
- 对比应公允，不能只强调某篇论文的优点

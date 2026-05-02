---
name: chart-analysis
description: "分析和解读论文中的图表（实验结果图、架构图、流程图、数据分布图等），提取关键发现和洞察。Supervisor 应调度 ChartAnalysisAgent 执行分析。"
version: 1.0.0
author: Research-Copilot
category: research
tags: [chart, figure, graph, plot, 图表, 分析, 解读, 可视化]
triggers:
  - "分析.*图"
  - "解读.*图"
  - "看.*这张图"
  - "图表.*分析"
  - "analyze.*chart"
  - "analyze.*figure"
  - "explain.*figure"
  - "explain.*graph"
  - "这个图.*说明.*什么"
  - "实验.*结果.*图"
  - "figure.*说明"
  - "plot.*shows"
  - "看.*图.*几"
  - "Figure.*\\d"
  - "Table.*\\d"
  - "表.*\\d.*分析"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Chart Analysis Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **ResearchKnowledgeAgent (import_papers)** — 确保论文已导入（需要图表原始数据）
2. **ChartAnalysisAgent** — 执行图表理解和分析
3. **ResearchQAAgent** — 可选：结合论文文本回答关于图表的追问

## 使用场景
当用户要求分析或解读论文中的图表时激活此技能。典型请求：
- "分析一下这篇论文的 Figure 3"
- "这个实验结果图说明了什么"
- "帮我解读一下 Table 2 的消融实验"

## 执行步骤

1. **定位图表**：
   - 从用户请求中识别目标图表（Figure/Table 编号或描述）
   - 如果未指定，分析论文中最重要的图表
2. **图表类型识别**：
   - 折线图 / 柱状图 / 散点图 → 趋势和对比分析
   - 架构图 / 流程图 → 系统设计解读
   - 表格 → 数值对比和排名分析
   - 热力图 / 混淆矩阵 → 模式识别
3. **深度解读**：
   - 提取关键数据点和趋势
   - 识别异常值或有趣的 pattern
   - 与论文正文的描述交叉验证
   - 指出图表中可能被忽略的信息
4. **洞察总结**：
   - 这张图支撑了论文的什么结论
   - 是否有图表与文字描述不一致的地方
   - 对读者理解论文的关键启示

## 质量标准
- 数值分析必须精确，不能大致估计（如"A 方法在 X 数据集上的 F1 为 92.3%，比 B 高 1.8%"）
- 趋势分析必须说明变化方向和幅度
- 必须指出图表中最值得注意的 2-3 个发现
- 如果是消融实验表，必须指出哪个组件贡献最大

## 输出格式
- 图表类型 + 一句话概括
- 关键发现（编号列表）
- 详细分析段落
- 与论文结论的关联
- 跟随用户语言

## 约束
- 分析必须基于图表内容本身，不编造数据
- 如果图表分辨率不足以读取精确数值，明确说明
- 需要区分"图表展示的事实"和"我的解读/推断"

---
name: paper-discovery
description: "根据研究主题主动搜索和发现相关论文，支持多数据源检索、时间范围过滤和关键词扩展。Supervisor 应调度 LiteratureScoutAgent 执行搜索。"
version: 1.0.0
author: Research-Copilot
category: research
tags: [discovery, search, find, 搜索, 发现, 找论文, 检索]
triggers:
  - "搜.*论文"
  - "找.*论文"
  - "帮我.*找"
  - "search.*paper"
  - "find.*paper"
  - "有没有.*相关.*论文"
  - "有哪些.*论文"
  - "最新.*进展"
  - "latest.*work"
  - "recent.*paper"
  - "new.*research"
  - "discover.*paper"
  - "关于.*的.*论文"
  - "papers.*on"
  - "papers.*about"
  - "arxiv.*paper"
  - "哪些.*工作"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Paper Discovery Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **LiteratureScoutAgent** — 执行论文搜索和发现
2. **ResearchKnowledgeAgent (import_papers)** — 可选：导入用户感兴趣的论文
3. **ResearchWriterAgent** — 可选：生成搜索结果的摘要报告

## 使用场景
当用户想要搜索和发现特定主题的论文时激活此技能。典型请求：
- "帮我找 Vision-Language Navigation 的最新论文"
- "搜索一下 2024 年关于 LLM Agent 的工作"
- "有没有关于 diffusion model 做视频生成的论文"

## 执行步骤

1. **理解搜索意图**：
   - 提取用户的研究主题和关键词
   - 识别时间范围约束（如"最近一个月"、"2024年"）
   - 识别数据源偏好（如"arxiv"、"顶会"）
2. **查询扩展**：
   - 将中文主题翻译为英文搜索词（学术论文英文为主）
   - 生成 2-3 个同义/相关查询词（如 VLN → Vision-Language Navigation, Embodied Navigation）
   - 考虑缩写和全称的对应（如 RL → Reinforcement Learning）
3. **多源搜索**：默认搜索 arxiv + semantic_scholar，用户指定时切换
4. **结果筛选**：按相关度、时间、引用量综合排序
5. **结果呈现**：返回 top-k 论文的标题、作者、摘要、链接

## 质量标准
- 搜索结果必须与用户主题相关度 >80%
- 默认返回 8-12 篇论文，除非用户指定数量
- 结果应覆盖该主题的不同子方向，避免重复
- 每篇论文必须有可访问的链接

## 输出格式
- 论文列表：标题、作者、年份、来源、摘要（前 2 句）、链接
- 按相关度排序
- 可选：按子主题分组展示

## 约束
- 如果搜索结果为空，给出可能原因和建议（如换关键词、扩大时间范围）
- 不虚构不存在的论文
- 跟随用户语言

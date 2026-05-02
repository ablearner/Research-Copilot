---
name: academic-concepts
description: "解释学术概念、术语、方法论和研究范式，提供通俗易懂的讲解和类比。Supervisor 应调度 GeneralAnswerAgent 执行回答。"
version: 1.0.0
author: Research-Copilot
category: education
tags: [concept, explanation, tutorial, 概念, 解释, 教程, 入门, 术语, 科普]
triggers:
  - "什么是"
  - "解释.*一下"
  - "介绍.*一下"
  - "what.*is"
  - "explain.*what"
  - "how.*does.*work"
  - "怎么.*理解"
  - "原理.*是什么"
  - "通俗.*讲"
  - "简单.*说"
  - "入门.*指南"
  - "tutorial"
  - "beginner.*guide"
  - "background.*on"
  - "概念.*介绍"
  - "区别.*是什么"
  - "difference.*between"
  - "为什么.*用"
  - "有什么.*用"
  - "适用.*场景"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Academic Concepts Skill

## Agent 路由策略
Supervisor 应调度：
1. **GeneralAnswerAgent** — 基于知识回答学术概念问题
2. **ResearchQAAgent** — 可选：如果已导入论文中有相关说明，用原文佐证

## 使用场景
当用户询问学术概念、术语或方法的解释时激活此技能。典型请求：
- "什么是 Transformer 的 attention 机制"
- "解释一下 contrastive learning"
- "RL 和 imitation learning 的区别是什么"
- "为什么要用 batch normalization"

## 执行步骤

1. **概念定位**：
   - 识别用户问的具体概念或术语
   - 判断用户的知识水平（从问法推断：新手 vs 有基础）
2. **核心解释**：
   - 用一段话给出清晰的核心定义
   - 用直观类比帮助理解（如"attention 就像人看图时会聚焦关键区域"）
   - 说明为什么这个概念重要 / 它解决了什么问题
3. **技术细节**：
   - 关键公式（如果有的话，用直觉解释每一项的含义）
   - 与相关概念的对比（如 GAN vs VAE vs Diffusion Model）
   - 典型应用场景
4. **进一步学习**：
   - 推荐 1-2 篇经典论文
   - 指出学习路径中的前置知识

## 质量标准
- 解释必须分层：先直觉 → 再定义 → 再细节
- 类比必须准确，不能为了通俗而牺牲正确性
- 涉及公式时，必须用文字解释每个符号的含义
- 对比类问题必须列出具体的区别维度

## 输出格式
- 一句话定义
- 直觉类比
- 详细解释（可分小节）
- 与相关概念的对比（如适用）
- 跟随用户语言

## 约束
- 解释必须准确，不能为通俗而误导
- 承认不确定的地方，不编造
- 区分"共识"和"某些论文的观点"
- 如果用户已导入论文中有相关定义，优先引用原文

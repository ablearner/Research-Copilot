---
name: research-qa
description: "基于已导入的论文集合进行 grounded 研究问答，回答必须有论文证据支撑。Supervisor 应调度 ResearchQAAgent 执行问答，必要时先导入论文。"
version: 1.0.0
author: Research-Copilot
category: research
tags: [qa, question, answer, 问答, 提问, 询问, grounded]
triggers:
  - "这篇.*怎么.*做"
  - "论文.*中.*提到"
  - "用了什么.*方法"
  - "实验.*结果.*怎么样"
  - "how.*does.*paper"
  - "what.*method"
  - "according.*to.*paper"
  - "论文.*说"
  - "作者.*提到"
  - "数据集.*是什么"
  - "baseline.*是"
  - "loss.*function"
  - "训练.*细节"
  - "实现.*细节"
  - "implementation.*detail"
  - "reproducibility"
requires:
  tools: [hybrid_retrieve]
  skills: []
trust_level: builtin
enabled: true
---

# Research QA Skill

## Agent 路由策略
Supervisor 应按以下顺序编排 agent：
1. **ResearchKnowledgeAgent (import_papers)** — 确保相关论文已导入
2. **ResearchQAAgent** — 执行 grounded 问答

## 使用场景
当用户针对已导入的论文提出具体问题时激活此技能。典型请求：
- "这篇论文用了什么 loss function"
- "作者的实验是在什么数据集上做的"
- "这个方法的时间复杂度是多少"
- "训练时的 batch size 和 learning rate 是多少"

## 执行步骤

1. **问题理解**：
   - 识别用户问的是哪篇论文的什么问题
   - 确定问题类型：方法细节 / 实验设置 / 结果数据 / 理论推导
2. **证据检索**：
   - 在已导入论文的全文中检索相关段落
   - 优先检索方法章节、实验章节、附录
3. **回答生成**：
   - 基于检索到的证据生成回答
   - 直接引用原文中的关键句子或数据
   - 如果证据不足，明确说明
4. **补充说明**：
   - 可选：解释技术术语
   - 可选：与其他导入论文的交叉对比

## 质量标准
- 回答必须有原文证据支撑（grounded），不允许凭 LLM 知识回答
- 涉及数字的回答必须精确引用原文数据
- 如果原文未提及用户问的信息，明确回答"论文中未提及"
- 引用证据时标注来源段落位置

## 输出格式
- 直接回答用户问题
- 引用原文关键段落作为证据
- 如有必要，用列表或表格组织复杂信息
- 跟随用户语言

## 约束
- 严禁用 LLM 预训练知识替代论文原文内容
- 如果论文未导入，先提示用户导入
- 区分"论文明确说明的"和"可以推断的"，后者需标注
- 保持论文标题和术语的原始语言

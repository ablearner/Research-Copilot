---
name: knowledge-management
description: "管理论文库：导入论文、同步到 Zotero、整理分类、构建知识图谱。Supervisor 应调度 ResearchKnowledgeAgent 或 ResearchDocumentAgent 执行。"
version: 1.0.0
author: Research-Copilot
category: management
tags: [import, zotero, organize, 导入, 管理, 整理, 同步, 知识库]
triggers:
  - "导入.*论文"
  - "import.*paper"
  - "同步.*zotero"
  - "sync.*zotero"
  - "整理.*论文"
  - "organize.*paper"
  - "管理.*论文"
  - "加入.*知识库"
  - "添加.*论文"
  - "解析.*文档"
  - "parse.*document"
  - "上传.*论文"
  - "upload.*paper"
  - "导出.*论文"
  - "export.*paper"
  - "论文.*分类"
requires:
  tools: []
  skills: []
trust_level: builtin
enabled: true
---

# Knowledge Management Skill

## Agent 路由策略
Supervisor 根据具体操作选择 agent：
- **论文导入** → ResearchKnowledgeAgent (task_type=import_papers)
- **文档解析** → ResearchDocumentAgent
- **Zotero 同步** → ResearchKnowledgeAgent (task_type=sync_to_zotero)
- **上下文压缩** → ResearchKnowledgeAgent (task_type=compress_context)

## 使用场景
当用户需要管理论文库时激活此技能。典型请求：
- "把搜索到的论文导入"
- "同步这些论文到 Zotero"
- "帮我解析这个 PDF"
- "整理一下当前的论文集合"

## 执行步骤

### 论文导入
1. 确认要导入的论文列表（从搜索结果或用户指定）
2. 下载 PDF 文件
3. 解析文档结构（章节、图表、公式）
4. 构建向量索引（用于后续检索）
5. 可选：构建知识图谱
6. 更新研究任务状态

### Zotero 同步
1. 确认要同步的论文
2. 调用 Zotero API 导入论文元数据
3. 可选：指定目标 collection

### 文档解析
1. 接收用户上传的文档文件路径
2. 解析文档结构
3. 创建向量索引
4. 返回文档概览信息

## 质量标准
- 导入成功的论文必须可用于后续 QA 和分析
- 解析结果必须包含完整的章节结构
- 同步到 Zotero 后必须返回确认信息

## 输出格式
- 操作结果摘要：成功数 / 跳过数 / 失败数
- 每篇论文的处理状态
- 如有失败，说明原因
- 跟随用户语言

## 约束
- 导入操作可能耗时较长，应给用户进度反馈
- 如果 PDF 无法下载，给出明确错误信息
- 不重复导入已存在的论文（跳过）
- Zotero 同步需要 API 配置可用

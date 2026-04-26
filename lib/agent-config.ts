export interface AgentOption {
  value: string;
  label: string;
  hint: string;
}

export const PROCESSING_SKILL_OPTIONS: AgentOption[] = [
  {
    value: "",
    label: "Auto / default",
    hint: "沿用后端默认处理链路，保持当前 Parse / Index 行为。"
  },
  {
    value: "document_qa",
    label: "document_qa",
    hint: "通用文档工作流，适合常规 PDF 解析与后续问答。"
  },
  {
    value: "research_report",
    label: "research_report",
    hint: "研究报告导向的 profile，便于后续切到科研问答模式。"
  },
  {
    value: "financial_report",
    label: "financial_report",
    hint: "财报 / 图表导向的 profile，适合后续做图文联合分析。"
  }
];

export const DOCUMENT_SKILL_OPTIONS: AgentOption[] = [
  {
    value: "",
    label: "Auto / default",
    hint: "由服务端选择默认文档问答 skill，保持现有 RAG 行为。"
  },
  {
    value: "document_qa",
    label: "document_qa",
    hint: "通用文档问答，适合混合文本、图表和图谱证据。"
  },
  {
    value: "research_report",
    label: "research_report",
    hint: "偏研究报告阅读，向量检索权重更高，回答更偏研究总结。"
  },
  {
    value: "financial_report",
    label: "financial_report",
    hint: "偏财报与结构化分析，图摘要和图谱证据权重更高。"
  }
];

export const CHART_SKILL_OPTIONS: AgentOption[] = [
  {
    value: "",
    label: "Auto / default",
    hint: "默认图表理解 / 图文融合行为。纯图表追问仍走视觉模型主链路。"
  },
  {
    value: "document_qa",
    label: "document_qa",
    hint: "更偏通用图文问答。图表理解阶段若不适配会回落到默认 skill。"
  },
  {
    value: "research_report",
    label: "research_report",
    hint: "适合科研图表的图文联合解读。图表理解阶段若不适配会回落到默认 skill。"
  },
  {
    value: "financial_report",
    label: "financial_report",
    hint: "最适合财报 / 数据图表的图文联合分析，也支持图表理解阶段。"
  }
];

export const REASONING_STYLE_OPTIONS: AgentOption[] = [
  {
    value: "",
    label: "Default CoT",
    hint: "默认使用隐藏 CoT 推理，只输出答案和简短推理摘要，不暴露完整思维链。"
  },
  {
    value: "auto",
    label: "Graph / auto",
    hint: "沿用后端默认自动推理策略，保持当前 tool-first runtime 行为。"
  },
  {
    value: "cot",
    label: "CoT",
    hint: "显式使用隐藏 CoT 证据综合。"
  },
  {
    value: "react",
    label: "ReAct tools",
    hint: "切到 tool-calling / ReAct 分支，让 agent 自主决定工具调用。"
  }
];

export function findAgentOption(options: AgentOption[], value: string): AgentOption {
  return options.find((option) => option.value === value) ?? options[0];
}

import type {
  ResearchAgentRunResponse,
  ResearchAgentRuntimeMetadata,
  UnifiedActionOutput,
  UnifiedExecutionEntry,
} from "./types";
import { asNumber, asRecord, asStringArray } from "./value-coercion";

export type NormalizedUnifiedExecutionItem = {
  taskType: string;
  agentName: string | null;
  status: string;
  actionOutput: UnifiedActionOutput | null;
  executionMode: string | null;
  preferredSkillName: string | null;
};

function asNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function normalizeUnifiedActionOutput(
  value: unknown
): UnifiedActionOutput | null {
  const record = asRecord(value);
  if (!record) return null;
  const adapter = asNonEmptyString(record.unified_input_adapter);
  if (!adapter) return null;
  return {
    ...record,
    unified_input_adapter: adapter,
  };
}

export function normalizeUnifiedExecutionEntries(
  value: UnifiedExecutionEntry[] | unknown
): UnifiedExecutionEntry[] {
  if (!Array.isArray(value)) return [];
  const items: UnifiedExecutionEntry[] = [];
  for (const entry of value) {
    const record = asRecord(entry);
    if (!record) continue;
    const taskType = asNonEmptyString(record.task_type);
    if (!taskType) continue;
    items.push({
      ...record,
      task_type: taskType,
      agent_name: asNonEmptyString(record.agent_name),
      agent_to: asNonEmptyString(record.agent_to),
      status: asNonEmptyString(record.status) ?? "planned",
      action_output: normalizeUnifiedActionOutput(record.action_output),
      execution_mode: asNonEmptyString(record.execution_mode),
      preferred_skill_name: asNonEmptyString(record.preferred_skill_name),
    });
  }
  return items;
}

export function normalizeResearchAgentRuntimeMetadata(
  value: ResearchAgentRuntimeMetadata | unknown
): ResearchAgentRuntimeMetadata {
  const metadata = asRecord(value);
  if (!metadata) {
    return {
      unified_delegation_plan: [],
      unified_agent_results: [],
    };
  }
  return {
    ...metadata,
    unified_supervisor_mode: asNonEmptyString(metadata.unified_supervisor_mode),
    manager_decision_count: asNumber(metadata.manager_decision_count),
    supervisor_action_trace_count: asNumber(metadata.supervisor_action_trace_count),
    agent_result_count: asNumber(metadata.agent_result_count),
    unified_delegation_plan: normalizeUnifiedExecutionEntries(
      metadata.unified_delegation_plan
    ),
    unified_agent_results: normalizeUnifiedExecutionEntries(
      metadata.unified_agent_results
    ),
  };
}

export function normalizeResearchAgentRunResponse(
  response: ResearchAgentRunResponse
): ResearchAgentRunResponse {
  return {
    ...response,
    metadata: normalizeResearchAgentRuntimeMetadata(response.metadata),
  };
}

export function parseUnifiedExecutionItems(
  value: UnifiedExecutionEntry[] | unknown
): NormalizedUnifiedExecutionItem[] {
  return normalizeUnifiedExecutionEntries(value).map((entry) => ({
    taskType: entry.task_type,
    agentName: entry.agent_name ?? entry.agent_to ?? null,
    status:
      typeof entry.status === "string" && entry.status.trim()
        ? entry.status.trim()
        : "planned",
    actionOutput: normalizeUnifiedActionOutput(entry.action_output),
    executionMode:
      typeof entry.execution_mode === "string" && entry.execution_mode.trim()
        ? entry.execution_mode.trim()
        : null,
    preferredSkillName:
      typeof entry.preferred_skill_name === "string" &&
      entry.preferred_skill_name.trim()
        ? entry.preferred_skill_name.trim()
        : null,
  }));
}

export function formatUnifiedActionAdapterLabel(adapter: string): string {
  const labels: Record<string, string> = {
    literature_search_input: "文献检索",
    review_draft_input: "综述草稿",
    collection_qa_input: "研究问答",
    paper_import_input: "论文导入",
    document_understanding_input: "文档理解",
    chart_understanding_input: "图表理解",
    paper_analysis_input: "论文分析",
    context_compression_input: "上下文压缩",
  };
  return labels[adapter] ?? adapter.replace(/_/g, " ");
}

export function summarizeUnifiedActionOutput(
  output: UnifiedActionOutput | null
): string[] {
  if (!output) return [];
  switch (output.unified_input_adapter) {
    case "literature_search_input": {
      const paperCount = asNumber(output.paper_count);
      const warnings = asStringArray(output.warnings);
      const summary: string[] = [];
      if (paperCount != null) summary.push(`papers ${paperCount}`);
      if (warnings.length) summary.push(`warnings ${warnings.length}`);
      if (typeof output.report_id === "string" && output.report_id.trim()) {
        summary.push("report ready");
      }
      return summary;
    }
    case "review_draft_input": {
      const wordCount = asNumber(output.report_word_count);
      const retryCount = asNumber(output.retry_count);
      const summary: string[] = [];
      if (wordCount != null) summary.push(`words ${wordCount}`);
      if (output.report_has_citations === true) summary.push("with citations");
      if (retryCount != null) summary.push(`retries ${retryCount}`);
      return summary;
    }
    case "collection_qa_input": {
      const evidenceCount = asNumber(output.evidence_count);
      const documentIds = asStringArray(output.document_ids);
      const confidence = asNumber(output.confidence);
      const summary: string[] = [];
      if (evidenceCount != null) summary.push(`evidence ${evidenceCount}`);
      if (documentIds.length) summary.push(`docs ${documentIds.length}`);
      if (confidence != null) summary.push(`confidence ${confidence.toFixed(2)}`);
      return summary;
    }
    case "paper_import_input": {
      const importedCount = asNumber(output.imported_count);
      const skippedCount = asNumber(output.skipped_count);
      const failedCount = asNumber(output.failed_count);
      const summary: string[] = [];
      if (importedCount != null) summary.push(`imported ${importedCount}`);
      if (skippedCount != null) summary.push(`skipped ${skippedCount}`);
      if (failedCount != null) summary.push(`failed ${failedCount}`);
      return summary;
    }
    case "document_understanding_input": {
      const pageCount = asNumber(output.page_count);
      const summary: string[] = [];
      if (pageCount != null) summary.push(`pages ${pageCount}`);
      if (
        typeof output.index_status === "string" &&
        output.index_status.trim()
      ) {
        summary.push(output.index_status.trim());
      }
      return summary;
    }
    case "chart_understanding_input": {
      const summary: string[] = [];
      if (
        typeof output.chart_type === "string" &&
        output.chart_type.trim()
      ) {
        summary.push(output.chart_type.trim());
      }
      if (
        typeof output.document_id === "string" &&
        output.document_id.trim()
      ) {
        summary.push(`doc ${output.document_id.trim()}`);
      }
      return summary;
    }
    case "paper_analysis_input": {
      const paperCount = asNumber(output.paper_count);
      const recommendedPaperIds = asStringArray(output.recommended_paper_ids);
      const summary: string[] = [];
      if (paperCount != null) summary.push(`papers ${paperCount}`);
      if (
        typeof output.analysis_focus === "string" &&
        output.analysis_focus.trim()
      ) {
        summary.push(output.analysis_focus.trim());
      }
      if (recommendedPaperIds.length) {
        summary.push(`recommended ${recommendedPaperIds.length}`);
      }
      return summary;
    }
    case "context_compression_input": {
      const paperCount = asNumber(output.paper_count);
      const summaryCount = asNumber(output.summary_count);
      const levels = asStringArray(output.levels);
      const summary: string[] = [];
      if (paperCount != null) summary.push(`papers ${paperCount}`);
      if (summaryCount != null) summary.push(`summaries ${summaryCount}`);
      if (levels.length) summary.push(`levels ${levels.length}`);
      return summary;
    }
    default:
      return [];
  }
}

export function statusBadgeClass(status: string): string {
  if (status === "succeeded") {
    return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200/70";
  }
  if (status === "failed") {
    return "bg-red-50 text-red-700 ring-1 ring-red-200/70";
  }
  if (status === "skipped") {
    return "bg-amber-50 text-amber-700 ring-1 ring-amber-200/70";
  }
  return "bg-slate-100 text-slate-600 ring-1 ring-slate-200/80";
}

"use client";

import { Layers } from "lucide-react";
import type { ResearchAgentRuntimeMetadata } from "@/lib/types";
import {
  formatUnifiedActionAdapterLabel,
  normalizeResearchAgentRuntimeMetadata,
  parseUnifiedExecutionItems,
  statusBadgeClass,
  summarizeUnifiedActionOutput,
} from "@/lib/unified-runtime";
import { asNumber, buildListKey } from "@/lib/value-coercion";

export function SupervisorExecutionCard({
  metadata,
}: {
  metadata?: ResearchAgentRuntimeMetadata | null;
}) {
  if (!metadata) return null;
  const runtimeMetadata = normalizeResearchAgentRuntimeMetadata(metadata);

  const unifiedDelegationPlan = parseUnifiedExecutionItems(
    runtimeMetadata.unified_delegation_plan
  );
  const unifiedAgentResults = parseUnifiedExecutionItems(
    runtimeMetadata.unified_agent_results
  );

  if (!unifiedDelegationPlan.length && !unifiedAgentResults.length) {
    return null;
  }

  const runtimeMode =
    typeof runtimeMetadata.unified_supervisor_mode === "string" &&
    runtimeMetadata.unified_supervisor_mode.trim()
      ? runtimeMetadata.unified_supervisor_mode.trim()
      : null;
  const decisionCount = asNumber(runtimeMetadata.manager_decision_count);
  const actionTraceCount = asNumber(runtimeMetadata.supervisor_action_trace_count);
  const resultCount = asNumber(runtimeMetadata.agent_result_count);
  const executionItems = unifiedDelegationPlan.length
    ? unifiedDelegationPlan
    : unifiedAgentResults;

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[13px] font-semibold text-gray-800">
            <Layers className="h-3.5 w-3.5 text-gray-500" />
            执行摘要
          </div>
          {runtimeMode ? (
            <span className="text-[11px] text-gray-400">{runtimeMode}</span>
          ) : null}
        </div>

        <div className="mt-2 grid grid-cols-3 gap-2 text-[12px]">
          <div className="rounded-lg bg-gray-50 px-3 py-2">
            <div className="text-[10px] font-medium uppercase tracking-wider text-gray-400">decisions</div>
            <div className="mt-0.5 font-medium text-gray-900">{decisionCount ?? executionItems.length}</div>
          </div>
          <div className="rounded-lg bg-gray-50 px-3 py-2">
            <div className="text-[10px] font-medium uppercase tracking-wider text-gray-400">traces</div>
            <div className="mt-0.5 font-medium text-gray-900">{actionTraceCount ?? 0}</div>
          </div>
          <div className="rounded-lg bg-gray-50 px-3 py-2">
            <div className="text-[10px] font-medium uppercase tracking-wider text-gray-400">results</div>
            <div className="mt-0.5 font-medium text-gray-900">{resultCount ?? unifiedAgentResults.length}</div>
          </div>
        </div>

        <div className="mt-4 space-y-3">
          {executionItems.map((item, index) => {
            const actionSummary = summarizeUnifiedActionOutput(item.actionOutput);
            return (
              <div
                key={buildListKey(
                  "supervisor-execution-item",
                  `${item.taskType}:${item.agentName}`,
                  index
                )}
                className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2.5"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium text-gray-900">
                      {item.taskType}
                    </div>
                    <div className="mt-0.5 text-[11px] leading-5 text-gray-500">
                      {item.agentName ?? "unknown agent"}
                      {item.executionMode ? ` · ${item.executionMode}` : ""}
                      {item.preferredSkillName ? ` · ${item.preferredSkillName}` : ""}
                    </div>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.14em] ${statusBadgeClass(item.status)}`}
                  >
                    {item.status}
                  </span>
                </div>
                {item.actionOutput ? (
                  <>
                    <div className="mt-3">
                      <span className="badge-info">
                        {formatUnifiedActionAdapterLabel(
                          item.actionOutput.unified_input_adapter
                        )}
                      </span>
                    </div>
                    {actionSummary.length ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {actionSummary.map((entry, summaryIndex) => (
                          <span
                            key={buildListKey(
                              `supervisor-execution-summary:${item.taskType}`,
                              entry,
                              summaryIndex
                            )}
                            className="badge-muted"
                          >
                            {entry}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="mt-2 text-[12px] leading-5 text-gray-500">
                    这一步当前还没有结构化 action output。
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <details className="mt-3 overflow-hidden rounded-lg border border-gray-200 bg-white">
          <summary className="cursor-pointer bg-gray-50 px-3 py-2 text-[12px] font-medium text-gray-600">
            查看 unified runtime JSON
          </summary>
          <pre className="max-h-80 overflow-auto bg-gray-900 p-3 text-[11px] leading-5 text-gray-100">
            {JSON.stringify(
              {
                unified_delegation_plan:
                  runtimeMetadata.unified_delegation_plan ?? [],
                unified_agent_results:
                  runtimeMetadata.unified_agent_results ?? [],
              },
              null,
              2
            )}
          </pre>
        </details>
      </div>
    </div>
  );
}

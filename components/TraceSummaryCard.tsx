"use client";

import { GitBranch } from "lucide-react";
import type { QAResponse, ToolTrace } from "@/lib/types";

export function TraceSummaryCard({ qa }: { qa?: QAResponse | null }) {
  if (!qa) {
    return (
      <section className="card">
        <h2 className="flex items-center gap-2 text-sm font-bold text-ink"><GitBranch className="h-4 w-4 text-accent" />Trace summary</h2>
        <p className="mt-3 text-xs text-muted">Run a QA request to inspect runtime latency and node traces.</p>
      </section>
    );
  }

  const metadata = qa.metadata ?? {};
  const toolTraces = (Array.isArray(metadata.tool_traces) ? metadata.tool_traces : []) as ToolTrace[];
  const totalLatency = metadata.runtime_total_latency_ms;
  const traceId = metadata.trace_id;

  return (
    <section className="card">
      <div className="flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-sm font-bold text-ink"><GitBranch className="h-4 w-4 text-accent" />Trace summary</h2>
        <span className="badge-muted">
          total {typeof totalLatency === "number" ? `${totalLatency} ms` : "unknown"}
        </span>
      </div>
      <p className="mt-2 break-all text-xs text-muted">trace_id: {typeof traceId === "string" ? traceId : "unknown"}</p>
      {!toolTraces.length ? <p className="mt-3 text-xs text-muted">No node trace available.</p> : null}
      <div className="mt-3 space-y-2">
        {toolTraces.map((trace, index) => (
          <div key={`${trace.node_name ?? "node"}-${index}`} className="rounded-2xl border border-line bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold text-slate-800">{trace.node_name ?? trace.tool_name ?? "node"}</p>
              <span className="badge-info">{trace.latency_ms != null ? `${trace.latency_ms} ms` : "n/a"}</span>
            </div>
            <p className="mt-1 text-xs text-muted">{trace.tool_name ?? "tool"}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

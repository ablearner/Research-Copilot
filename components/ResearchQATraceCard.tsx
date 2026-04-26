"use client";

import {
  buildResearchQATraceSummary,
  formatResearchQARouteLabel,
  formatResearchQARuntimeLabel,
} from "@/lib/research-payloads";
import { buildListKey } from "@/lib/value-coercion";

export function ResearchQATraceCard({
  metadataSource,
  className = "",
}: {
  metadataSource: unknown;
  className?: string;
}) {
  const trace = buildResearchQATraceSummary(metadataSource);
  if (!trace) return null;

  return (
    <div
      className={`rounded-2xl border border-slate-200/80 bg-slate-50/80 px-4 py-3 text-[12px] leading-6 text-slate-700 ${className}`.trim()}
    >
      <div className="flex flex-wrap items-center gap-2">
        {trace.route && (
          <span className="badge-info">
            route: {formatResearchQARouteLabel(trace.route)}
          </span>
        )}
        {trace.runtime && (
          <span className="badge-muted">
            runtime: {formatResearchQARuntimeLabel(trace.runtime)}
          </span>
        )}
        {trace.confidence != null && (
          <span className="badge-muted">
            confidence: {trace.confidence.toFixed(2)}
          </span>
        )}
      </div>
      {trace.rationale && (
        <div className="mt-2 text-[12px] leading-5 text-slate-600">
          {trace.rationale}
        </div>
      )}
      {trace.anchorRationale && (
        <div className="mt-2 rounded-xl border border-sky-100 bg-sky-50/70 px-3 py-2 text-[12px] leading-5 text-sky-800">
          选图原因：{trace.anchorRationale}
        </div>
      )}
      {Object.keys(trace.anchor).length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {Object.entries(trace.anchor).map(([key, value], index) => (
            <span
              key={buildListKey(`qa-trace-anchor:${key}`, String(value), index)}
              className="rounded-full border border-slate-200 bg-white px-2.5 py-0.5 font-mono text-[11px] text-slate-600"
            >
              {key}={String(value)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

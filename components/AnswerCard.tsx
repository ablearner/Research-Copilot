"use client";

import type { QAResponse } from "@/lib/types";

export function AnswerCard({ qa }: { qa?: QAResponse | null }) {
  if (!qa) {
    return <div className="card border-dashed text-sm text-muted">Ask a question to inspect the answer, confidence, evidence, and retrieval trace.</div>;
  }
  const metadata = qa.metadata ?? {};
  const reasoningSummary = (metadata.reasoning_summary ?? {}) as Record<string, string>;
  const warnings = Array.isArray(metadata.warnings) ? (metadata.warnings as string[]) : [];
  const evidenceMix = (metadata.evidence_mix ?? {}) as Record<string, number>;
  return (
    <section className="card">
      <p className="kv-label">Question</p>
      <p className="mt-1 text-[14px] leading-7 text-slate-800">{qa.question}</p>
      <div className="mt-5 flex items-center justify-between gap-3">
        <p className="kv-label">Answer</p>
        <span className="badge-success">confidence {qa.confidence ?? "empty"}</span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-[15px] leading-8 text-ink">{qa.answer}</p>
      <div className="mt-6 grid grid-cols-2 gap-2 text-[12px] leading-5 text-slate-600">
        <div className="metric-tile">vector hits {evidenceMix.vector_hits ?? 0}</div>
        <div className="metric-tile">graph hits {evidenceMix.graph_hits ?? 0}</div>
        <div className="metric-tile">summary hits {evidenceMix.graph_summary_hits ?? 0}</div>
        <div className="metric-tile">evidence count {evidenceMix.evidence_count ?? qa.evidence_bundle.evidences.length}</div>
      </div>
      {warnings.length ? (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-4">
          <p className="kv-label text-amber-800">Warnings</p>
          <ul className="mt-2 space-y-1 text-[13px] leading-6 text-amber-900">
            {warnings.map((warning, index) => (
              <li key={`${warning}:${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {!!Object.keys(reasoningSummary).length ? (
        <div className="mt-4 rounded-2xl border border-line bg-surface p-4">
          <p className="kv-label text-slate-700">Reasoning Summary</p>
          <div className="mt-3 space-y-3">
            {Object.entries(reasoningSummary).map(([key, value]) => (
              <div key={key}>
                <p className="kv-label">{key}</p>
                <p className="mt-1 text-[13px] leading-6 text-slate-800">{value}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

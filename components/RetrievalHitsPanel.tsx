"use client";

import type { RetrievalHit } from "@/lib/types";

export function RetrievalHitsPanel({ hits }: { hits?: RetrievalHit[] }) {
  return (
    <section className="card">
      <h2 className="text-sm font-bold text-ink">Retrieval hits</h2>
      {!hits?.length ? <p className="mt-3 text-xs text-muted">No retrieval hits yet.</p> : null}
      <div className="mt-3 max-h-72 space-y-3 overflow-auto pr-1">
        {hits?.map((hit) => (
          <article key={hit.id} className="rounded-2xl border border-line bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-semibold text-slate-800">{hit.source_type}</span>
              <span className="badge-muted">merged {hit.merged_score ?? "empty"}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
              {hit.vector_score != null ? <span className="badge-info">vector {hit.vector_score}</span> : null}
              {hit.graph_score != null ? <span className="badge-success">graph {hit.graph_score}</span> : null}
            </div>
            <p className="mt-3 line-clamp-4 text-sm leading-6 text-slate-700">{hit.content ?? "No content returned."}</p>
            <p className="mt-3 break-words text-xs text-muted">{hit.document_id ?? "no doc"} · {hit.source_id}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

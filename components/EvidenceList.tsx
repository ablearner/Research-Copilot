"use client";

import { Library } from "lucide-react";
import type { Evidence } from "@/lib/types";

export function EvidenceList({ evidences }: { evidences?: Evidence[] }) {
  return (
    <section className="card">
      <h2 className="flex items-center gap-2 text-sm font-bold text-ink"><Library className="h-4 w-4 text-accent" />Evidence</h2>
      {!evidences?.length ? <p className="mt-3 text-xs text-muted">No evidence yet.</p> : null}
      <div className="mt-3 max-h-64 space-y-3 overflow-auto pr-1">
        {evidences?.map((item, index) => (
          <article
            key={`${item.id}:${item.source_id ?? "source"}:${index}`}
            className="rounded-2xl border border-line bg-white p-4 shadow-sm"
          >
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-semibold text-slate-800">{item.source_type}</span>
              <span className="badge-info">score {item.score ?? "empty"}</span>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">{item.snippet ?? "No snippet returned."}</p>
            <p className="mt-3 break-words text-xs text-muted">page {item.page_number ?? "empty"} · {item.source_id ?? item.id}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

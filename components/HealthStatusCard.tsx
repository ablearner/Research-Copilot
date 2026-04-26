"use client";

import { Activity, RefreshCw } from "lucide-react";
import type { HealthResponse, RequestState } from "@/lib/types";

interface Props {
  health?: HealthResponse | null;
  state: RequestState;
  error?: string | null;
  onRefresh: () => void;
}

export function HealthStatusCard({ health, state, error, onRefresh }: Props) {
  const healthy = health?.status === "ok";
  return (
    <section className="card">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600"><Activity className="h-4 w-4" /></span>
            <h2 className="break-words text-[15px] font-semibold leading-6 text-ink">系统健康状态</h2>
          </div>
          <p className="mt-1 break-words text-[12px] leading-5 text-muted">FastAPI / Runtime / Store connectivity</p>
        </div>
        <span className={`${healthy ? "badge-success" : "badge-warning"} shrink-0`}>
          {state === "loading" ? "checking" : health?.status ?? "unknown"}
        </span>
      </div>
      {health ? (
        <div className="mt-4 grid grid-cols-1 gap-2 text-[12px] leading-5 text-slate-600 sm:grid-cols-2">
          <div className="metric-tile break-words">app: {health.app_name}</div>
          <div className="metric-tile break-words">env: {health.app_env}</div>
          <div className="metric-tile break-words">llm: {health.llm_provider}</div>
          <div className="metric-tile break-words">embed: {health.embedding_provider}</div>
          <div className="metric-tile break-words">vector: {health.vector_store_provider}</div>
          <div className="metric-tile break-words">graph: {health.graph_store_provider}</div>
          <div className="metric-tile break-words">checkpointer: {health.checkpointer_backend ?? "unknown"}</div>
          <div className="metric-tile break-words">memory: {health.session_memory_backend ?? "unknown"}</div>
        </div>
      ) : null}
      {error ? <p className="mt-4 rounded-xl bg-red-50 p-3 text-xs text-red-700">{error}</p> : null}
      <button onClick={onRefresh} disabled={state === "loading"} className="btn-accent mt-4 w-full">
        <span className="inline-flex items-center justify-center gap-2"><RefreshCw className="h-4 w-4" />刷新健康检查</span>
      </button>
    </section>
  );
}

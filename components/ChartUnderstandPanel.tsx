"use client";

import { useState } from "react";
import { BarChart3 } from "lucide-react";
import { CHART_SKILL_OPTIONS, findAgentOption } from "@/lib/agent-config";
import type { ChartUnderstandRequest, ChartUnderstandResponse, JsonObject, RequestState } from "@/lib/types";

interface Props {
  defaultDocumentId?: string | null;
  defaultImagePath?: string | null;
  result?: ChartUnderstandResponse | null;
  state: RequestState;
  error?: string | null;
  skillName: string;
  onSkillChange: (value: string) => void;
  onRun: (payload: ChartUnderstandRequest) => void;
  onReset?: () => void;
}

function Field({ label, value }: { label: string; value?: unknown }) {
  return (
    <div className="kv-row">
      <p className="kv-label">{label}</p>
      <p className="kv-value font-medium text-slate-800">{value == null || value === "" ? "empty" : String(value)}</p>
    </div>
  );
}

export function ChartUnderstandPanel({
  defaultDocumentId,
  defaultImagePath,
  result,
  state,
  error,
  skillName,
  onSkillChange,
  onRun,
  onReset
}: Props) {
  const [imagePathOverride, setImagePathOverride] = useState<string | null>(
    null
  );
  const [documentIdOverride, setDocumentIdOverride] = useState<string | null>(
    null
  );
  const [pageId, setPageId] = useState("page-1");
  const [pageNumber, setPageNumber] = useState(1);
  const [chartId, setChartId] = useState("chart-1");
  const activeSkill = findAgentOption(CHART_SKILL_OPTIONS, skillName);

  const chart = result?.result.chart;
  const imagePath = imagePathOverride ?? defaultImagePath ?? "";
  const documentId = documentIdOverride ?? defaultDocumentId ?? "";

  function submit() {
    onRun({
      image_path: imagePath,
      document_id: documentId || defaultDocumentId || "doc-1",
      page_id: pageId || "page-1",
      page_number: pageNumber,
      chart_id: chartId || "chart-1",
      context: {} satisfies JsonObject,
      skill_name: skillName || undefined
    });
  }

  return (
    <section className="card">
      <h2 className="section-title flex items-center gap-2"><BarChart3 className="h-4 w-4 text-accent" />Chart understanding</h2>
      <div className="mt-4 grid gap-2.5">
        <input value={imagePath} onChange={(event) => setImagePathOverride(event.target.value)} placeholder=".data/uploads/chart.png" className="input-base" />
        <div className="grid grid-cols-2 gap-2">
          <input value={documentId} onChange={(event) => setDocumentIdOverride(event.target.value)} placeholder={defaultDocumentId ?? "document_id"} className="input-base" />
          <input value={pageId} onChange={(event) => setPageId(event.target.value)} placeholder="page_id" className="input-base" />
          <input value={pageNumber} min={1} onChange={(event) => setPageNumber(Number(event.target.value) || 1)} type="number" className="input-base" />
          <input value={chartId} onChange={(event) => setChartId(event.target.value)} placeholder="chart_id" className="input-base" />
        </div>
        <label>
          <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Chart skill</span>
          <select value={skillName} onChange={(event) => onSkillChange(event.target.value)} className="input-base w-full">
            {CHART_SKILL_OPTIONS.map((option) => (
              <option key={option.value || "auto"} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="mt-2 text-[11px] leading-5 text-slate-500">{activeSkill.hint}</p>
        </label>
        <div className="flex gap-2">
          <button disabled={!imagePath || state === "loading"} onClick={submit} className="btn-accent">
            {state === "loading" ? "Running..." : "Understand chart"}
          </button>
          {state === "loading" || error ? (
            <button onClick={onReset} className="btn-secondary">
              Reset
            </button>
          ) : null}
        </div>
        {state === "loading" ? <p className="section-subtitle">Waiting for /charts/understand to finish. Check the FastAPI terminal for the POST request.</p> : null}
      </div>
      {error ? <p className="mt-4 rounded-xl bg-red-50 p-3 text-[12px] leading-5 text-red-700">{error}</p> : null}
      {!chart ? <p className="mt-4 section-subtitle">Run chart understanding to inspect the structured result.</p> : null}
      {chart ? (
        <div className="mt-5 rounded-2xl border border-line bg-white p-4 shadow-sm">
          <Field label="chart_type" value={chart.chart_type} />
          <Field label="title" value={chart.title} />
          <Field label="caption" value={chart.caption} />
          <Field label="x_axis" value={chart.x_axis?.label ?? chart.x_axis?.name} />
          <Field label="y_axis" value={chart.y_axis?.label ?? chart.y_axis?.name} />
          <Field label="series" value={chart.series.map((series) => `${series.name} (${series.points.length})`).join(", ")} />
          <Field label="summary" value={chart.summary} />
          <Field label="confidence" value={chart.confidence} />
          <Field label="graph_text" value={result?.result.graph_text} />
        </div>
      ) : null}
    </section>
  );
}

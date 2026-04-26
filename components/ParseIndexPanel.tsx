"use client";

import { findAgentOption, PROCESSING_SKILL_OPTIONS } from "@/lib/agent-config";
import type { GraphIndexStats, IndexDocumentResponse, ParseDocumentResponse, RequestState } from "@/lib/types";

interface Props {
  parseResult?: ParseDocumentResponse | null;
  indexResult?: IndexDocumentResponse | null;
  parseState: RequestState;
  indexState: RequestState;
  parseError?: string | null;
  indexError?: string | null;
  includeGraph: boolean;
  includeEmbeddings: boolean;
  processingSkillName: string;
  canParse: boolean;
  canIndex: boolean;
  onIncludeGraphChange: (value: boolean) => void;
  onIncludeEmbeddingsChange: (value: boolean) => void;
  onProcessingSkillChange: (value: string) => void;
  onParse: () => void;
  onIndex: () => void;
}

function metric(label: string, value?: number | string | null) {
  return <div className="metric-tile"><p className="text-xs text-muted">{label}</p><p className="mt-1 text-sm font-semibold text-slate-800">{value ?? "empty"}</p></div>;
}

function isGraphIndexStats(value: unknown): value is GraphIndexStats {
  return typeof value === "object" && value !== null && "node_count" in value && "edge_count" in value;
}

export function ParseIndexPanel(props: Props) {
  const processingSkill = findAgentOption(PROCESSING_SKILL_OPTIONS, props.processingSkillName);
  const parsed = props.parseResult?.parsed_document;
  const textBlocks = parsed?.pages.reduce((sum, page) => sum + page.text_blocks.length, 0);
  const result = props.indexResult?.result;
  const graph = result?.graph_index ?? result?.graph;
  const graphExtraction = result?.graph_extraction;
  const textIndex = result?.text_embedding_index ?? result?.text_embeddings;
  const pageIndex = result?.page_embedding_index ?? result?.page_embeddings;
  const chartIndex = result?.chart_embedding_index ?? result?.chart_embeddings;
  const graphIndexError = isGraphIndexStats(graph) ? graph.error_message : null;
  const graphExtractionWarning = graphExtraction?.status === "failed"
    ? graphExtraction.error_message ?? "Graph extraction failed."
    : null;
  const nonFatalIndexWarning = graphExtractionWarning ?? graphIndexError;

  return (
    <section className="card">
      <h2 className="text-sm font-bold text-ink">Parse and index</h2>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <button disabled={!props.canParse || props.parseState === "loading"} onClick={props.onParse} className="btn-accent">
          {props.parseState === "loading" ? "Parsing..." : "Parse"}
        </button>
        <button disabled={!props.canIndex || props.indexState === "loading"} onClick={props.onIndex} className="btn-secondary">
          {props.indexState === "loading" ? "Indexing..." : "Index"}
        </button>
      </div>
      <div className="mt-4 rounded-2xl border border-line bg-white/80 p-3">
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <label className="min-w-0">
            <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Processing skill</span>
            <select value={props.processingSkillName} onChange={(event) => props.onProcessingSkillChange(event.target.value)} className="input-base w-full">
              {PROCESSING_SKILL_OPTIONS.map((option) => (
                <option key={option.value || "auto"} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <div className="rounded-xl bg-surface px-3 py-2.5 text-[12px] leading-5 text-slate-600">
            <p className="font-medium text-slate-700">当前 profile</p>
            <p className="mt-1">{processingSkill.hint}</p>
          </div>
        </div>
      </div>
      <label className="mt-4 flex items-center justify-between rounded-xl bg-surface px-3 py-2 text-sm"><span>include_graph</span><input type="checkbox" checked={props.includeGraph} onChange={(event) => props.onIncludeGraphChange(event.target.checked)} /></label>
      <label className="mt-2 flex items-center justify-between rounded-xl bg-surface px-3 py-2 text-sm"><span>include_embeddings</span><input type="checkbox" checked={props.includeEmbeddings} onChange={(event) => props.onIncludeEmbeddingsChange(event.target.checked)} /></label>
      <div className="mt-4 grid grid-cols-2 gap-2">
        {metric("parsed status", parsed?.status ?? props.parseResult?.status)}
        {metric("pages", parsed?.pages.length)}
        {metric("text blocks", textBlocks)}
        {metric("index status", props.indexResult?.status ?? result?.status)}
        {metric("text emb", textIndex?.record_count)}
        {metric("page emb", pageIndex?.record_count)}
        {metric("chart emb", chartIndex?.record_count)}
        {metric("graph nodes", isGraphIndexStats(graph) ? graph.node_count : graphExtraction?.nodes.length)}
        {metric("graph edges", isGraphIndexStats(graph) ? graph.edge_count : graphExtraction?.edges.length)}
        {metric("triples", graphExtraction?.triples.length)}
      </div>
      {props.parseError ? <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">{props.parseError}</p> : null}
      {props.indexError ? <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">{props.indexError}</p> : null}
      {!props.indexError && nonFatalIndexWarning ? (
        <p className="mt-3 rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-800">
          Graph indexing was partially skipped: {nonFatalIndexWarning}
          {" "}
          {textIndex?.record_count ? "Text embeddings are available, so document QA can still continue with vector retrieval." : ""}
        </p>
      ) : null}
    </section>
  );
}

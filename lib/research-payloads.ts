import type {
  ComparePapersResult,
  ContextCompressionSummary,
  ImportPapersResponse,
  JsonObject,
  PaperAnalysisResult,
  PaperCandidate,
  RecommendPapersResult,
  ResearchAgentTraceStep,
  ResearchQAPaperScope,
  ResearchQATraceSummary,
  ResearchPaperFigurePreview,
  ResearchSource,
  ResearchVisualAnchor,
  ResearchWorkspaceState,
} from "./types";
import { asNumber, asRecord, asStringArray } from "./value-coercion";

function asNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function parsePaperCandidate(value: unknown): PaperCandidate | null {
  const record = asRecord(value);
  if (!record) return null;
  const paperId =
    typeof record.paper_id === "string" ? record.paper_id : null;
  const title = typeof record.title === "string" ? record.title : null;
  const source = typeof record.source === "string" ? record.source : null;
  if (!paperId || !title || !source) return null;
  const citations = asNumber(record.citations);
  const year = asNumber(record.year);
  const relevanceScore = asNumber(record.relevance_score);
  return {
    paper_id: paperId,
    title,
    authors: asStringArray(record.authors),
    abstract: typeof record.abstract === "string" ? record.abstract : "",
    year,
    venue: typeof record.venue === "string" ? record.venue : null,
    source: source as ResearchSource,
    doi: typeof record.doi === "string" ? record.doi : null,
    arxiv_id: typeof record.arxiv_id === "string" ? record.arxiv_id : null,
    pdf_url: typeof record.pdf_url === "string" ? record.pdf_url : null,
    url: typeof record.url === "string" ? record.url : null,
    citations,
    is_open_access:
      typeof record.is_open_access === "boolean"
        ? record.is_open_access
        : null,
    published_at:
      typeof record.published_at === "string" ? record.published_at : null,
    relevance_score: relevanceScore,
    summary: typeof record.summary === "string" ? record.summary : null,
    ingest_status:
      typeof record.ingest_status === "string"
        ? (record.ingest_status as PaperCandidate["ingest_status"])
        : "not_selected",
    metadata: asRecord(record.metadata) ?? {},
  };
}

export function papersFromPayload(
  payload: Record<string, unknown>
): PaperCandidate[] {
  if (!Array.isArray(payload.papers)) return [];
  return payload.papers
    .map((item) => parsePaperCandidate(item))
    .filter((paper): paper is PaperCandidate => Boolean(paper));
}

export function comparisonFromPayload(
  payload: Record<string, unknown>
): ComparePapersResult | null {
  const comparison = asRecord(payload.comparison);
  if (!comparison) return null;
  const summary =
    typeof comparison.summary === "string" ? comparison.summary : "";
  const table = Array.isArray(comparison.table)
    ? comparison.table
        .map((item) => {
          const row = asRecord(item);
          const values = asRecord(row?.values);
          const dimension =
            row && typeof row.dimension === "string" ? row.dimension : null;
          if (!dimension || !values) return null;
          return {
            dimension,
            values: Object.fromEntries(
              Object.entries(values).flatMap(([key, value]) =>
                typeof value === "string" ? [[key, value]] : []
              )
            ),
          };
        })
        .filter((item): item is ComparePapersResult["table"][number] =>
          Boolean(item)
        )
    : [];
  if (!summary && table.length === 0) return null;
  return { summary, table };
}

export function recommendationFromPayload(
  payload: Record<string, unknown>
): RecommendPapersResult | null {
  const recommendationRoot = asRecord(payload.recommendations);
  if (!recommendationRoot || !Array.isArray(recommendationRoot.recommendations)) {
    return null;
  }
  const recommendations: RecommendPapersResult["recommendations"] = [];
  for (const item of recommendationRoot.recommendations) {
    const record = asRecord(item);
    const paperId =
      record && typeof record.paper_id === "string" ? record.paper_id : null;
    const title =
      record && typeof record.title === "string" ? record.title : null;
    if (!paperId || !title) continue;
    recommendations.push({
      paper_id: paperId,
      title,
      reason:
        record && typeof record.reason === "string" ? record.reason : "",
      source:
        record && typeof record.source === "string" ? record.source : null,
      year: record ? asNumber(record.year) : null,
      url: record && typeof record.url === "string" ? record.url : null,
    });
  }
  if (recommendations.length === 0) return null;
  return { recommendations };
}

export function paperAnalysisFromPayload(
  payload: Record<string, unknown>
): PaperAnalysisResult | null {
  const root = asRecord(payload.paper_analysis);
  if (!root) return null;
  const answer = typeof root.answer === "string" ? root.answer : "";
  const focus = typeof root.focus === "string" ? root.focus : "analysis";
  const recommendedPaperIds = asStringArray(root.recommended_paper_ids);
  const keyPoints = asStringArray(root.key_points);
  if (!answer && keyPoints.length === 0 && recommendedPaperIds.length === 0) {
    return null;
  }
  return {
    answer,
    focus,
    key_points: keyPoints,
    recommended_paper_ids: recommendedPaperIds,
  };
}

export function contextCompressionFromPayload(
  payload: Record<string, unknown>
): ContextCompressionSummary | null {
  const compression = asRecord(payload.context_compression);
  if (!compression) return null;
  const paperCount = asNumber(compression.paper_count);
  const summaryCount = asNumber(compression.summary_count);
  if (paperCount == null || summaryCount == null) return null;
  return {
    paper_count: paperCount,
    summary_count: summaryCount,
    levels: asStringArray(compression.levels),
    compressed_paper_ids: asStringArray(compression.compressed_paper_ids),
  };
}

export function parseResearchPaperFigurePreview(
  value: unknown
): ResearchPaperFigurePreview | null {
  const record = asRecord(value);
  if (!record) return null;
  const figureId =
    typeof record.figure_id === "string" ? record.figure_id : null;
  const paperId =
    typeof record.paper_id === "string" ? record.paper_id : null;
  const documentId =
    typeof record.document_id === "string" ? record.document_id : null;
  const pageId =
    typeof record.page_id === "string" ? record.page_id : null;
  const chartId =
    typeof record.chart_id === "string" ? record.chart_id : null;
  const source =
    typeof record.source === "string" ? record.source : null;
  const pageNumber = asNumber(record.page_number);
  if (
    !figureId ||
    !paperId ||
    !documentId ||
    !pageId ||
    !chartId ||
    !source ||
    pageNumber == null
  ) {
    return null;
  }
  return {
    figure_id: figureId,
    paper_id: paperId,
    document_id: documentId,
    page_id: pageId,
    page_number: pageNumber,
    chart_id: chartId,
    title: typeof record.title === "string" ? record.title : null,
    caption: typeof record.caption === "string" ? record.caption : null,
    source,
    bbox: (asRecord(record.bbox) as ResearchPaperFigurePreview["bbox"]) ?? null,
    image_path: typeof record.image_path === "string" ? record.image_path : null,
    preview_data_url:
      typeof record.preview_data_url === "string"
        ? record.preview_data_url
        : null,
    metadata: asRecord(record.metadata) ?? {},
  };
}

export function qaMetadataFromPayload(
  payload: Record<string, unknown>
): JsonObject | null {
  const askResult = asRecord(payload.ask_result);
  const qa = asRecord(askResult?.qa);
  return asRecord(qa?.metadata);
}

export function parseResearchVisualAnchor(
  value: unknown
): ResearchVisualAnchor | null {
  const record = asRecord(value);
  if (!record) return null;
  const imagePath = asNonEmptyString(record.image_path);
  const pageId = asNonEmptyString(record.page_id);
  const chartId = asNonEmptyString(record.chart_id);
  const figureId = asNonEmptyString(record.figure_id);
  const pageNumber = asNumber(record.page_number);
  const anchorRationale = asNonEmptyString(record.anchor_rationale);
  if (!imagePath && !pageId && !chartId && pageNumber == null && !figureId) {
    return null;
  }
  return {
    image_path: imagePath,
    page_id: pageId,
    page_number: pageNumber,
    chart_id: chartId,
    figure_id: figureId,
    anchor_rationale: anchorRationale,
  };
}

export function formatResearchQARouteLabel(route: string): string {
  if (route === "collection_qa") return "collection_qa";
  if (route === "document_drilldown") return "document_drilldown";
  if (route === "chart_drilldown") return "chart_drilldown";
  return route;
}

export function formatResearchQARuntimeLabel(runtime: string): string {
  if (runtime === "fused_chart") return "fused_chart";
  if (runtime === "document") return "document";
  return runtime;
}

export function buildResearchQATraceSummary(
  metadataSource: unknown
): ResearchQATraceSummary | null {
  const metadata = asRecord(metadataSource);
  if (!metadata) return null;
  const visualAnchor =
    asRecord(metadata.visual_anchor) ?? asRecord(metadata.anchor);
  const parsedAnchor = parseResearchVisualAnchor(visualAnchor);
  const anchor: Record<string, string | number> = {};
  for (const [key, value] of Object.entries(visualAnchor ?? {})) {
    if (typeof value === "string" && value.trim()) {
      anchor[key] = value.trim();
    } else if (typeof value === "number" && Number.isFinite(value)) {
      anchor[key] = value;
    }
  }
  const route = asNonEmptyString(metadata.qa_route);
  const confidence = asNumber(metadata.qa_route_confidence);
  const rationale = asNonEmptyString(metadata.qa_route_rationale);
  const runtime = asNonEmptyString(metadata.drilldown_runtime);
  const anchorRationale = parsedAnchor?.anchor_rationale ?? null;
  if (
    !route &&
    confidence == null &&
    !rationale &&
    !runtime &&
    !anchorRationale &&
    !Object.keys(anchor).length
  ) {
    return null;
  }
  return { route, confidence, rationale, runtime, anchorRationale, anchor };
}

export function parseResearchQAPaperScope(
  metadataSource: unknown
): ResearchQAPaperScope {
  const metadata = asRecord(metadataSource);
  const paperScope = asRecord(metadata?.paper_scope);
  const selectedPaperIds = asStringArray(metadata?.selected_paper_ids);
  return {
    paper_ids:
      selectedPaperIds.length > 0
        ? selectedPaperIds
        : asStringArray(paperScope?.paper_ids),
    scope_mode:
      asNonEmptyString(metadata?.qa_scope_mode) ??
      asNonEmptyString(paperScope?.scope_mode) ??
      "all_imported",
  };
}

export function parseResearchQADocumentIds(metadataSource: unknown): string[] {
  const metadata = asRecord(metadataSource);
  const scopeMetadata = asRecord(metadata?.scope_metadata);
  const selectedDocumentIds = asStringArray(metadata?.selected_document_ids);
  return selectedDocumentIds.length > 0
    ? selectedDocumentIds
    : asStringArray(scopeMetadata?.matched_document_ids);
}

export function selectionWarningsFromResearchQAMetadata(
  metadataSource: unknown
): string[] {
  const metadata = asRecord(metadataSource);
  return asStringArray(metadata?.selection_warnings);
}

export function visualAnchorFigureFromResearchQAMetadata(
  metadataSource: unknown
): ResearchPaperFigurePreview | null {
  const metadata = asRecord(metadataSource);
  return parseResearchPaperFigurePreview(metadata?.visual_anchor_figure);
}

export function importResultFromPayload(
  payload: Record<string, unknown>
): ImportPapersResponse | null {
  const importResult = asRecord(payload.import_result);
  if (!importResult || !Array.isArray(importResult.results)) return null;
  const importedCount = asNumber(importResult.imported_count);
  const skippedCount = asNumber(importResult.skipped_count);
  const failedCount = asNumber(importResult.failed_count);
  if (importedCount == null || skippedCount == null || failedCount == null) {
    return null;
  }
  const results: ImportPapersResponse["results"] = [];
  for (const item of importResult.results) {
    const record = asRecord(item);
    const paperId =
      record && typeof record.paper_id === "string" ? record.paper_id : null;
    const title =
      record && typeof record.title === "string" ? record.title : null;
    const status =
      record && typeof record.status === "string" ? record.status : null;
    if (!record || !paperId || !title || !status) continue;
    results.push({
      paper_id: paperId,
      title,
      status,
      document_id:
        typeof record.document_id === "string" ? record.document_id : null,
      storage_uri:
        typeof record.storage_uri === "string" ? record.storage_uri : null,
      parsed: typeof record.parsed === "boolean" ? record.parsed : false,
      indexed: typeof record.indexed === "boolean" ? record.indexed : false,
      error_message:
        typeof record.error_message === "string" ? record.error_message : null,
      metadata: asRecord(record.metadata) ?? {},
    });
  }
  return {
    imported_count: importedCount,
    skipped_count: skippedCount,
    failed_count: failedCount,
    results,
  };
}

export function traceFromPayload(
  payload: Record<string, unknown>
): ResearchAgentTraceStep[] {
  if (!Array.isArray(payload.trace)) return [];
  const trace: ResearchAgentTraceStep[] = [];
  payload.trace.forEach((item, index) => {
    const record = asRecord(item);
    if (!record) return;
    trace.push({
      step_index: asNumber(record.step_index) ?? index + 1,
      agent: typeof record.agent === "string" ? record.agent : "agent",
      thought: typeof record.thought === "string" ? record.thought : "",
      action_name:
        typeof record.action_name === "string"
          ? record.action_name
          : "unknown",
      phase: typeof record.phase === "string" ? record.phase : "act",
      action_input: asRecord(record.action_input) ?? {},
      status: typeof record.status === "string" ? record.status : "planned",
      observation:
        typeof record.observation === "string" ? record.observation : "",
      rationale: typeof record.rationale === "string" ? record.rationale : "",
      estimated_gain: asNumber(record.estimated_gain),
      estimated_cost: asNumber(record.estimated_cost),
      stop_signal:
        typeof record.stop_signal === "boolean" ? record.stop_signal : false,
      workspace_summary:
        typeof record.workspace_summary === "string"
          ? record.workspace_summary
          : null,
      metadata: asRecord(record.metadata) ?? {},
    });
  });
  return trace;
}

export function workspaceFromPayload(
  payload: Record<string, unknown>
): Pick<
  ResearchWorkspaceState,
  "status_summary" | "key_findings" | "evidence_gaps" | "next_actions"
> | null {
  const workspace = asRecord(payload.workspace);
  if (!workspace) return null;
  return {
    status_summary:
      typeof workspace.status_summary === "string"
        ? workspace.status_summary
        : "",
    key_findings: asStringArray(workspace.key_findings),
    evidence_gaps: asStringArray(workspace.evidence_gaps),
    next_actions: asStringArray(workspace.next_actions),
  };
}

export function comparisonFromWorkspaceMetadata(
  metadataSource: unknown
): ComparePapersResult | null {
  const metadata = asRecord(metadataSource);
  const payload = asRecord(metadata?.latest_comparison);
  if (!payload) return null;
  return comparisonFromPayload({ comparison: payload });
}

export function recommendationFromWorkspaceMetadata(
  metadataSource: unknown
): RecommendPapersResult | null {
  const metadata = asRecord(metadataSource);
  const payload = asRecord(metadata?.latest_recommendations);
  if (!payload) return null;
  return recommendationFromPayload({ recommendations: payload });
}

export function paperAnalysisFromWorkspaceMetadata(
  metadataSource: unknown
): PaperAnalysisResult | null {
  const metadata = asRecord(metadataSource);
  const payload = asRecord(metadata?.latest_paper_analysis);
  if (!payload) return null;
  return paperAnalysisFromPayload({ paper_analysis: payload });
}

export function contextCompressionFromWorkspaceMetadata(
  metadataSource: unknown
): ContextCompressionSummary | null {
  const metadata = asRecord(metadataSource);
  const payload = asRecord(metadata?.context_compression);
  if (!payload) return null;
  return contextCompressionFromPayload({ context_compression: payload });
}

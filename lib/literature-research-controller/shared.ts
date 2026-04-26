import type {
  PaperCandidate,
  ResearchReport,
  ResearchSource,
  ResearchTaskResponse,
} from "@/lib/types";
import { asRecord } from "@/lib/value-coercion";

export type ComposerMode = "research" | "qa";

export const RESEARCH_TASK_STORAGE_KEY = "research-task-id";
export const RESEARCH_CONVERSATION_STORAGE_KEY = "research-conversation-id";
export const IMPORT_JOB_STORAGE_KEY = "research-import-job-id";
export const IMPORT_JOB_POLL_INTERVAL_MS = 2_000;
export const DEFAULT_SOURCES: ResearchSource[] = [
  "arxiv",
  "openalex",
  "semantic_scholar",
  "zotero",
];
export const DEFAULT_TOPIC =
  "最近 6 个月无人机路径规划方向有哪些值得关注的论文？";
export const DEFAULT_ASK_QUESTION =
  "基于我勾选的论文，帮我分析它们的方法差异、优缺点或哪篇更值得先读。";

export function getDefaultSources(): ResearchSource[] {
  return [...DEFAULT_SOURCES];
}

export function uniqueTrimmedStrings(
  values: Array<string | null | undefined>
): string[] {
  const seen = new Set<string>();
  const results: string[] = [];
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    results.push(trimmed);
  }
  return results;
}

export function sanitizeSelectedPaperIds(
  paperIds: string[],
  papers: PaperCandidate[]
): string[] {
  const availablePaperIds = new Set(papers.map((paper) => paper.paper_id));
  return paperIds.filter((paperId) => availablePaperIds.has(paperId));
}

export function getPaperDocumentId(paper: PaperCandidate): string | null {
  const metadata = asRecord(paper.metadata);
  if (!metadata) return null;
  const documentId = metadata.document_id;
  return typeof documentId === "string" && documentId.trim()
    ? documentId
    : null;
}

export function isImportedPaper(paper: PaperCandidate): boolean {
  return paper.ingest_status === "ingested" || Boolean(getPaperDocumentId(paper));
}

export function defaultImportedPaperIds(papers: PaperCandidate[]): string[] {
  return papers.filter(isImportedPaper).map((paper) => paper.paper_id);
}

export function sanitizeImportedPaperIds(
  paperIds: string[],
  papers: PaperCandidate[]
): string[] {
  const importedPaperIds = new Set(defaultImportedPaperIds(papers));
  return paperIds.filter((paperId) => importedPaperIds.has(paperId));
}

export function nextImportedPaperIds(
  current: string[],
  papers: PaperCandidate[]
): string[] {
  const next = sanitizeImportedPaperIds(current, papers);
  return next.length ? next : defaultImportedPaperIds(papers);
}

export function toTaskSnapshot(input: {
  task: ResearchTaskResponse["task"];
  papers: PaperCandidate[];
  report?: ResearchReport | null;
  warnings?: string[] | null;
}): ResearchTaskResponse {
  return {
    task: input.task,
    papers: input.papers,
    report: input.report ?? null,
    warnings: input.warnings ?? [],
  };
}

"use client";

import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  PaperSelectionCard,
  ThreadBubble,
} from "@/components/ResearchConversationMessage";
import { ResearchFigurePreviewCard } from "@/components/ResearchFigurePreviewCard";
import { ResearchQATraceCard } from "@/components/ResearchQATraceCard";
import { normalizeMathDelimiters } from "@/lib/markdown";
import type {
  ImportPapersResponse,
  PaperCandidate,
  ResearchPaperFigurePreview,
  ResearchTaskAskResponse,
} from "@/lib/types";
import { buildListKey } from "@/lib/value-coercion";

function uniqueTrimmedStrings(
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

type ResearchThreadArtifactsModel = {
  importResult: ImportPapersResponse | null;
  activePapers: PaperCandidate[];
  selectedPaperIds: string[];
  recommendedPaperIds: Set<string>;
  mustReadPaperIds: Set<string>;
  askResult: ResearchTaskAskResponse | null;
  askResultFigure: ResearchPaperFigurePreview | null;
  askResultRoute: string | null;
};

type ResearchThreadArtifactsActions = {
  onTogglePaperSelection: (paperId: string) => void;
  onOpenAskResultFigure: (figure: ResearchPaperFigurePreview) => void;
};

export function ResearchThreadArtifacts({
  model,
  actions,
}: {
  model: ResearchThreadArtifactsModel;
  actions: ResearchThreadArtifactsActions;
}) {
  const {
    importResult,
    activePapers,
    selectedPaperIds,
    recommendedPaperIds,
    mustReadPaperIds,
    askResult,
    askResultFigure,
    askResultRoute,
  } = model;
  const { onTogglePaperSelection, onOpenAskResultFigure } = actions;
  const askWarnings = askResult
    ? uniqueTrimmedStrings(askResult.warnings)
    : [];

  return (
    <>
      {importResult && (
        <ThreadBubble
          role="assistant"
          title="导入结果"
          meta="候选论文已进入文档链路"
        >
          <div className="flex flex-wrap gap-3">
            <span className="badge-success">
              imported: {importResult.imported_count}
            </span>
            <span className="badge-warning">
              skipped: {importResult.skipped_count}
            </span>
            {importResult.failed_count > 0 && (
              <span className="badge-danger">
                failed: {importResult.failed_count}
              </span>
            )}
          </div>
          <ul className="mt-3">
            {importResult.results.slice(0, 5).map((result) => (
              <li key={result.paper_id}>
                {result.title} · {result.status}
                {result.document_id ? ` · doc=${result.document_id}` : ""}
                {result.error_message ? ` · ${result.error_message}` : ""}
              </li>
            ))}
          </ul>
        </ThreadBubble>
      )}

      {activePapers.length > 0 && (
        <ThreadBubble
          role="assistant"
          title="候选论文池"
          meta={`当前共 ${activePapers.length} 篇，可勾选后导入`}
        >
          <div className="grid gap-4">
            {activePapers.map((paper) => (
              <PaperSelectionCard
                key={paper.paper_id}
                paper={paper}
                selected={selectedPaperIds.includes(paper.paper_id)}
                recommended={recommendedPaperIds.has(paper.paper_id)}
                mustRead={mustReadPaperIds.has(paper.paper_id)}
                onToggle={onTogglePaperSelection}
              />
            ))}
          </div>
        </ThreadBubble>
      )}

      {askResult && (
        <>
          <ThreadBubble role="user" title="研究集合提问">
            <p>{askResult.qa.question}</p>
          </ThreadBubble>
          <ThreadBubble role="assistant" title="研究集合回答">
            {askResultRoute === "chart_drilldown" && askResultFigure && (
              <div className="mb-4">
                <ResearchFigurePreviewCard
                  figure={askResultFigure}
                  title="本次回答关联图片"
                  onOpen={onOpenAskResultFigure}
                />
              </div>
            )}
            <div className="chart-markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {normalizeMathDelimiters(askResult.qa.answer)}
              </ReactMarkdown>
            </div>
            <ResearchQATraceCard
              metadataSource={askResult.qa.metadata}
              className="mt-4"
            />
            {askWarnings.length > 0 && (
              <div className="mt-3 rounded-lg bg-amber-50 px-3 py-2.5 text-[12px] leading-6 text-amber-700">
                {askWarnings.map((warning, index) => (
                  <div
                    key={buildListKey(
                      "ask-result-warning",
                      warning,
                      index
                    )}
                  >
                    - {warning}
                  </div>
                ))}
              </div>
            )}
          </ThreadBubble>
        </>
      )}
    </>
  );
}

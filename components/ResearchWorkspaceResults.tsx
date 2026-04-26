"use client";

import { ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { ThreadBubble } from "@/components/ResearchConversationMessage";
import { normalizeMathDelimiters } from "@/lib/markdown";
import type {
  ComparePapersResult,
  ContextCompressionSummary,
  PaperAnalysisResult,
  RecommendPapersResult,
  ResearchReport,
} from "@/lib/types";
import { buildListKey } from "@/lib/value-coercion";

type ResearchWorkspaceResultsModel = {
  currentTopic: string;
  topicMeta: string;
  activeReport: ResearchReport | null;
  contextCompression: ContextCompressionSummary | null;
  paperAnalysisResult: PaperAnalysisResult | null;
  comparisonResult: ComparePapersResult | null;
  recommendationResult: RecommendPapersResult | null;
  paperTitleById: Map<string, string>;
  selectedPaperIds: string[];
};

type ResearchWorkspaceResultsActions = {
  onTogglePaperSelection: (paperId: string) => void;
};

export function ResearchWorkspaceResults({
  model,
  actions,
}: {
  model: ResearchWorkspaceResultsModel;
  actions: ResearchWorkspaceResultsActions;
}) {
  const {
    currentTopic,
    topicMeta,
    activeReport,
    contextCompression,
    paperAnalysisResult,
    comparisonResult,
    recommendationResult,
    paperTitleById,
    selectedPaperIds,
  } = model;
  const { onTogglePaperSelection } = actions;
  if (
    !activeReport &&
    !contextCompression &&
    !paperAnalysisResult &&
    !comparisonResult &&
    !recommendationResult
  ) {
    return null;
  }

  return (
    <>
      {activeReport && (
        <>
          <ThreadBubble
            role="user"
            title="当前研究主题"
            meta={topicMeta}
          >
            <p>{currentTopic}</p>
          </ThreadBubble>
          <ThreadBubble
            role="assistant"
            title="文献综述结果"
            meta={`候选论文 ${activeReport.paper_count} 篇`}
          >
            <div className="max-h-[520px] overflow-y-auto pr-1 chart-chat-scroll">
              <div className="chart-markdown">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {normalizeMathDelimiters(activeReport.markdown)}
                </ReactMarkdown>
              </div>
            </div>
          </ThreadBubble>
        </>
      )}

      {contextCompression && (
        <ThreadBubble
          role="assistant"
          title="上下文压缩摘要"
          meta={`papers=${contextCompression.paper_count} · summaries=${contextCompression.summary_count}`}
        >
          <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
            当前研究上下文已经压缩为更短的多层摘要视图，后续对比、推荐和 QA 会优先复用这批摘要。
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {contextCompression.levels.map((level, index) => (
              <span
                key={buildListKey(
                  "workspace-compression-level",
                  level,
                  index
                )}
                className="badge-muted"
              >
                {level}
              </span>
            ))}
          </div>
        </ThreadBubble>
      )}

      {paperAnalysisResult && (
        <ThreadBubble
          role="assistant"
          title="论文分析结果"
          meta={`focus=${paperAnalysisResult.focus}`}
        >
          {paperAnalysisResult.answer && (
            <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
              {paperAnalysisResult.answer}
            </div>
          )}
          {paperAnalysisResult.key_points.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {paperAnalysisResult.key_points.map((point, index) => (
                <span
                  key={buildListKey(
                    "workspace-paper-analysis-point",
                    point,
                    index
                  )}
                  className="badge-muted"
                >
                  {point}
                </span>
              ))}
            </div>
          )}
        </ThreadBubble>
      )}

      {!paperAnalysisResult && comparisonResult && (
        <ThreadBubble
          role="assistant"
          title="多论文对比结果"
          meta={`维度 ${comparisonResult.table.length} · 论文 ${comparisonResult.table[0] ? Object.keys(comparisonResult.table[0].values).length : 0}`}
        >
          {comparisonResult.summary && (
            <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
              {comparisonResult.summary}
            </div>
          )}
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full overflow-hidden rounded-lg border border-gray-200 bg-white text-left text-[12px]">
              <thead className="bg-gray-50 text-gray-500">
                <tr>
                  <th className="px-3 py-2 font-semibold">维度</th>
                  {comparisonResult.table[0] &&
                    Object.keys(comparisonResult.table[0].values).map((paperId) => (
                      <th key={paperId} className="px-3 py-2 font-semibold">
                        {paperTitleById.get(paperId) ?? paperId}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {comparisonResult.table.map((row) => (
                  <tr key={row.dimension} className="border-t border-gray-100">
                    <td className="px-3 py-2 font-semibold text-gray-800">
                      {row.dimension}
                    </td>
                    {Object.entries(row.values).map(([paperId, value]) => (
                      <td
                        key={`${row.dimension}:${paperId}`}
                        className="px-3 py-2 align-top text-gray-600"
                      >
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ThreadBubble>
      )}

      {!paperAnalysisResult && recommendationResult && (
        <ThreadBubble
          role="assistant"
          title="优先阅读推荐"
          meta={`top_k=${recommendationResult.recommendations.length}`}
        >
          <div className="grid gap-3">
            {recommendationResult.recommendations.map((item, index) => (
              <div
                key={item.paper_id}
                className="rounded-xl border border-gray-200 bg-white px-4 py-3"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-gray-800 px-1.5 text-[10px] font-bold text-white">
                    {index + 1}
                  </span>
                  <div className="text-[14px] font-semibold text-gray-900">
                    {item.title}
                  </div>
                  {item.year != null && (
                    <span className="badge-muted">{item.year}</span>
                  )}
                  {item.source && (
                    <span className="badge-info">{item.source}</span>
                  )}
                </div>
                <div className="mt-1.5 text-[13px] leading-6 text-gray-600">
                  {item.reason}
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => onTogglePaperSelection(item.paper_id)}
                    className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:border-blue-300 hover:text-blue-600"
                  >
                    {selectedPaperIds.includes(item.paper_id) ? "取消勾选" : "加入勾选"}
                  </button>
                  {item.url && (
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] font-medium text-blue-600 hover:underline"
                    >
                      查看论文 <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ThreadBubble>
      )}
    </>
  );
}

"use client";

import {
  AlertTriangle,
  Bot,
  CheckCheck,
  ChevronDown,
  ExternalLink,
  Hash,
} from "lucide-react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { normalizeMathDelimiters } from "@/lib/markdown";
import {
  comparisonFromPayload,
  contextCompressionFromPayload,
  importResultFromPayload,
  paperAnalysisFromPayload,
  papersFromPayload,
  qaMetadataFromPayload,
  recommendationFromPayload,
  traceFromPayload,
  workspaceFromPayload,
} from "@/lib/research-payloads";
import type { PaperCandidate, ResearchMessage } from "@/lib/types";
import { buildListKey, asRecord } from "@/lib/value-coercion";
import { ResearchQATraceCard } from "./ResearchQATraceCard";

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

function normalizeMessageRole(
  role: string
): "assistant" | "user" | "system" {
  return role === "user" || role === "system" ? role : "assistant";
}

export function ThreadBubble({
  role,
  title,
  meta,
  children,
}: {
  role: "assistant" | "user" | "system";
  title: string;
  meta?: string;
  children: ReactNode;
}) {
  const isUser = role === "user";
  const isSystem = role === "system";

  if (isUser) {
    return (
      <div className="flex justify-end chat-pop">
        <div className="max-w-[85%] rounded-2xl bg-blue-600 px-4 py-3 text-white sm:max-w-[75%]">
          {title && (
            <div className="text-[12px] font-medium text-white/80">{title}</div>
          )}
          <div className="text-[14px] leading-7">{children}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 chat-pop">
      <div
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
          isSystem
            ? "bg-amber-100 text-amber-600"
            : "bg-gray-200 text-gray-600"
        }`}
      >
        {isSystem ? (
          <AlertTriangle className="h-3.5 w-3.5" />
        ) : (
          <Bot className="h-3.5 w-3.5" />
        )}
      </div>
      <div className="min-w-0 max-w-full flex-1">
        {title && (
          <div className={`text-[12px] font-semibold ${isSystem ? "text-amber-800" : "text-gray-500"}`}>
            {title}
          </div>
        )}
        {meta && (
          <div className="text-[11px] text-gray-400">{meta}</div>
        )}
        <div className="mt-1 chat-markdown text-[14px] leading-7 text-gray-800">
          {children}
        </div>
      </div>
    </div>
  );
}

export function PaperSelectionCard({
  paper,
  selected,
  recommended,
  mustRead,
  onToggle,
}: {
  paper: PaperCandidate;
  selected: boolean;
  recommended?: boolean;
  mustRead?: boolean;
  onToggle: (paperId: string) => void;
}) {
  return (
    <article
      className={`group rounded-xl border p-3 transition-colors ${
        selected
          ? "border-blue-200 bg-blue-50/50"
          : "border-gray-200 bg-white hover:border-gray-300"
      }`}
    >
      <div className="flex items-start gap-3">
        <div
          onClick={() => onToggle(paper.paper_id)}
          className={`mt-0.5 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded transition-colors ${
            selected
              ? "bg-blue-600 text-white"
              : "border border-gray-300 bg-white group-hover:border-blue-400"
          }`}
        >
          {selected && <CheckCheck className="h-3 w-3" />}
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-[14px] font-semibold leading-5 text-gray-900">
            {paper.title}
          </h3>
          <p className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] text-gray-500">
            <span>{paper.authors.slice(0, 4).join(", ") || "unknown"}</span>
            <span className="text-gray-300">·</span>
            <span>{paper.year ?? "n/a"}</span>
            <span className="text-gray-300">·</span>
            <span className="uppercase text-[10px] font-medium text-gray-400">{paper.source}</span>
            {paper.relevance_score != null && (
              <>
                <span className="text-gray-300">·</span>
                <span className="text-blue-600 font-medium">
                  <Hash className="inline h-3 w-3" />{paper.relevance_score}
                </span>
              </>
            )}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {recommended && <span className="badge-warning">推荐</span>}
            {mustRead && <span className="badge-success">必读</span>}
            {paper.pdf_url && <span className="badge-info">PDF</span>}
            {paper.ingest_status === "ingested" && <span className="badge-success">已入库</span>}
          </div>
          <p className="mt-2 text-[13px] leading-6 text-gray-600 line-clamp-3">
            {paper.summary || paper.abstract || "摘要缺失。"}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[12px]">
            {paper.url && (
              <a href={paper.url} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-600 hover:underline">
                来源 <ExternalLink className="h-3 w-3" />
              </a>
            )}
            {paper.pdf_url && (
              <a href={paper.pdf_url} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-600 hover:underline">
                PDF <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

export function ConversationMessageBubble({
  message,
  selectedPaperIds,
  recommendedPaperIds,
  mustReadPaperIds,
  paperTitleById,
  onTogglePaperSelection,
}: {
  message: ResearchMessage;
  selectedPaperIds: string[];
  recommendedPaperIds: Set<string>;
  mustReadPaperIds: Set<string>;
  paperTitleById: Map<string, string>;
  onTogglePaperSelection: (paperId: string) => void;
}) {
  const payload = asRecord(message.payload) ?? {};
  const papers = papersFromPayload(payload);
  const comparison = comparisonFromPayload(payload);
  const recommendations = recommendationFromPayload(payload);
  const paperAnalysis = paperAnalysisFromPayload(payload);
  const contextCompression = contextCompressionFromPayload(payload);
  const importResult = importResultFromPayload(payload);
  const trace = traceFromPayload(payload);
  const workspace = workspaceFromPayload(payload);
  const qaTraceMetadata = qaMetadataFromPayload(payload);
  const citations = uniqueTrimmedStrings(message.citations ?? []);

  const renderMarkdown = (content: string) => (
    <div className="chart-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {normalizeMathDelimiters(content)}
      </ReactMarkdown>
    </div>
  );

  const renderDefaultContent = () => {
    if (!message.content.trim()) {
      return <p>已记录该步骤。</p>;
    }
    return renderMarkdown(message.content);
  };

  return (
    <ThreadBubble
      role={normalizeMessageRole(message.role)}
      title={message.title}
      meta={message.meta ?? undefined}
    >
      {message.kind === "candidates" && papers.length > 0 ? (
        <div className="grid gap-4">
          {papers.map((paper) => (
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
      ) : message.kind === "import_result" && importResult ? (
        <>
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
        </>
      ) : message.kind === "report" ? (
        <div className="max-h-[520px] overflow-y-auto pr-1 chart-chat-scroll">
          {renderDefaultContent()}
        </div>
      ) : message.title === "上下文压缩摘要" && contextCompression ? (
        <>
          <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
            {message.content.trim()
              ? message.content
              : "当前研究上下文已经压缩为更短的多层摘要视图，后续对比、推荐和 QA 会优先复用这批摘要。"}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {contextCompression.levels.map((level, index) => (
              <span
                key={buildListKey(
                  `conversation-level:${message.message_id}`,
                  level,
                  index
                )}
                className="badge-muted"
              >
                {level}
              </span>
            ))}
          </div>
        </>
      ) : message.title === "论文分析结果" && paperAnalysis ? (
        <>
          {(paperAnalysis.answer || message.content.trim()) && (
            <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
              {paperAnalysis.answer || message.content}
            </div>
          )}
          {paperAnalysis.key_points.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {paperAnalysis.key_points.map((point, index) => (
                <span
                  key={buildListKey(
                    `analysis-point:${message.message_id}`,
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
        </>
      ) : message.title === "多论文对比结果" && comparison ? (
        <>
          {(comparison.summary || message.content.trim()) && (
            <div className="rounded-lg bg-gray-50 px-4 py-3 text-[13px] leading-6 text-gray-700">
              {comparison.summary || message.content}
            </div>
          )}
          {comparison.table.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full overflow-hidden rounded-lg border border-gray-200 bg-white text-left text-[12px]">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="px-3 py-2 font-semibold">维度</th>
                    {Object.keys(comparison.table[0]?.values ?? {}).map(
                      (paperId) => (
                        <th key={paperId} className="px-3 py-2 font-semibold">
                          {paperTitleById.get(paperId) ?? paperId}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {comparison.table.map((row) => (
                    <tr
                      key={row.dimension}
                      className="border-t border-gray-100"
                    >
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
          )}
        </>
      ) : ["优先阅读推荐", "长期兴趣论文推荐"].includes(message.title) && recommendations ? (
        <div className="grid gap-3">
          {recommendations.recommendations.map((item, index) => (
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
                {item.source && <span className="badge-info">{item.source}</span>}
              </div>
              <div className="mt-1.5 text-[13px] leading-6 text-gray-600">
                {item.reason}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => onTogglePaperSelection(item.paper_id)}
                  className="btn-ghost rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:border-blue-300 hover:text-blue-600"
                >
                  {selectedPaperIds.includes(item.paper_id)
                    ? "取消勾选"
                    : "加入勾选"}
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
      ) : message.title === "Agent 决策轨迹" && trace.length > 0 ? (
        <ol className="space-y-2">
          {trace.map((step) => {
            const isSuccess =
              step.status === "success" || step.status === "succeeded";
            const isError =
              step.status === "error" || step.status === "failed";
            return (
              <li
                key={`${message.message_id}:${step.step_index}:${step.action_name}`}
                className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2"
              >
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-0.5 text-[10px] font-bold uppercase text-blue-600">
                    {step.phase}
                  </span>
                  <strong className="text-[13px]">{step.action_name}</strong>
                  <span
                    className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-bold ${
                      isSuccess
                        ? "bg-emerald-50 text-emerald-700"
                        : isError
                          ? "bg-red-50 text-red-700"
                          : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {step.status}
                  </span>
                </div>
                <div className="mt-1 text-[12px] leading-5 text-gray-600">
                  {step.observation}
                </div>
                {step.workspace_summary && (
                  <div className="mt-1 text-[11px] text-gray-400">
                    {step.workspace_summary}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      ) : message.title === "Research Workspace" && workspace ? (
        <>
          {workspace.key_findings.length > 0 && (
            <div className="space-y-1.5 rounded-lg bg-emerald-50 px-4 py-3">
              {workspace.key_findings.slice(0, 4).map((item, index) => (
                <div
                  key={buildListKey(
                    `conversation-finding:${message.message_id}`,
                    item,
                    index
                  )}
                  className="flex items-start gap-2 text-[13px] leading-6 text-emerald-700"
                >
                  <CheckCheck className="mt-1 h-3.5 w-3.5 shrink-0 text-emerald-500" />
                  {item}
                </div>
              ))}
            </div>
          )}
          {workspace.evidence_gaps.length > 0 && (
            <div className="mt-3 space-y-1.5 rounded-lg bg-amber-50 px-4 py-3">
              {workspace.evidence_gaps.slice(0, 4).map((item, index) => (
                <div
                  key={buildListKey(
                    `conversation-gap:${message.message_id}`,
                    item,
                    index
                  )}
                  className="flex items-start gap-2 text-[13px] leading-6 text-amber-700"
                >
                  <AlertTriangle className="mt-1 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  {item}
                </div>
              ))}
            </div>
          )}
          {workspace.next_actions.length > 0 && (
            <div className="mt-3 space-y-1.5 rounded-lg bg-gray-50 px-4 py-3">
              {workspace.next_actions.slice(0, 4).map((item, index) => (
                <div
                  key={buildListKey(
                    `conversation-next:${message.message_id}`,
                    item,
                    index
                  )}
                  className="flex items-start gap-2 text-[12px] leading-5 text-gray-600"
                >
                  <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-400" />
                  {item}
                </div>
              ))}
            </div>
          )}
          {!workspace.key_findings.length &&
            !workspace.evidence_gaps.length &&
            !workspace.next_actions.length &&
            renderDefaultContent()}
        </>
      ) : message.kind === "answer" && qaTraceMetadata ? (
        <>
          {renderDefaultContent()}
          <ResearchQATraceCard metadataSource={qaTraceMetadata} className="mt-4" />
        </>
      ) : (
        renderDefaultContent()
      )}

      {citations.length > 0 && (
        <div className="mt-3 rounded-lg bg-gray-50 px-4 py-3 text-[12px] leading-6 text-gray-600">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
            Evidence Citations
          </div>
          <div className="mt-2 space-y-1">
            {citations.map((citation, index) => (
              <div
                key={buildListKey(
                  `conversation-citation:${message.message_id}`,
                  citation,
                  index
                )}
              >
                - {citation}
              </div>
            ))}
          </div>
        </div>
      )}
    </ThreadBubble>
  );
}

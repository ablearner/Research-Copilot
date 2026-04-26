"use client";

import {
  CheckCheck,
  ExternalLink,
  ImageIcon,
  MessageSquareText,
} from "lucide-react";
import type { PaperCandidate, ResearchTodoItem } from "@/lib/types";

export function ImportedPaperScopeCard({
  paper,
  selected,
  onToggle,
  onOpenFigures,
  figuresActive,
  documentId,
}: {
  paper: PaperCandidate;
  selected: boolean;
  onToggle: (paperId: string) => void;
  onOpenFigures: (paper: PaperCandidate) => void;
  figuresActive?: boolean;
  documentId?: string | null;
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
        <button
          type="button"
          onClick={() => onToggle(paper.paper_id)}
          className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded transition-colors ${
            selected
              ? "bg-blue-600 text-white"
              : "border border-gray-300 bg-white group-hover:border-blue-400"
          }`}
          aria-label={`${selected ? "取消勾选" : "勾选"} ${paper.title}`}
        >
          {selected && <CheckCheck className="h-3 w-3" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-3">
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
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="badge-success">已入库</span>
              {documentId && (
                <span className="badge-muted max-w-[180px] truncate">
                  doc={documentId}
                </span>
              )}
            </div>
          </div>
          <div className="mt-2 text-[13px] leading-6 text-gray-600 line-clamp-3">
            {paper.summary || paper.abstract || "摘要缺失。"}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-4 text-[12px]">
            <button
              type="button"
              onClick={() => onOpenFigures(paper)}
              className={`inline-flex items-center gap-1 font-medium transition-colors ${
                figuresActive ? "text-blue-700" : "text-blue-600 hover:underline"
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" />
              {figuresActive ? "查看当前图表" : "读取图表"}
            </button>
            {paper.url && (
              <a
                href={paper.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-600 hover:underline"
              >
                查看来源 <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            {paper.pdf_url && (
              <a
                href={paper.pdf_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-600 hover:underline"
              >
                打开 PDF <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

export function TodoWorkbenchCard({
  item,
  onToggleDone,
  onToggleDismissed,
  onSearch,
  onImport,
  isBusy,
}: {
  item: ResearchTodoItem;
  onToggleDone: () => void;
  onToggleDismissed: () => void;
  onSearch: () => void;
  onImport: () => void;
  isBusy: (action: string) => boolean;
}) {
  const statusStyle =
    item.status === "dismissed"
      ? "border-gray-200 bg-gray-50 opacity-60"
      : item.status === "done"
        ? "border-emerald-200 bg-emerald-50/50"
        : "border-gray-200 bg-white hover:border-gray-300";

  const priorityColor =
    item.priority === "high"
      ? "text-red-600 bg-red-50"
      : item.priority === "medium"
        ? "text-amber-600 bg-amber-50"
        : "text-gray-600 bg-gray-100";

  const statusBadge =
    item.status === "done"
      ? "text-emerald-700 bg-emerald-50"
      : item.status === "dismissed"
        ? "text-gray-500 bg-gray-100"
        : "text-blue-600 bg-blue-50";

  return (
    <article
      className={`rounded-xl border px-3 py-3 transition-colors ${statusStyle}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.14em] ${priorityColor}`}
        >
          {item.priority}
        </span>
        <span className="text-gray-300">·</span>
        <span className="text-[11px] text-gray-500">
          {item.source}
        </span>
        <span className="text-gray-300">·</span>
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.14em] ${statusBadge}`}
        >
          {item.status}
        </span>
      </div>
      <div className="mt-2 text-[14px] font-medium leading-6 text-gray-900">
        {item.content}
      </div>
      {item.rationale && (
        <div className="mt-1.5 text-[12px] leading-5 text-gray-600">
          {item.rationale}
        </div>
      )}
      {item.question && (
        <div className="mt-1.5 flex items-start gap-1.5 text-[12px] leading-5 text-gray-500">
          <MessageSquareText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-400" />
          {item.question}
        </div>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onToggleDone}
          disabled={isBusy("done") || isBusy("open")}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:border-emerald-300 hover:text-emerald-600"
        >
          {item.status === "done" ? "恢复打开" : "标记完成"}
        </button>
        <button
          type="button"
          onClick={onToggleDismissed}
          disabled={isBusy("dismissed") || isBusy("open")}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:bg-gray-50"
        >
          {item.status === "dismissed" ? "恢复" : "关闭"}
        </button>
        <button
          type="button"
          onClick={onSearch}
          disabled={isBusy("search") || item.status === "dismissed"}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:border-blue-300 hover:text-blue-600"
        >
          {isBusy("search") ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-blue-300 border-t-blue-600" />
              检索中...
            </span>
          ) : (
            "重新检索"
          )}
        </button>
        <button
          type="button"
          onClick={onImport}
          disabled={isBusy("import") || item.status === "dismissed"}
          className="btn-accent rounded-lg px-3 py-1 text-[12px]"
        >
          {isBusy("import") ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
              导入中...
            </span>
          ) : (
            "补充导入"
          )}
        </button>
      </div>
    </article>
  );
}

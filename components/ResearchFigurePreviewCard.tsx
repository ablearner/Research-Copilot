"use client";

import { ImageIcon } from "lucide-react";
import type { ResearchPaperFigurePreview } from "@/lib/types";
import {
  buildResearchFigureDownloadName,
  buildResearchFigurePreviewHref,
} from "@/lib/research-figures";

export function ResearchFigurePreviewCard({
  figure,
  title,
  onOpen,
}: {
  figure: ResearchPaperFigurePreview;
  title?: string;
  onOpen: (figure: ResearchPaperFigurePreview) => void;
}) {
  const previewHref = buildResearchFigurePreviewHref(figure);

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={() => onOpen(figure)}
          className="h-20 w-20 shrink-0 overflow-hidden rounded-lg border border-gray-200 bg-white"
        >
          {previewHref ? (
            <img
              src={previewHref}
              alt={figure.title ?? figure.chart_id}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-gray-400">
              <ImageIcon className="h-5 w-5" />
            </div>
          )}
        </button>
        <div className="min-w-0 flex-1 text-[12px] leading-5 text-gray-700">
          <div className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
            {title ?? "图表预览"}
          </div>
          <div className="mt-1 font-semibold">
            {figure.title || figure.chart_id}
          </div>
          <div className="mt-1">
            page {figure.page_number} · {figure.source}
          </div>
          {figure.caption && (
            <div className="mt-1 line-clamp-3">
              {figure.caption}
            </div>
          )}
          <div className="mt-2 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => onOpen(figure)}
              className="rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-[12px] text-gray-600 hover:bg-white/80"
            >
              查看并缩放
            </button>
            {previewHref ? (
              <a
                href={previewHref}
                download={buildResearchFigureDownloadName(figure)}
                className="rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-[12px] text-gray-600 hover:bg-white/80"
              >
                下载图片
              </a>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

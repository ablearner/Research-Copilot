"use client";

import { ExternalLink, ImageIcon } from "lucide-react";
import type { ResearchPaperFigurePreview } from "@/lib/types";
import {
  buildResearchFigureDownloadName,
  buildResearchFigurePreviewHref,
} from "@/lib/research-figures";

export function ResearchActiveFigureAnchorCard({
  figure,
  onOpen,
  onClear,
}: {
  figure: ResearchPaperFigurePreview;
  onOpen: (figure: ResearchPaperFigurePreview) => void;
  onClear: () => void;
}) {
  const previewHref = buildResearchFigurePreviewHref(figure);
  const anchorRationale =
    typeof figure.metadata?.anchor_rationale === "string" &&
    figure.metadata.anchor_rationale.trim()
      ? figure.metadata.anchor_rationale.trim()
      : null;

  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50/50 px-3 py-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[12px] font-medium text-gray-800">
            <ImageIcon className="h-3.5 w-3.5" />
            当前图表锚点
          </div>
          <div className="mt-0.5 text-[12px] leading-5 text-gray-500">
            研究问答会优先按 `chart_drilldown` 路由分析这张图。
          </div>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-600 hover:bg-gray-50"
        >
          清除
        </button>
      </div>
      <div className="mt-3 flex gap-3">
        <button
          type="button"
          onClick={() => onOpen(figure)}
          className="h-16 w-16 shrink-0 overflow-hidden rounded-lg border border-gray-200 bg-white"
        >
          {previewHref ? (
            <img
              src={previewHref}
              alt={figure.title ?? figure.chart_id}
              className="h-full w-full object-cover"
            />
          ) : null}
        </button>
        <div className="min-w-0 flex-1 text-[12px] leading-5 text-gray-700">
          <div className="font-semibold">
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
            {previewHref ? (
              <>
                <button
                  type="button"
                  onClick={() => onOpen(figure)}
                  className="inline-flex items-center gap-1 rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-[12px] text-gray-600 hover:bg-gray-50"
                >
                  查看并缩放 <ExternalLink className="h-3.5 w-3.5" />
                </button>
                <a
                  href={previewHref}
                  download={buildResearchFigureDownloadName(figure)}
                  className="inline-flex items-center gap-1 rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-[12px] text-gray-600 hover:bg-gray-50"
                >
                  下载图片
                </a>
              </>
            ) : (
              <span className="rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-[12px] text-gray-500">
                当前图像未提供可访问预览地址
              </span>
            )}
          </div>
          {anchorRationale && (
            <div className="mt-2 rounded-lg bg-white px-3 py-2 text-[12px] leading-5 text-gray-600">
              为什么选这张图：{anchorRationale}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

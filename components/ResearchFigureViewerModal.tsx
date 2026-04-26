"use client";

import { useEffect, useState } from "react";
import type { ResearchPaperFigurePreview } from "@/lib/types";
import {
  buildResearchFigureDownloadName,
  buildResearchFigurePreviewHref,
} from "@/lib/research-figures";

export function ResearchFigureViewerModal({
  figure,
  onClose,
}: {
  figure: ResearchPaperFigurePreview | null;
  onClose: () => void;
}) {
  if (!figure) return null;

  return (
    <ResearchFigureViewerModalContent
      key={figure.figure_id}
      figure={figure}
      onClose={onClose}
    />
  );
}

function ResearchFigureViewerModalContent({
  figure,
  onClose,
}: {
  figure: ResearchPaperFigurePreview;
  onClose: () => void;
}) {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      } else if (event.key === "+" || event.key === "=") {
        setScale((current) => Math.min(current + 0.25, 3));
      } else if (event.key === "-") {
        setScale((current) => Math.max(current - 0.25, 0.5));
      } else if (event.key === "0") {
        setScale(1);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [figure, onClose]);

  const previewHref = buildResearchFigurePreviewHref(figure);
  if (!previewHref) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <button
        type="button"
        aria-label="关闭图片查看器"
        className="absolute inset-0"
        onClick={onClose}
      />
      <div className="relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-white/10 bg-gray-900">
        <div className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3 text-white">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold">
              {figure.title || figure.chart_id}
            </div>
            <div className="mt-0.5 text-xs text-gray-400">
              page {figure.page_number} · {figure.source}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setScale((current) => Math.max(current - 0.25, 0.5))}
              className="rounded-lg border border-white/15 px-2.5 py-1 text-xs text-white/80 hover:bg-white/10"
            >
              缩小
            </button>
            <button
              type="button"
              onClick={() => setScale(1)}
              className="rounded-lg border border-white/15 px-2.5 py-1 text-xs text-white/80 hover:bg-white/10"
            >
              重置
            </button>
            <button
              type="button"
              onClick={() => setScale((current) => Math.min(current + 0.25, 3))}
              className="rounded-lg border border-white/15 px-2.5 py-1 text-xs text-white/80 hover:bg-white/10"
            >
              放大
            </button>
            <a
              href={previewHref}
              download={buildResearchFigureDownloadName(figure)}
              className="rounded-lg border border-white/15 px-2.5 py-1 text-xs text-white/80 hover:bg-white/10"
            >
              下载
            </a>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-white/15 px-2.5 py-1 text-xs text-white/80 hover:bg-white/10"
            >
              关闭
            </button>
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto bg-gray-950 p-6">
          <img
            src={previewHref}
            alt={figure.title ?? figure.chart_id}
            className="mx-auto max-w-none origin-center rounded-lg transition-transform duration-150"
            style={{ transform: `scale(${scale})` }}
          />
        </div>
      </div>
    </div>
  );
}

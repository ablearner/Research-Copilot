"use client";

import { useRef, useEffect, useCallback } from "react";
import {
  ArrowUp,
  MessageSquareText,
  Search,
} from "lucide-react";
import { ResearchActiveFigureAnchorCard } from "@/components/ResearchActiveFigureAnchorCard";
import type {
  ResearchPaperFigurePreview,
  ResearchSource,
} from "@/lib/types";

type ResearchComposerModel = {
  mode: "research" | "qa";
  value: string;
  sources: ResearchSource[];
  selectedCount: number;
  selectedImportedCount: number;
  importedPapersCount: number;
  activeSelectedFigure: ResearchPaperFigurePreview | null;
  isResearchLoading: boolean;
  isAskLoading: boolean;
  canRunResearch: boolean;
  canAskCollection: boolean;
};

type ResearchComposerActions = {
  onModeChange: (mode: "research" | "qa") => void;
  onValueChange: (value: string) => void;
  onRunResearch: () => void;
  onAskCollection: () => void;
  onOpenActiveFigure: (figure: ResearchPaperFigurePreview) => void;
  onClearActiveFigure: () => void;
};

export function ResearchComposer({
  model,
  actions,
}: {
  model: ResearchComposerModel;
  actions: ResearchComposerActions;
}) {
  const {
    mode,
    value,
    activeSelectedFigure,
    isResearchLoading,
    isAskLoading,
    canRunResearch,
    canAskCollection,
  } = model;
  const {
    onModeChange,
    onValueChange,
    onRunResearch,
    onAskCollection,
    onOpenActiveFigure,
    onClearActiveFigure,
  } = actions;

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isLoading = isResearchLoading || isAskLoading;
  const canSend = mode === "research" ? canRunResearch : canAskCollection;

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useEffect(() => { autoResize(); }, [value, autoResize]);

  const handleSend = () => {
    if (mode === "research") onRunResearch();
    else onAskCollection();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend && !isLoading) handleSend();
    }
  };

  return (
    <div className="shrink-0 border-t border-gray-200 bg-white">
      <div className="mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
        {/* Figure anchor (if any) */}
        {mode === "qa" && activeSelectedFigure && (
          <div className="mb-2">
            <ResearchActiveFigureAnchorCard
              figure={activeSelectedFigure}
              onOpen={onOpenActiveFigure}
              onClear={onClearActiveFigure}
            />
          </div>
        )}

        {/* Input container */}
        <div className="rounded-2xl border border-gray-300 bg-white shadow-sm transition-colors focus-within:border-gray-400 focus-within:shadow-md">
          <div className="flex items-end gap-2 px-4 py-3">
            <textarea
              ref={textareaRef}
              rows={1}
              value={value}
              onChange={(e) => onValueChange(e.target.value)}
              onKeyDown={handleKeyDown}
              className="min-h-[24px] max-h-[200px] flex-1 resize-none border-0 bg-transparent text-[15px] leading-6 text-gray-900 placeholder-gray-400 outline-none"
              placeholder={
                mode === "research"
                  ? "输入研究主题，如：近半年无人机路径规划方向的重要论文..."
                  : "对已导入论文提问..."
              }
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend || isLoading}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-900 text-white transition-colors hover:bg-gray-700 disabled:bg-gray-300 disabled:text-gray-500"
            >
              {isLoading ? (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              ) : (
                <ArrowUp className="h-4 w-4" />
              )}
            </button>
          </div>

          {/* Bottom row: mode toggle + hint */}
          <div className="flex items-center justify-between border-t border-gray-100 px-4 py-2">
            <div className="inline-flex rounded-lg bg-gray-100 p-0.5">
              <button
                type="button"
                onClick={() => onModeChange("research")}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-[11px] font-medium transition-colors ${
                  mode === "research"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <Search className="h-3 w-3" />
                检索
              </button>
              <button
                type="button"
                onClick={() => onModeChange("qa")}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-[11px] font-medium transition-colors ${
                  mode === "qa"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <MessageSquareText className="h-3 w-3" />
                问答
              </button>
            </div>
            <span className="text-[11px] text-gray-400">
              Enter 发送 · Shift+Enter 换行
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

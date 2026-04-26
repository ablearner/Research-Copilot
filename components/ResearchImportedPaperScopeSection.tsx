"use client";

import { ImageIcon, Sparkles } from "lucide-react";
import type {
  AnalyzeResearchPaperFigureResponse,
  PaperCandidate,
  RequestState,
  ResearchPaperFigureListResponse,
  ResearchPaperFigurePreview,
} from "@/lib/types";
import { buildListKey } from "@/lib/value-coercion";
import { ImportedPaperScopeCard } from "@/components/ResearchWorkspaceCards";

export function ResearchImportedPaperScopeSection({
  importedPapers,
  selectedImportedPaperIds,
  onSelectAll,
  onClearSelection,
  onTogglePaper,
  onOpenFigures,
  getPaperDocumentId,
  figurePanelPaperId,
  paperFigureResult,
  figureState,
  figureError,
  selectedFigure,
  onAnalyzeFigure,
  onUseFigureAsAnchor,
  figureAnalysisState,
  figureAnalysisError,
  figureAnalysisResult,
}: {
  importedPapers: PaperCandidate[];
  selectedImportedPaperIds: string[];
  onSelectAll: () => void;
  onClearSelection: () => void;
  onTogglePaper: (paperId: string) => void;
  onOpenFigures: (paper: PaperCandidate) => void;
  getPaperDocumentId: (paper: PaperCandidate) => string | null;
  figurePanelPaperId: string | null;
  paperFigureResult: ResearchPaperFigureListResponse | null;
  figureState: RequestState;
  figureError: string | null;
  selectedFigure: ResearchPaperFigurePreview | null;
  onAnalyzeFigure: (figure: ResearchPaperFigurePreview) => void;
  onUseFigureAsAnchor: (figure: ResearchPaperFigurePreview) => void;
  figureAnalysisState: RequestState;
  figureAnalysisError: string | null;
  figureAnalysisResult: AnalyzeResearchPaperFigureResponse | null;
}) {
  if (!importedPapers.length) {
    return (
      <div className="mt-3 rounded-lg border border-dashed border-gray-200 px-3 py-4 text-center text-[12px] text-gray-400">
        还没有已导入论文。先在上方候选论文池里勾选并导入，再回来限定问答范围。
      </div>
    );
  }

  return (
    <>
      <div className="mt-4 flex gap-2">
        <button
          type="button"
          onClick={onSelectAll}
          className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-[12px] font-medium text-gray-700 hover:bg-gray-50"
        >
          全选
        </button>
        <button
          type="button"
          onClick={onClearSelection}
          className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-[12px] text-gray-500 hover:bg-gray-50"
        >
          清空
        </button>
      </div>
      <div className="mt-4 flex max-h-[36vh] flex-col gap-3 overflow-y-auto pr-1 chart-chat-scroll">
        {importedPapers.map((paper) => (
          <ImportedPaperScopeCard
            key={paper.paper_id}
            paper={paper}
            documentId={getPaperDocumentId(paper)}
            selected={selectedImportedPaperIds.includes(paper.paper_id)}
            onToggle={onTogglePaper}
            onOpenFigures={onOpenFigures}
            figuresActive={figurePanelPaperId === paper.paper_id}
          />
        ))}
      </div>
      {(figurePanelPaperId || paperFigureResult || figureError) && (
        <div className="mt-3 rounded-lg border border-gray-200 bg-white p-3">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-gray-800">图表预览</div>
            {selectedFigure && (
              <span className="badge-info">
                当前锚点: p.{selectedFigure.page_number}
              </span>
            )}
          </div>
          {figureState === "loading" && (
            <div className="mt-2 text-[12px] text-gray-500">
              正在读取论文图表候选...
            </div>
          )}
          {figureError && (
            <div className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] text-red-700">
              {figureError}
            </div>
          )}
          {paperFigureResult && paperFigureResult.figures.length > 0 && (
            <div className="mt-4 grid gap-3">
              {paperFigureResult.figures.map((figure) => (
                <div
                  key={figure.figure_id}
                  className={`rounded-lg border p-3 ${
                    selectedFigure?.figure_id === figure.figure_id
                      ? "border-blue-200 bg-blue-50/50"
                      : "border-gray-200 bg-white"
                  }`}
                >
                  <div className="flex gap-3">
                    <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg bg-gray-100">
                      {figure.preview_data_url ? (
                        <img
                          src={figure.preview_data_url}
                          alt={figure.title ?? figure.chart_id}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="flex h-full items-center justify-center text-gray-400">
                          <ImageIcon className="h-5 w-5" />
                        </div>
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] font-medium text-gray-900">
                        {figure.title || `图表 ${figure.chart_id}`}
                      </div>
                      <div className="mt-0.5 text-[12px] text-gray-500">
                        page {figure.page_number} · {figure.source}
                      </div>
                      {figure.caption && (
                        <div className="mt-1.5 text-[12px] leading-5 text-gray-600">
                          {figure.caption}
                        </div>
                      )}
                      <div className="mt-3 flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => onAnalyzeFigure(figure)}
                          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-700 hover:bg-gray-50"
                        >
                          分析图表
                        </button>
                        <button
                          type="button"
                          onClick={() => onUseFigureAsAnchor(figure)}
                          className="rounded-lg border border-gray-200 bg-white px-3 py-1 text-[12px] text-gray-500 hover:bg-gray-50"
                        >
                          用作问答锚点
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          {paperFigureResult &&
            paperFigureResult.figures.length === 0 &&
            figureState === "success" && (
              <div className="mt-2 text-[12px] text-gray-500">
                这篇论文暂时没有找到可直接分析的图表候选。
              </div>
            )}
          {figureAnalysisError && (
            <div className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] text-red-700">
              {figureAnalysisError}
            </div>
          )}
          {figureAnalysisState === "loading" && (
            <div className="mt-2 rounded-lg bg-blue-50 px-3 py-2 text-[12px] text-blue-700">
              正在分析这张图表，请稍等...
            </div>
          )}
          {figureAnalysisResult && (
            <div className="mt-3 rounded-lg border border-gray-200 bg-white px-3 py-3">
              <div className="flex items-center gap-2 text-[12px] font-semibold text-gray-700">
                <Sparkles className="h-3.5 w-3.5 text-gray-500" />
                图表分析结果
              </div>
              <div className="mt-1.5 text-[13px] leading-6 text-gray-700">
                {figureAnalysisResult.answer}
              </div>
              {figureAnalysisResult.key_points.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {figureAnalysisResult.key_points.map((point, index) => (
                    <span
                      key={buildListKey("figure-analysis-point", point, index)}
                      className="badge-muted"
                    >
                      {point}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}

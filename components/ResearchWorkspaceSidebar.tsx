"use client";

import {
  AlertTriangle,
  BookOpenText,
  ChevronDown,
  RefreshCcw,
} from "lucide-react";
import { ResearchImportedPaperScopeSection } from "@/components/ResearchImportedPaperScopeSection";
import { ResearchTodoWorkbenchSection } from "@/components/ResearchTodoWorkbenchSection";
import { SupervisorExecutionCard } from "@/components/SupervisorExecutionCard";
import type {
  AnalyzeResearchPaperFigureResponse,
  PaperCandidate,
  RequestState,
  ResearchAgentRuntimeMetadata,
  ResearchJob,
  ResearchPaperFigureListResponse,
  ResearchPaperFigurePreview,
  ResearchTask,
  ResearchTodoItem,
  ResearchWorkspaceState,
} from "@/lib/types";
import { buildListKey } from "@/lib/value-coercion";

function TaskSummaryCard({
  task,
  workspace,
  lastTaskId,
  onRefresh,
  loading,
}: {
  task: ResearchTask | null;
  workspace: ResearchWorkspaceState | null;
  lastTaskId: string | null;
  onRefresh: () => void;
  loading: boolean;
}) {
  const stageColor: Record<string, string> = {
    discover: "text-indigo-700 bg-indigo-50",
    ingest: "text-amber-700 bg-amber-50",
    qa: "text-emerald-700 bg-emerald-50",
    complete: "text-slate-700 bg-slate-100",
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="p-3">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[13px] font-semibold text-gray-800">
              {task?.topic ?? "还没有研究任务"}
            </div>
          </div>
          {lastTaskId && (
            <button
              type="button"
              onClick={onRefresh}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-2.5 py-1.5 text-[12px] text-gray-600 hover:bg-gray-50"
            >
              <RefreshCcw
                className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
              />
              刷新
            </button>
          )}
        </div>
        <div className="mt-3 grid gap-1.5 text-[12px]">
          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2">
            <span className="text-gray-500">状态</span>
            <span className="font-medium text-gray-900">{task?.status ?? "idle"}</span>
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2">
            <span className="text-gray-500">Workspace</span>
            <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium uppercase ${stageColor[workspace?.current_stage ?? "discover"] ?? "text-gray-700 bg-gray-100"}`}>
              {workspace?.current_stage ?? "discover"}
            </span>
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2">
            <span className="text-gray-500">已入库文档</span>
            <span className="font-medium text-gray-900">{task?.imported_document_ids.length ?? 0}</span>
          </div>
          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2">
            <span className="text-gray-500">自动待办</span>
            <span className="font-medium text-gray-900">{task?.todo_items.length ?? 0}</span>
          </div>
        </div>
        {workspace?.status_summary && (
          <div className="mt-2 rounded-lg bg-gray-50 px-3 py-2 text-[12px] leading-5 text-gray-600">
            {workspace.status_summary}
          </div>
        )}
        {workspace?.stop_reason && (
          <div className="mt-2 flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-[12px] leading-5 text-amber-700">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
            <span>stop reason: {workspace.stop_reason}</span>
          </div>
        )}
        {workspace?.next_actions.length ? (
          <div className="mt-2 space-y-1 rounded-lg bg-gray-50 px-3 py-2.5">
            {workspace.next_actions.slice(0, 3).map((item, index) => (
              <div
                key={buildListKey("sidebar-next-action", item, index)}
                className="flex items-start gap-2 text-[12px] leading-5 text-gray-600"
              >
                <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-400" />
                {item}
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

type ResearchWorkspaceSidebarModel = {
  task: ResearchTask | null;
  workspace: ResearchWorkspaceState | null;
  lastTaskId: string | null;
  loading: boolean;
  runtimeMetadata?: ResearchAgentRuntimeMetadata | null;
  selectedImportedCount: number;
  selectedCount: number;
  activePapersCount: number;
  importedPapers: PaperCandidate[];
  importJobRunning: boolean;
  activeImportJob: ResearchJob | null;
  importError: string | null;
  selectedImportedPaperIds: string[];
  getPaperDocumentId: (paper: PaperCandidate) => string | null;
  figurePanelPaperId: string | null;
  paperFigureResult: ResearchPaperFigureListResponse | null;
  figureState: RequestState;
  figureError: string | null;
  selectedFigure: ResearchPaperFigurePreview | null;
  figureAnalysisState: RequestState;
  figureAnalysisError: string | null;
  figureAnalysisResult: AnalyzeResearchPaperFigureResponse | null;
  todoItems: ResearchTodoItem[];
  todoActionError: string | null;
  todoActionNotice: string | null;
};

type ResearchWorkspaceSidebarActions = {
  onRefreshTask: () => void;
  onImportSelected: () => void;
  onSelectAllImported: () => void;
  onClearImportedSelection: () => void;
  onToggleImportedPaper: (paperId: string) => void;
  onOpenFigures: (paper: PaperCandidate) => void;
  onAnalyzeFigure: (figure: ResearchPaperFigurePreview) => void;
  onUseFigureAsAnchor: (figure: ResearchPaperFigurePreview) => void;
  onToggleTodoDone: (todoId: string, status: ResearchTodoItem["status"]) => void;
  onToggleTodoDismissed: (
    todoId: string,
    status: ResearchTodoItem["status"]
  ) => void;
  onSearchTodo: (todoId: string) => void;
  onImportTodo: (todoId: string) => void;
  isTodoBusy: (todoId: string, action: string) => boolean;
};

export function ResearchWorkspaceSidebar({
  model,
  actions,
}: {
  model: ResearchWorkspaceSidebarModel;
  actions: ResearchWorkspaceSidebarActions;
}) {
  const {
    task,
    workspace,
    lastTaskId,
    loading,
    runtimeMetadata,
    selectedImportedCount,
    selectedCount,
    activePapersCount,
    importedPapers,
    importJobRunning,
    activeImportJob,
    importError,
    selectedImportedPaperIds,
    getPaperDocumentId,
    figurePanelPaperId,
    paperFigureResult,
    figureState,
    figureError,
    selectedFigure,
    figureAnalysisState,
    figureAnalysisError,
    figureAnalysisResult,
    todoItems,
    todoActionError,
    todoActionNotice,
  } = model;
  const {
    onRefreshTask,
    onImportSelected,
    onSelectAllImported,
    onClearImportedSelection,
    onToggleImportedPaper,
    onOpenFigures,
    onAnalyzeFigure,
    onUseFigureAsAnchor,
    onToggleTodoDone,
    onToggleTodoDismissed,
    onSearchTodo,
    onImportTodo,
    isTodoBusy,
  } = actions;
  return (
    <>
      <TaskSummaryCard
        task={task}
        workspace={workspace}
        lastTaskId={lastTaskId}
        onRefresh={onRefreshTask}
        loading={loading}
      />

      <SupervisorExecutionCard metadata={runtimeMetadata} />

      <div className="rounded-lg border border-gray-200 bg-white">
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-gray-800">论文分析</div>
            <span className="text-[11px] text-gray-400">
              {selectedImportedCount > 0
                ? `已圈定 ${selectedImportedCount} 篇`
                : "先勾选已导入论文"}
            </span>
          </div>
          <div className="mt-2 rounded-lg bg-gray-50 px-3 py-2 text-[12px] leading-5 text-gray-500">
            勾选已导入论文后直接提问，系统自动做多论文分析、对比或推荐。
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white">
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-gray-800">
              已选 {selectedCount} 篇候选
            </div>
            <span className="text-[11px] text-gray-400">{activePapersCount} total</span>
          </div>
          <div className="mt-2 text-[12px] leading-5 text-gray-500">
            勾选候选论文后执行导入，导入后在下方勾选问答范围。
          </div>
          <button
            type="button"
            onClick={onImportSelected}
            disabled={importJobRunning || !selectedCount}
            className="btn-accent mt-4 flex w-full items-center justify-center gap-2"
          >
            {importJobRunning ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                后台导入中...
              </>
            ) : (
              <>
                <BookOpenText className="h-4 w-4" />
                导入选中论文（{selectedCount}）
              </>
            )}
          </button>
          {activeImportJob && (
            <div className="mt-2 space-y-1 rounded-lg bg-gray-50 px-3 py-2 text-[12px] leading-5 text-gray-600">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
                job: {activeImportJob.job_id}
              </div>
              <div>status: {activeImportJob.status}</div>
              {activeImportJob.progress_current != null &&
                activeImportJob.progress_total != null && (
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-blue-600 transition-all duration-500"
                        style={{
                          width: `${Math.round(
                            ((activeImportJob.progress_current ?? 0) /
                              (activeImportJob.progress_total ?? 1)) *
                              100
                          )}%`,
                        }}
                      />
                    </div>
                    <span className="text-[11px] font-semibold tabular-nums">
                      {activeImportJob.progress_current}/
                      {activeImportJob.progress_total}
                    </span>
                  </div>
                )}
              {activeImportJob.progress_message && (
                <div className="text-blue-600">
                  {activeImportJob.progress_message}
                </div>
              )}
            </div>
          )}
          {importError && (
            <div className="mt-2 flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] leading-5 text-red-700">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-400" />
              {importError}
            </div>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white">
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="text-[13px] font-semibold text-gray-800">
              已导入论文 {importedPapers.length} 篇
            </div>
            <span className="text-[11px] text-gray-400">scope {selectedImportedCount}</span>
          </div>
          <div className="mt-2 text-[12px] leading-5 text-gray-500">
            问答只针对勾选的已导入论文执行。
          </div>
          <ResearchImportedPaperScopeSection
            importedPapers={importedPapers}
            selectedImportedPaperIds={selectedImportedPaperIds}
            onSelectAll={onSelectAllImported}
            onClearSelection={onClearImportedSelection}
            onTogglePaper={onToggleImportedPaper}
            onOpenFigures={onOpenFigures}
            getPaperDocumentId={getPaperDocumentId}
            figurePanelPaperId={figurePanelPaperId}
            paperFigureResult={paperFigureResult}
            figureState={figureState}
            figureError={figureError}
            selectedFigure={selectedFigure}
            onAnalyzeFigure={onAnalyzeFigure}
            onUseFigureAsAnchor={onUseFigureAsAnchor}
            figureAnalysisState={figureAnalysisState}
            figureAnalysisError={figureAnalysisError}
            figureAnalysisResult={figureAnalysisResult}
          />
        </div>
      </div>

      <ResearchTodoWorkbenchSection
        todoItems={todoItems}
        todoActionError={todoActionError}
        todoActionNotice={todoActionNotice}
        onToggleDone={onToggleTodoDone}
        onToggleDismissed={onToggleTodoDismissed}
        onSearch={onSearchTodo}
        onImport={onImportTodo}
        isBusy={isTodoBusy}
      />
    </>
  );
}

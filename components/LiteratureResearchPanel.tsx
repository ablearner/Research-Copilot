"use client";

import {
  Bot,
  BookOpenText,
  CheckCheck,
  FileText,
  Layers,
  Library,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type {
  ResearchPaperFigurePreview,
  ResearchSource,
  ResearchTodoItem,
} from "@/lib/types";
import { ResearchComposer } from "@/components/ResearchComposer";
import { ThreadBubble } from "@/components/ResearchConversationMessage";
import { ResearchFigureViewerModal } from "@/components/ResearchFigureViewerModal";
import {
  ResearchConversationsSection,
  ResearchSetupSection,
} from "@/components/ResearchSidebarSections";
import { ResearchThreadArtifacts } from "@/components/ResearchThreadArtifacts";
import { ResearchThreadPreamble } from "@/components/ResearchThreadPreamble";
import { ResearchWorkspaceSidebar } from "@/components/ResearchWorkspaceSidebar";
import { ResearchWorkspaceResults } from "@/components/ResearchWorkspaceResults";
import { useLiteratureResearchController } from "@/lib/use-literature-research-controller";

const SOURCE_OPTIONS: Array<{
  value: ResearchSource;
  label: string;
  hint: string;
  icon: React.ReactNode;
}> = [
  {
    value: "arxiv",
    label: "arXiv",
    hint: "开放预印本与最新论文发现。",
    icon: <FileText className="h-4 w-4" />,
  },
  {
    value: "openalex",
    label: "OpenAlex",
    hint: "跨来源 works 元数据聚合。",
    icon: <Layers className="h-4 w-4" />,
  },
  {
    value: "semantic_scholar",
    label: "Semantic Scholar",
    hint: "学术图谱搜索，适合补 citation 与开放 PDF 元数据。",
    icon: <Search className="h-4 w-4" />,
  },
  {
    value: "ieee",
    label: "IEEE",
    hint: "IEEE Xplore 元数据检索，通常更适合工程与机器人方向。",
    icon: <Zap className="h-4 w-4" />,
  },
  {
    value: "zotero",
    label: "Zotero",
    hint: "本地文献库检索，适合先复用已有条目与 PDF 附件。",
    icon: <Library className="h-4 w-4" />,
  },
];

const RESEARCH_SUGGESTIONS = [
  "最近 6 个月无人机路径规划方向有哪些值得关注的论文？",
  "帮我整理多模态图表理解方向最近一年的代表性论文。",
  "低空复杂环境鲁棒导航最近有哪些可直接入库的开放论文？",
];

/* ═══════════════════ Main Panel ═══════════════════ */

export function LiteratureResearchPanel() {
  const sidebarScrollRef = useRef<HTMLDivElement>(null);
  const threadScrollRef = useRef<HTMLDivElement>(null);
  const threadEndRef = useRef<HTMLDivElement>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const {
    composerActions,
    composerState,
    conversationActions,
    conversationState,
    figureActions,
    figureState: figureViewState,
    importedPaperScopeActions,
    qaState,
    runtimeState,
    setupActions: controllerSetupActions,
    setupState,
    todoActions,
    todoState,
    workspaceActions,
    workspaceState,
  } = useLiteratureResearchController();

  useEffect(() => {
    const container = threadScrollRef.current;
    if (!container) return;
    const frameId = window.requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [
    runtimeState.agentResult,
    qaState.askResult,
    workspaceState.importResult,
    runtimeState.error,
    runtimeState.taskResult,
  ]);

  const shouldReplayConversation = false;
  const replayHasError = false;
  const replayHasWarning = false;

  const workspaceSidebarModel = {
    task: workspaceState.activeTask,
    workspace: workspaceState.activeWorkspace,
    lastTaskId: workspaceState.lastTaskId,
    loading: runtimeState.isLoading,
    runtimeMetadata: runtimeState.agentResult?.metadata,
    selectedImportedCount: workspaceState.selectedImportedCount,
    selectedCount: workspaceState.selectedCount,
    activePapersCount: workspaceState.activePapers.length,
    importedPapers: workspaceState.importedPapers,
    importJobRunning: workspaceState.importJobRunning,
    activeImportJob: workspaceState.activeImportJob,
    importError: workspaceState.importError,
    selectedImportedPaperIds: workspaceState.selectedImportedPaperIds,
    getPaperDocumentId: workspaceState.getPaperDocumentId,
    figurePanelPaperId: figureViewState.panelPaperId,
    paperFigureResult: figureViewState.previewResult,
    figureState: figureViewState.requestState,
    figureError: figureViewState.error,
    selectedFigure: figureViewState.selectedFigure,
    figureAnalysisState: figureViewState.analysisState,
    figureAnalysisError: figureViewState.analysisError,
    figureAnalysisResult: figureViewState.analysisResult,
    todoItems: workspaceState.activeTask?.todo_items ?? [],
    todoActionError: todoState.actionError,
    todoActionNotice: todoState.actionNotice,
  };

  const workspaceSidebarActions = {
    onRefreshTask: () => { void workspaceActions.refreshTask(); },
    onImportSelected: () => { void workspaceActions.importSelected(); },
    onSelectAllImported: importedPaperScopeActions.selectAll,
    onClearImportedSelection: importedPaperScopeActions.clear,
    onToggleImportedPaper: importedPaperScopeActions.toggle,
    onOpenFigures: (targetPaper: typeof workspaceState.activePapers[number]) => {
      void figureActions.loadPaperFigures(targetPaper);
    },
    onAnalyzeFigure: (figure: ResearchPaperFigurePreview) => {
      void figureActions.analyzePaperFigure(figure);
    },
    onUseFigureAsAnchor: figureActions.useAsAnchor,
    onToggleTodoDone: (todoId: string, status: ResearchTodoItem["status"]) => {
      void todoActions.toggleDone(todoId, status);
    },
    onToggleTodoDismissed: (todoId: string, status: ResearchTodoItem["status"]) => {
      void todoActions.toggleDismissed(todoId, status);
    },
    onSearchTodo: (todoId: string) => { void todoActions.search(todoId); },
    onImportTodo: (todoId: string) => { void todoActions.import(todoId); },
    isTodoBusy: todoActions.isBusy,
  };

  const workspaceResultsModel = {
    currentTopic:
      workspaceState.activeTask?.topic ??
      workspaceState.searchResult?.plan.topic ??
      composerState.topic,
    topicMeta: workspaceState.activeTask
      ? "Supervisor Graph 已建立研究任务"
      : "自主调研结果",
    activeReport: workspaceState.activeReport,
    contextCompression: workspaceState.contextCompression,
    paperAnalysisResult: workspaceState.paperAnalysisResult,
    comparisonResult: workspaceState.comparisonResult,
    recommendationResult: workspaceState.recommendationResult,
    paperTitleById: workspaceState.paperTitleById,
    selectedPaperIds: workspaceState.selectedPaperIds,
  };

  const threadArtifactsModel = {
    importResult: workspaceState.importResult,
    activePapers: workspaceState.activePapers,
    selectedPaperIds: workspaceState.selectedPaperIds,
    recommendedPaperIds: workspaceState.recommendedPaperIds,
    mustReadPaperIds: workspaceState.mustReadPaperIds,
    askResult: qaState.askResult,
    askResultFigure: figureViewState.askResultFigure,
    askResultRoute: qaState.askResultQATrace?.route ?? null,
  };

  const conversationsModel = {
    conversations: conversationState.conversations,
    conversationId: conversationState.conversationId,
    currentConversation: conversationState.currentConversation,
    conversationMessageCount: conversationState.conversationMessageCount,
  };

  const conversationsActions = {
    onCreateConversation: () => { void conversationActions.create(); },
    onDeleteConversation: () => {
      if (conversationState.conversationId) {
        void conversationActions.remove(conversationState.conversationId);
      }
    },
    onSelectConversation: (targetConversationId: string) => {
      void conversationActions.load(targetConversationId);
    },
  };

  const setupModel = {
    sourceOptions: SOURCE_OPTIONS,
    sources: setupState.sources,
    daysBack: setupState.daysBack,
    maxPapers: setupState.maxPapers,
  };

  const setupActions = {
    onToggleSource: controllerSetupActions.toggleSource,
    onDaysBackChange: controllerSetupActions.updateDaysBack,
    onMaxPapersChange: controllerSetupActions.updateMaxPapers,
  };

  const threadPreambleModel = {
    currentConversation: conversationState.currentConversation,
    shouldReplayConversation,
    conversationMessageCount: conversationState.conversationMessageCount,
    conversationMessages: conversationState.conversationMessages,
    selectedPaperIds: workspaceState.selectedPaperIds,
    recommendedPaperIds: workspaceState.recommendedPaperIds,
    mustReadPaperIds: workspaceState.mustReadPaperIds,
    paperTitleById: workspaceState.paperTitleById,
    hasWorkspace: workspaceState.hasWorkspace,
    suggestions: RESEARCH_SUGGESTIONS,
    error: runtimeState.error,
    replayHasError,
    activeWarnings: workspaceState.activeWarnings,
    replayHasWarning,
  };

  const composerModel = {
    mode: composerState.mode,
    value: composerState.value,
    sources: setupState.sources,
    selectedCount: workspaceState.selectedCount,
    selectedImportedCount: workspaceState.selectedImportedCount,
    importedPapersCount: workspaceState.importedPapers.length,
    activeSelectedFigure: figureViewState.activeSelectedFigure,
    isResearchLoading: runtimeState.isLoading,
    isAskLoading: composerState.isAskLoading,
    canRunResearch: composerState.canRunResearch,
    canAskCollection: composerState.canAskCollection,
  };

  const paperCount = workspaceState.activePapers.length;
  const docCount = workspaceState.activeTask?.imported_document_ids.length ?? 0;

  return (
    <section className="flex h-[100dvh] overflow-hidden">
      {/* ─── Sidebar ─── */}
      <aside
        className={`flex h-full shrink-0 flex-col border-r border-gray-200 bg-gray-50 transition-all duration-200 ${
          isSidebarOpen ? "w-[300px]" : "w-0 overflow-hidden border-r-0"
        }`}
      >
        <div className="flex h-full w-[300px] flex-col">
          {/* Sidebar header */}
          <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
            <span className="text-[13px] font-semibold text-gray-800">工作区</span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => { void conversationActions.create(); }}
                className="rounded-md p-1.5 text-gray-400 hover:bg-gray-200 hover:text-gray-600"
                title="新建会话"
              >
                <Plus className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={conversationActions.resetWorkspace}
                disabled={conversationState.isResetting}
                className="rounded-md p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500 disabled:opacity-40"
                title="清空记录"
              >
                <Trash2 className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => setIsSidebarOpen(false)}
                className="rounded-md p-1.5 text-gray-400 hover:bg-gray-200 hover:text-gray-600"
                title="收起侧栏"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div className="flex gap-3 border-b border-gray-200 px-4 py-2.5 text-[11px] text-gray-500">
            <span className="flex items-center gap-1"><FileText className="h-3 w-3" /> {paperCount} 篇</span>
            <span className="flex items-center gap-1"><BookOpenText className="h-3 w-3" /> {docCount} 文档</span>
            <span className="flex items-center gap-1"><CheckCheck className="h-3 w-3" /> {workspaceState.selectedCount} 选中</span>
          </div>

          {/* Sidebar content (scrollable) */}
          <div
            ref={sidebarScrollRef}
            className="min-h-0 flex-1 overflow-y-auto px-3 py-3 chat-scroll"
          >
            <div className="flex flex-col gap-3">
              <ResearchConversationsSection
                model={conversationsModel}
                actions={conversationsActions}
              />
              <ResearchSetupSection
                model={setupModel}
                actions={setupActions}
              />
              <ResearchWorkspaceSidebar
                model={workspaceSidebarModel}
                actions={workspaceSidebarActions}
              />
            </div>
          </div>
        </div>
      </aside>

      {/* ─── Main Conversation Area ─── */}
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        {/* Minimal top bar */}
        <header className="flex shrink-0 items-center gap-3 border-b border-gray-200 bg-white px-4 py-2.5">
          {!isSidebarOpen && (
            <button
              type="button"
              onClick={() => setIsSidebarOpen(true)}
              className="rounded-md p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              title="打开侧栏"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
          )}
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <Sparkles className="h-4 w-4 shrink-0 text-blue-600" />
            <h1 className="truncate text-[14px] font-semibold text-gray-900">
              {workspaceState.activeTask?.topic ?? "Research-Copilot"}
            </h1>
          </div>
          {paperCount > 0 && (
            <div className="flex items-center gap-2 text-[11px] text-gray-400">
              <span>{paperCount} 篇论文</span>
              {docCount > 0 && <span>{docCount} 文档</span>}
            </div>
          )}
        </header>

        {/* Thread area — centered, max-width for readability */}
        <div className="min-h-0 flex-1 overflow-hidden">
          <div
            ref={threadScrollRef}
            className="mx-auto h-full w-full max-w-3xl overflow-y-auto px-4 py-6 sm:px-6 chat-scroll"
          >
            <div className="flex min-h-full flex-col gap-4">
              <ResearchThreadPreamble
                model={threadPreambleModel}
                actions={{
                  onTogglePaperSelection: workspaceActions.togglePaperSelection,
                  onSelectSuggestion: composerActions.selectSuggestion,
                }}
              />

              {!shouldReplayConversation && (
                <ResearchWorkspaceResults
                  model={workspaceResultsModel}
                  actions={{
                    onTogglePaperSelection: workspaceActions.togglePaperSelection,
                  }}
                />
              )}

              {!shouldReplayConversation && (
                <ResearchThreadArtifacts
                  model={threadArtifactsModel}
                  actions={{
                    onTogglePaperSelection: workspaceActions.togglePaperSelection,
                    onOpenAskResultFigure: figureActions.openViewer,
                  }}
                />
              )}

              {qaState.askError && (
                <ThreadBubble role="system" title="问答失败">
                  <p>{qaState.askError}</p>
                </ThreadBubble>
              )}

              {runtimeState.isLoading && (
                <div className="flex items-start gap-3 chat-pop">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-500">
                    <Bot className="h-4 w-4" />
                  </div>
                  <div className="flex items-center gap-2 rounded-2xl bg-gray-100 px-4 py-3">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-400" />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-400" style={{ animationDelay: "200ms" }} />
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-gray-400" style={{ animationDelay: "400ms" }} />
                    <span className="ml-1 text-[13px] text-gray-500">思考中...</span>
                  </div>
                </div>
              )}

              <div ref={threadEndRef} />
            </div>
          </div>
        </div>

        {/* Bottom composer */}
        <ResearchComposer
          model={composerModel}
          actions={{
            onModeChange: composerActions.changeMode,
            onValueChange: composerActions.updateValue,
            onRunResearch: () => { void composerActions.runResearch(); },
            onAskCollection: () => { void composerActions.askCollection(); },
            onOpenActiveFigure: figureActions.openViewer,
            onClearActiveFigure: figureActions.clearActive,
          }}
        />
      </main>

      <ResearchFigureViewerModal
        figure={figureViewState.viewerFigure}
        onClose={figureActions.closeViewer}
      />
    </section>
  );
}

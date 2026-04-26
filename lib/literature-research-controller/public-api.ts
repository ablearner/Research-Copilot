import type {
  ResearchPaperFigurePreview,
  ResearchTodoItem,
} from "@/lib/types";
import { getPaperDocumentId, defaultImportedPaperIds, type ComposerMode } from "./shared";
import type { ResearchDerivedState } from "./derived";
import type { ResearchImportFlow } from "./import-flow";
import type { ResearchQaFigureFlow } from "./qa-figure-flow";
import type { ResearchRuntime } from "./runtime";
import type { ResearchConversationSession } from "./session";
import type { ResearchControllerState } from "./state";
import type { ResearchTodoFlow } from "./todo-flow";

function buildResearchControllerPublicStateGroups(
  controller: ResearchControllerState,
  derived: ResearchDerivedState
) {
  return {
    conversationState: {
      conversationId: controller.conversationId,
      conversations: controller.conversations,
      currentConversation: derived.currentConversation,
      conversationMessages: controller.conversationMessages,
      conversationMessageCount: derived.conversationMessageCount,
      isResetting: controller.isResetting,
    },
    setupState: {
      sources: controller.sources,
      daysBack: controller.daysBack,
      maxPapers: controller.maxPapers,
    },
    runtimeState: {
      agentResult: controller.agentResult,
      error: controller.error,
      isLoading: derived.isLoading,
      taskResult: controller.taskResult,
    },
    workspaceState: {
      activeImportJob: controller.activeImportJob,
      activePapers: derived.activePapers,
      activeReport: derived.activeReport,
      activeTask: derived.activeTask,
      activeWarnings: derived.activeWarnings,
      activeWorkspace: derived.activeWorkspace,
      comparisonResult: derived.comparisonResult,
      contextCompression: derived.contextCompression,
      hasWorkspace: derived.hasWorkspace,
      importedPapers: derived.importedPapers,
      importError: controller.importError,
      importJobRunning: derived.importJobRunning,
      importResult: controller.importResult,
      lastTaskId: controller.lastTaskId,
      mustReadPaperIds: derived.mustReadPaperIds,
      paperAnalysisResult: derived.paperAnalysisResult,
      paperTitleById: derived.paperTitleById,
      recommendationResult: derived.recommendationResult,
      recommendedPaperIds: derived.recommendedPaperIds,
      searchResult: controller.searchResult,
      selectedCount: derived.selectedCount,
      selectedImportedCount: derived.selectedImportedCount,
      selectedImportedPaperIds: controller.selectedImportedPaperIds,
      selectedPaperIds: controller.selectedPaperIds,
      getPaperDocumentId,
    },
    qaState: {
      askError: controller.askError,
      askQuestion: controller.askQuestion,
      askResult: controller.askResult,
      askResultQATrace: derived.askResultQATrace,
      askState: controller.askState,
    },
    composerState: {
      mode: controller.composerMode,
      value: derived.composerValue,
      topic: controller.topic,
      canAskCollection: derived.canAskCollection,
      canRunResearch: derived.canRunResearch,
      isAskLoading: derived.isAskLoading,
    },
    figureState: {
      activeSelectedFigure: derived.activeSelectedFigure,
      askResultFigure: controller.askResultFigure,
      analysisError: controller.figureAnalysisError,
      analysisResult: controller.figureAnalysisResult,
      analysisState: controller.figureAnalysisState,
      error: controller.figureError,
      panelPaperId: controller.figurePanelPaperId,
      previewResult: controller.paperFigureResult,
      requestState: controller.figureState,
      selectedFigure: controller.selectedFigure,
      viewerFigure: controller.imageViewerFigure,
    },
    todoState: {
      actionError: controller.todoActionError,
      actionNotice: controller.todoActionNotice,
    },
  };
}

function buildResearchControllerPublicActionGroups(
  controller: ResearchControllerState,
  session: ResearchConversationSession,
  runtime: ResearchRuntime,
  derived: ResearchDerivedState,
  importFlow: ResearchImportFlow,
  qaFigureFlow: ResearchQaFigureFlow,
  todoFlow: ResearchTodoFlow
) {
  return {
    conversationActions: {
      create: session.handleCreateConversation,
      remove: session.handleDeleteConversation,
      load: session.loadConversationById,
      resetWorkspace: session.requestWorkspaceReset,
    },
    composerActions: {
      changeMode: (mode: ComposerMode) => {
        controller.setComposerMode(mode);
      },
      updateValue: (value: string) => {
        if (controller.composerMode === "research") {
          controller.setTopic(value);
          return;
        }
        controller.setAskQuestion(value);
      },
      selectSuggestion: (suggestion: string) => {
        controller.setTopic(suggestion);
        controller.setComposerMode("research");
      },
      runResearch: () => runtime.handleRunAgent("research"),
      askCollection: qaFigureFlow.handleAskTaskCollection,
    },
    setupActions: {
      toggleSource: runtime.toggleSource,
      updateDaysBack: (value: number) => {
        controller.setDaysBack(value);
      },
      updateMaxPapers: (value: number) => {
        controller.setMaxPapers(value);
      },
    },
    workspaceActions: {
      refreshTask: runtime.handleRefreshTask,
      importSelected: importFlow.handleImportSelected,
      togglePaperSelection: runtime.togglePaperSelection,
    },
    importedPaperScopeActions: {
      toggle: (paperId: string) => {
        controller.setSelectedImportedPaperIds((current) =>
          current.includes(paperId)
            ? current.filter((item) => item !== paperId)
            : [...current, paperId]
        );
      },
      selectAll: () => {
        controller.setSelectedImportedPaperIds(
          defaultImportedPaperIds(derived.activePapers)
        );
      },
      clear: () => {
        controller.setSelectedImportedPaperIds([]);
      },
    },
    figureActions: {
      useAsAnchor: (figure: ResearchPaperFigurePreview) => {
        controller.setSelectedFigure(figure);
        controller.setAskAnchorDraft(figure);
        controller.setDismissedFigureId(null);
        controller.setComposerMode("qa");
      },
      clearActive: () => {
        if (derived.activeSelectedFigure) {
          controller.setDismissedFigureId(
            derived.activeSelectedFigure.figure_id
          );
        }
        controller.setSelectedFigure(null);
        controller.setAskAnchorDraft(null);
        controller.setAskResultFigure(null);
        controller.setImageViewerFigure(null);
      },
      openViewer: (figure: ResearchPaperFigurePreview) => {
        controller.setImageViewerFigure(figure);
      },
      closeViewer: () => {
        controller.setImageViewerFigure(null);
      },
      loadPaperFigures: qaFigureFlow.handleLoadPaperFigures,
      analyzePaperFigure: qaFigureFlow.handleAnalyzePaperFigure,
    },
    todoActions: {
      setStatus: todoFlow.handleTodoStatus,
      toggleDone: (
        todoId: string,
        status: ResearchTodoItem["status"]
      ) =>
        todoFlow.handleTodoStatus(
          todoId,
          status === "done" ? "open" : "done"
        ),
      toggleDismissed: (
        todoId: string,
        status: ResearchTodoItem["status"]
      ) =>
        todoFlow.handleTodoStatus(
          todoId,
          status === "dismissed" ? "open" : "dismissed"
        ),
      search: todoFlow.handleTodoSearch,
      import: todoFlow.handleTodoImport,
      isBusy: todoFlow.isTodoBusy,
    },
  };
}

export function buildLiteratureResearchControllerResult(
  controller: ResearchControllerState,
  session: ResearchConversationSession,
  runtime: ResearchRuntime,
  derived: ResearchDerivedState,
  importFlow: ResearchImportFlow,
  qaFigureFlow: ResearchQaFigureFlow,
  todoFlow: ResearchTodoFlow
) {
  return {
    ...buildResearchControllerPublicStateGroups(controller, derived),
    ...buildResearchControllerPublicActionGroups(
      controller,
      session,
      runtime,
      derived,
      importFlow,
      qaFigureFlow,
      todoFlow
    ),
  };
}

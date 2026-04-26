import {
  buildResearchQATraceSummary,
  comparisonFromWorkspaceMetadata,
  contextCompressionFromWorkspaceMetadata,
  paperAnalysisFromWorkspaceMetadata,
  parseResearchPaperFigurePreview,
  parseResearchVisualAnchor,
  recommendationFromWorkspaceMetadata,
  visualAnchorFigureFromResearchQAMetadata,
} from "@/lib/research-payloads";
import type {
  ResearchReport,
  ResearchWorkspaceState,
} from "@/lib/types";
import { isImportedPaper, uniqueTrimmedStrings } from "./shared";
import type { ResearchControllerState } from "./state";

export function buildResearchDerivedState(controller: ResearchControllerState) {
  const activeWarnings = uniqueTrimmedStrings(
    controller.taskResult?.warnings?.length
      ? controller.taskResult.warnings
      : (controller.searchResult?.warnings ?? [])
  );
  const activeReport: ResearchReport | null =
    controller.taskResult?.report ?? controller.searchResult?.report ?? null;
  const activePapers = controller.taskResult?.papers?.length
    ? controller.taskResult.papers
    : (controller.searchResult?.papers ?? []);
  const importedPapers = activePapers.filter(isImportedPaper);
  const activeTask = controller.taskResult?.task ?? null;
  const currentConversation =
    controller.conversations.find(
      (item) => item.conversation_id === controller.conversationId
    ) ?? null;
  const activeWorkspace: ResearchWorkspaceState | null =
    controller.agentResult?.workspace ??
    activeTask?.workspace ??
    activeReport?.workspace ??
    null;
  const comparisonResult = comparisonFromWorkspaceMetadata(
    activeWorkspace?.metadata
  );
  const paperAnalysisResult = paperAnalysisFromWorkspaceMetadata(
    activeWorkspace?.metadata
  );
  const recommendationResult = recommendationFromWorkspaceMetadata(
    activeWorkspace?.metadata
  );
  const contextCompression = contextCompressionFromWorkspaceMetadata(
    activeWorkspace?.metadata
  );
  const askResultQATrace = buildResearchQATraceSummary(
    controller.askResult?.qa.metadata
  );
  const workspaceVisualAnchorFigure = parseResearchPaperFigurePreview(
    activeWorkspace?.metadata?.last_visual_anchor_figure
  );
  const askResultVisualAnchorFigure = visualAnchorFigureFromResearchQAMetadata(
    controller.askResult?.qa.metadata
  );
  const workspaceVisualAnchor = parseResearchVisualAnchor(
    activeWorkspace?.metadata?.last_visual_anchor
  );
  const askResultVisualAnchor = parseResearchVisualAnchor(
    controller.askResult?.qa.metadata?.visual_anchor
  );
  const effectiveSelectedFigure =
    controller.askAnchorDraft ??
    controller.selectedFigure ??
    controller.figureAnalysisResult?.figure ??
    askResultVisualAnchorFigure ??
    workspaceVisualAnchorFigure ??
    controller.askResultFigure ??
    null;
  const activeSelectedFigure =
    effectiveSelectedFigure &&
    controller.dismissedFigureId === effectiveSelectedFigure.figure_id
      ? null
      : effectiveSelectedFigure;
  const dismissedAnchorMatches =
    controller.dismissedFigureId != null &&
    (effectiveSelectedFigure?.figure_id === controller.dismissedFigureId ||
      workspaceVisualAnchor?.figure_id === controller.dismissedFigureId ||
      askResultVisualAnchor?.figure_id === controller.dismissedFigureId);
  const effectiveVisualAnchor = {
    image_path:
      dismissedAnchorMatches
        ? null
        : activeSelectedFigure?.image_path ??
          workspaceVisualAnchor?.image_path ??
          askResultVisualAnchor?.image_path ??
          null,
    page_id:
      dismissedAnchorMatches
        ? null
        : activeSelectedFigure?.page_id ??
          workspaceVisualAnchor?.page_id ??
          askResultVisualAnchor?.page_id ??
          null,
    page_number:
      dismissedAnchorMatches
        ? null
        : activeSelectedFigure?.page_number ??
          workspaceVisualAnchor?.page_number ??
          askResultVisualAnchor?.page_number ??
          null,
    chart_id:
      dismissedAnchorMatches
        ? null
        : activeSelectedFigure?.chart_id ??
          workspaceVisualAnchor?.chart_id ??
          askResultVisualAnchor?.chart_id ??
          null,
    figure_id:
      dismissedAnchorMatches
        ? null
        : activeSelectedFigure?.figure_id ??
          workspaceVisualAnchor?.figure_id ??
          askResultVisualAnchor?.figure_id ??
          null,
  };
  const hasWorkspace = Boolean(
    activeReport || activePapers.length || controller.askResult
  );
  const selectedCount = controller.selectedPaperIds.length;
  const selectedImportedCount = controller.selectedImportedPaperIds.length;
  const conversationMessageCount = controller.conversationMessages.length;
  const recommendedPaperIds = new Set([
    ...(paperAnalysisResult?.recommended_paper_ids ?? []),
    ...(recommendationResult?.recommendations.map((item) => item.paper_id) ??
      []),
  ]);
  const mustReadPaperIds = new Set(activeWorkspace?.must_read_paper_ids ?? []);
  const paperTitleById = new Map(
    activePapers.map((paper) => [paper.paper_id, paper.title])
  );
  const composerValue =
    controller.composerMode === "research"
      ? controller.topic
      : controller.askQuestion;
  const importJobRunning =
    controller.importState === "loading" ||
    controller.activeImportJob?.status === "queued" ||
    controller.activeImportJob?.status === "running";
  const isLoading = controller.state === "loading";
  const isAskLoading = controller.askState === "loading";
  const canRunResearch =
    !isLoading &&
    Boolean(controller.topic.trim()) &&
    controller.sources.length > 0;
  const canAskCollection =
    !isAskLoading &&
    Boolean(activeTask?.task_id) &&
    Boolean(controller.askQuestion.trim()) &&
    selectedImportedCount > 0;

  return {
    activeWarnings,
    activeReport,
    activePapers,
    importedPapers,
    activeTask,
    currentConversation,
    activeWorkspace,
    comparisonResult,
    paperAnalysisResult,
    recommendationResult,
    contextCompression,
    askResultQATrace,
    workspaceVisualAnchorFigure,
    askResultVisualAnchorFigure,
    workspaceVisualAnchor,
    askResultVisualAnchor,
    activeSelectedFigure,
    effectiveVisualAnchor,
    hasWorkspace,
    selectedCount,
    selectedImportedCount,
    conversationMessageCount,
    recommendedPaperIds,
    mustReadPaperIds,
    paperTitleById,
    composerValue,
    importJobRunning,
    isLoading,
    isAskLoading,
    canRunResearch,
    canAskCollection,
  };
}

export type ResearchDerivedState = ReturnType<typeof buildResearchDerivedState>;

"use client";

import { useRef, useState } from "react";
import type {
  AnalyzeResearchPaperFigureResponse,
  ImportPapersResponse,
  PaperCandidate,
  ResearchAgentRunResponse,
  ResearchConversation,
  ResearchJob,
  ResearchMessage,
  ResearchPaperFigureListResponse,
  ResearchPaperFigurePreview,
  ResearchSource,
  ResearchTaskAskResponse,
  ResearchTaskResponse,
  RequestState,
  SearchPapersResponse,
} from "@/lib/types";
import {
  DEFAULT_ASK_QUESTION,
  DEFAULT_TOPIC,
  type ComposerMode,
  getDefaultSources,
  nextImportedPaperIds,
  RESEARCH_CONVERSATION_STORAGE_KEY,
  RESEARCH_TASK_STORAGE_KEY,
  IMPORT_JOB_STORAGE_KEY,
  sanitizeSelectedPaperIds,
} from "./shared";

function useResearchWorkspaceConfigState() {
  const [isResetting, setIsResetting] = useState(false);
  const [composerMode, setComposerMode] = useState<ComposerMode>("research");
  const [topic, setTopic] = useState(DEFAULT_TOPIC);
  const [daysBack, setDaysBack] = useState(180);
  const [maxPapers, setMaxPapers] = useState(12);
  const [sources, setSources] = useState<ResearchSource[]>(getDefaultSources);

  return {
    isResetting,
    setIsResetting,
    composerMode,
    setComposerMode,
    topic,
    setTopic,
    daysBack,
    setDaysBack,
    maxPapers,
    setMaxPapers,
    sources,
    setSources,
  };
}

function useResearchConversationStateSlice() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ResearchConversation[]>([]);
  const [conversationMessages, setConversationMessages] = useState<
    ResearchMessage[]
  >([]);

  return {
    conversationId,
    setConversationId,
    conversations,
    setConversations,
    conversationMessages,
    setConversationMessages,
  };
}

function useResearchRuntimeStateSlice() {
  const [state, setState] = useState<RequestState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [searchResult, setSearchResult] =
    useState<SearchPapersResponse | null>(null);
  const [taskResult, setTaskResult] =
    useState<ResearchTaskResponse | null>(null);
  const [agentResult, setAgentResult] =
    useState<ResearchAgentRunResponse | null>(null);
  const [lastTaskId, setLastTaskId] = useState<string | null>(null);
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [selectedImportedPaperIds, setSelectedImportedPaperIds] = useState<
    string[]
  >([]);

  return {
    state,
    setState,
    error,
    setError,
    searchResult,
    setSearchResult,
    taskResult,
    setTaskResult,
    agentResult,
    setAgentResult,
    lastTaskId,
    setLastTaskId,
    selectedPaperIds,
    setSelectedPaperIds,
    selectedImportedPaperIds,
    setSelectedImportedPaperIds,
  };
}

function useResearchImportStateSlice() {
  const [importState, setImportState] = useState<RequestState>("idle");
  const [importError, setImportError] = useState<string | null>(null);
  const [importResult, setImportResult] =
    useState<ImportPapersResponse | null>(null);
  const [activeImportJob, setActiveImportJob] =
    useState<ResearchJob | null>(null);

  return {
    importState,
    setImportState,
    importError,
    setImportError,
    importResult,
    setImportResult,
    activeImportJob,
    setActiveImportJob,
  };
}

function useResearchQAFigureStateSlice() {
  const [askState, setAskState] = useState<RequestState>("idle");
  const [askError, setAskError] = useState<string | null>(null);
  const [askQuestion, setAskQuestion] = useState(DEFAULT_ASK_QUESTION);
  const [askResult, setAskResult] =
    useState<ResearchTaskAskResponse | null>(null);
  const [selectedFigure, setSelectedFigure] =
    useState<ResearchPaperFigurePreview | null>(null);
  const [askAnchorDraft, setAskAnchorDraft] =
    useState<ResearchPaperFigurePreview | null>(null);
  const [askResultFigure, setAskResultFigure] =
    useState<ResearchPaperFigurePreview | null>(null);
  const [dismissedFigureId, setDismissedFigureId] = useState<string | null>(null);
  const [imageViewerFigure, setImageViewerFigure] =
    useState<ResearchPaperFigurePreview | null>(null);
  const [figurePanelPaperId, setFigurePanelPaperId] = useState<string | null>(null);
  const [figureState, setFigureState] = useState<RequestState>("idle");
  const [figureError, setFigureError] = useState<string | null>(null);
  const [paperFigureResult, setPaperFigureResult] =
    useState<ResearchPaperFigureListResponse | null>(null);
  const [figureAnalysisState, setFigureAnalysisState] =
    useState<RequestState>("idle");
  const [figureAnalysisError, setFigureAnalysisError] =
    useState<string | null>(null);
  const [figureAnalysisResult, setFigureAnalysisResult] =
    useState<AnalyzeResearchPaperFigureResponse | null>(null);

  return {
    askState,
    setAskState,
    askError,
    setAskError,
    askQuestion,
    setAskQuestion,
    askResult,
    setAskResult,
    selectedFigure,
    setSelectedFigure,
    askAnchorDraft,
    setAskAnchorDraft,
    askResultFigure,
    setAskResultFigure,
    dismissedFigureId,
    setDismissedFigureId,
    imageViewerFigure,
    setImageViewerFigure,
    figurePanelPaperId,
    setFigurePanelPaperId,
    figureState,
    setFigureState,
    figureError,
    setFigureError,
    paperFigureResult,
    setPaperFigureResult,
    figureAnalysisState,
    setFigureAnalysisState,
    figureAnalysisError,
    setFigureAnalysisError,
    figureAnalysisResult,
    setFigureAnalysisResult,
  };
}

function useResearchTodoStateSlice() {
  const [todoActionKey, setTodoActionKey] = useState<string | null>(null);
  const [todoActionError, setTodoActionError] = useState<string | null>(null);
  const [todoActionNotice, setTodoActionNotice] = useState<string | null>(null);

  return {
    todoActionKey,
    setTodoActionKey,
    todoActionError,
    setTodoActionError,
    todoActionNotice,
    setTodoActionNotice,
  };
}

export function useResearchControllerState() {
  const hasInitializedRef = useRef(false);
  const workspaceConfig = useResearchWorkspaceConfigState();
  const conversation = useResearchConversationStateSlice();
  const runtime = useResearchRuntimeStateSlice();
  const importState = useResearchImportStateSlice();
  const qaFigure = useResearchQAFigureStateSlice();
  const todo = useResearchTodoStateSlice();

  return {
    hasInitializedRef,
    ...workspaceConfig,
    ...conversation,
    ...runtime,
    ...importState,
    ...qaFigure,
    ...todo,
  };
}

export type ResearchControllerState = ReturnType<
  typeof useResearchControllerState
>;

export function clearPersistedWorkspaceCache() {
  window.localStorage.removeItem(RESEARCH_TASK_STORAGE_KEY);
  window.localStorage.removeItem(RESEARCH_CONVERSATION_STORAGE_KEY);
  window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
}

export function clearWorkspaceView(controller: ResearchControllerState) {
  window.localStorage.removeItem(RESEARCH_TASK_STORAGE_KEY);
  window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
  controller.setConversationId(null);
  controller.setConversationMessages([]);
  controller.setComposerMode("research");
  controller.setTopic(DEFAULT_TOPIC);
  controller.setAskQuestion(DEFAULT_ASK_QUESTION);
  controller.setSelectedFigure(null);
  controller.setAskAnchorDraft(null);
  controller.setAskResultFigure(null);
  controller.setDismissedFigureId(null);
  controller.setImageViewerFigure(null);
  controller.setFigurePanelPaperId(null);
  controller.setFigureState("idle");
  controller.setFigureError(null);
  controller.setPaperFigureResult(null);
  controller.setFigureAnalysisState("idle");
  controller.setFigureAnalysisError(null);
  controller.setFigureAnalysisResult(null);
  controller.setState("idle");
  controller.setError(null);
  controller.setSearchResult(null);
  controller.setTaskResult(null);
  controller.setAgentResult(null);
  controller.setLastTaskId(null);
  controller.setSelectedPaperIds([]);
  controller.setSelectedImportedPaperIds([]);
  controller.setImportState("idle");
  controller.setImportError(null);
  controller.setImportResult(null);
  controller.setActiveImportJob(null);
  controller.setAskState("idle");
  controller.setAskError(null);
  controller.setAskResult(null);
  controller.setTodoActionKey(null);
  controller.setTodoActionError(null);
  controller.setTodoActionNotice(null);
}

export function applyCurrentPaperSelections(
  controller: ResearchControllerState,
  papers: PaperCandidate[]
) {
  controller.setSelectedPaperIds((current) =>
    sanitizeSelectedPaperIds(current, papers)
  );
  controller.setSelectedImportedPaperIds((current) =>
    nextImportedPaperIds(current, papers)
  );
}

export function applySnapshotPaperSelections(
  controller: ResearchControllerState,
  paperIds: string[],
  papers: PaperCandidate[]
) {
  controller.setSelectedPaperIds(sanitizeSelectedPaperIds(paperIds, papers));
  controller.setSelectedImportedPaperIds((current) =>
    nextImportedPaperIds(current, papers)
  );
}

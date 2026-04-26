"use client";

import { useEffect } from "react";
import {
  createResearchConversation,
  deleteResearchConversation,
  getResearchConversation,
  getResearchJob,
  listResearchConversations,
  resetResearchWorkspace,
} from "@/lib/research-api";
import type {
  CreateResearchConversationRequest,
  ResearchConversationResponse,
  ResearchRuntimeEvent,
} from "@/lib/types";
import {
  DEFAULT_TOPIC,
  getDefaultSources,
  IMPORT_JOB_STORAGE_KEY,
  RESEARCH_CONVERSATION_STORAGE_KEY,
  RESEARCH_TASK_STORAGE_KEY,
} from "./shared";
import {
  applySnapshotPaperSelections,
  clearPersistedWorkspaceCache,
  clearWorkspaceView,
  type ResearchControllerState,
} from "./state";

export function useResearchConversationSession(
  controller: ResearchControllerState
) {
  const hasInitializedRef = controller.hasInitializedRef;

  function findLatestEvent(
    events: ResearchRuntimeEvent[] | null | undefined,
    predicate: (event: ResearchRuntimeEvent) => boolean
  ): ResearchRuntimeEvent | null {
    if (!events?.length) return null;
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const event = events[index];
      if (predicate(event)) return event;
    }
    return null;
  }

  async function refreshConversationList(activeId?: string | null) {
    const items = await listResearchConversations();
    controller.setConversations(items);
    if (activeId) {
      const matched = items.find(
        (item) => item.conversation_id === activeId
      );
      if (matched) {
        controller.setConversationId(matched.conversation_id);
      }
    }
    return items;
  }

  function applyConversationSnapshot(
    response: ResearchConversationResponse,
    options?: { preserveAgentResult?: boolean }
  ) {
    const snapshot = response.conversation.snapshot;
    const contextSummary = snapshot.context_summary ?? null;
    const recentEvents = snapshot.recent_events ?? [];
    const preserveAgentResult = options?.preserveAgentResult ?? false;
    const snapshotPapers = snapshot.task_result?.papers?.length
      ? snapshot.task_result.papers
      : (snapshot.search_result?.papers ?? []);
    const nextTaskId =
      snapshot.task_result?.task.task_id ??
      response.conversation.task_id ??
      null;
    const activeJobId = snapshot.active_job_id;

    controller.setConversationId(response.conversation.conversation_id);
    window.localStorage.setItem(
      RESEARCH_CONVERSATION_STORAGE_KEY,
      response.conversation.conversation_id
    );
    controller.setConversationMessages(response.messages);
    controller.setTopic(snapshot.topic || contextSummary?.topic || DEFAULT_TOPIC);
    controller.setDaysBack(snapshot.days_back || 180);
    controller.setMaxPapers(snapshot.max_papers || 12);
    controller.setSources(
      snapshot.sources?.length ? snapshot.sources : getDefaultSources()
    );
    controller.setComposerMode(
      snapshot.composer_mode === "qa" ? "qa" : "research"
    );
    applySnapshotPaperSelections(
      controller,
      snapshot.selected_paper_ids ?? [],
      snapshotPapers
    );
    controller.setSearchResult(snapshot.search_result ?? null);
    controller.setTaskResult(snapshot.task_result ?? null);
    controller.setImportResult(snapshot.import_result ?? null);
    controller.setAskResult(snapshot.ask_result ?? null);
    controller.setAskAnchorDraft(null);
    if (!preserveAgentResult) {
      controller.setAgentResult(null);
    }
    controller.setLastTaskId(nextTaskId);
    if (nextTaskId) {
      window.localStorage.setItem(RESEARCH_TASK_STORAGE_KEY, nextTaskId);
    } else {
      window.localStorage.removeItem(RESEARCH_TASK_STORAGE_KEY);
    }
    const latestFailureEvent = findLatestEvent(
      recentEvents,
      (event) =>
        event.event_type === "task_failed" ||
        event.event_type === "tool_failed"
    );
    const latestSuccessEvent = findLatestEvent(
      recentEvents,
      (event) =>
        event.event_type === "task_completed" ||
        event.event_type === "tool_succeeded" ||
        event.event_type === "agent_routed"
    );
    const latestNoticeEvent = findLatestEvent(
      recentEvents,
      (event) =>
        event.event_type === "memory_updated" ||
        event.event_type === "task_completed"
    );
    controller.setState(
      snapshot.last_error
        ? "error"
        : latestFailureEvent
        ? "error"
        : snapshot.task_result ||
            snapshot.search_result ||
            snapshot.import_result ||
            snapshot.ask_result ||
            latestSuccessEvent
          ? "success"
          : "idle"
    );
    controller.setError(
      snapshot.last_error ??
        (typeof latestFailureEvent?.payload?.last_error === "string"
          ? latestFailureEvent.payload.last_error
          : null)
    );
    controller.setImportState(
      activeJobId
        ? "loading"
        : snapshot.import_result
          ? "success"
          : "idle"
    );
    controller.setImportError(null);
    controller.setAskState(snapshot.ask_result ? "success" : "idle");
    controller.setAskError(null);
    controller.setTodoActionError(null);
    controller.setTodoActionNotice(
      snapshot.last_notice ??
        (typeof contextSummary?.status_summary === "string" &&
        contextSummary.status_summary.trim()
          ? contextSummary.status_summary
          : typeof latestNoticeEvent?.payload?.notice === "string"
            ? String(latestNoticeEvent.payload.notice)
            : null)
    );
    controller.setActiveImportJob(null);

    if (activeJobId) {
      window.localStorage.setItem(IMPORT_JOB_STORAGE_KEY, activeJobId);
      void getResearchJob(activeJobId)
        .then((job) => {
          controller.setActiveImportJob(job);
        })
        .catch(() => {
          window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
        });
    } else {
      window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
    }
  }

  async function loadConversationById(
    nextConversationId: string,
    options?: { preserveAgentResult?: boolean }
  ) {
    const response = await getResearchConversation(nextConversationId);
    applyConversationSnapshot(response, options);
    await refreshConversationList(nextConversationId);
    return response;
  }

  function buildConversationCreateRequest(): CreateResearchConversationRequest {
    return {
      topic: controller.topic.trim() || DEFAULT_TOPIC,
      days_back: controller.daysBack,
      max_papers: controller.maxPapers,
      sources: controller.sources,
    };
  }

  async function createAndLoadConversation(
    options?: { preserveCurrentInput?: boolean }
  ) {
    const response = await createResearchConversation(
      buildConversationCreateRequest()
    );
    if (!options?.preserveCurrentInput) {
      clearWorkspaceView(controller);
    }
    applyConversationSnapshot(response);
    await refreshConversationList(response.conversation.conversation_id);
    return response.conversation.conversation_id;
  }

  async function ensureConversationId() {
    if (controller.conversationId) return controller.conversationId;
    return await createAndLoadConversation({ preserveCurrentInput: true });
  }

  async function handleClearWorkspace(options?: { silent?: boolean }) {
    const silent = options?.silent ?? false;
    controller.setIsResetting(true);
    try {
      await resetResearchWorkspace();
      clearPersistedWorkspaceCache();
      clearWorkspaceView(controller);
      const nextConversationId = await createAndLoadConversation();
      await refreshConversationList(nextConversationId);
      if (!silent) {
        controller.setTodoActionNotice("已清空研究记录，当前工作区已重置。");
      }
    } catch (resetError) {
      clearWorkspaceView(controller);
      if (!silent) {
        controller.setError(
          resetError instanceof Error
            ? resetError.message
            : "研究记录清空失败"
        );
      }
    } finally {
      controller.setIsResetting(false);
    }
  }

  async function handleCreateConversation() {
    try {
      const nextConversationId = await createAndLoadConversation();
      controller.setTodoActionNotice("已创建新的研究会话。");
      controller.setError(null);
      await refreshConversationList(nextConversationId);
    } catch (conversationError) {
      controller.setError(
        conversationError instanceof Error
          ? conversationError.message
          : "创建研究会话失败"
      );
    }
  }

  async function handleDeleteConversation(targetConversationId: string) {
    const confirmed = window.confirm("确认删除这个研究会话吗？会话消息和快照会一起移除。");
    if (!confirmed) return;
    try {
      await deleteResearchConversation(targetConversationId);
      if (controller.conversationId === targetConversationId) {
        clearWorkspaceView(controller);
      }
      const items = await refreshConversationList();
      const nextConversation = items.find(
        (item) => item.conversation_id !== targetConversationId
      );
      if (nextConversation) {
        await loadConversationById(nextConversation.conversation_id);
      } else {
        await createAndLoadConversation();
      }
      controller.setTodoActionNotice("研究会话已删除。");
      controller.setError(null);
    } catch (conversationError) {
      controller.setError(
        conversationError instanceof Error
          ? conversationError.message
          : "删除研究会话失败"
      );
    }
  }

  function requestWorkspaceReset() {
    if (controller.isResetting) return;
    const confirmed = window.confirm(
      "确认清空当前研究记录吗？这会删除已保存的任务、对话和后台导入作业。"
    );
    if (!confirmed) return;
    void handleClearWorkspace();
  }

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;
    void (async () => {
      try {
        const storedConversationId = window.localStorage.getItem(
          RESEARCH_CONVERSATION_STORAGE_KEY
        );
        const items = await refreshConversationList(storedConversationId);
        const preferredConversationId =
          (storedConversationId &&
            items.some(
              (item) => item.conversation_id === storedConversationId
            ) &&
            storedConversationId) ||
          items[0]?.conversation_id ||
          null;
        if (preferredConversationId) {
          await loadConversationById(preferredConversationId);
          return;
        }
        await createAndLoadConversation();
      } catch (initializationError) {
        clearWorkspaceView(controller);
        controller.setError(
          initializationError instanceof Error
            ? initializationError.message
            : "初始化研究会话失败"
        );
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    loadConversationById,
    ensureConversationId,
    handleCreateConversation,
    handleDeleteConversation,
    requestWorkspaceReset,
  };
}

export type ResearchConversationSession = ReturnType<
  typeof useResearchConversationSession
>;

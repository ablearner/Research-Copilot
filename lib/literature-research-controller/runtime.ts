"use client";

import { runResearchAgent } from "@/lib/research-api";
import {
  parseResearchQADocumentIds,
  parseResearchQAPaperScope,
  selectionWarningsFromResearchQAMetadata,
  visualAnchorFigureFromResearchQAMetadata,
} from "@/lib/research-payloads";
import type {
  ResearchAgentMode,
  ResearchAgentRunResponse,
  ResearchSource,
  ResearchTaskResponse,
} from "@/lib/types";
import { type ComposerMode, toTaskSnapshot } from "./shared";
import {
  applyCurrentPaperSelections,
  type ResearchControllerState,
} from "./state";
import type { ResearchConversationSession } from "./session";
import { RESEARCH_TASK_STORAGE_KEY } from "./shared";
import { isImportedPaper } from "./shared";

export function useResearchRuntime(
  controller: ResearchControllerState,
  session: ResearchConversationSession
) {
  function applyTaskSnapshot(taskSnapshot: ResearchTaskResponse) {
    controller.setTaskResult(taskSnapshot);
    controller.setLastTaskId(taskSnapshot.task.task_id);
    window.localStorage.setItem(
      RESEARCH_TASK_STORAGE_KEY,
      taskSnapshot.task.task_id
    );
    applyCurrentPaperSelections(controller, taskSnapshot.papers);
  }

  function applyAgentSnapshot(
    response: ResearchAgentRunResponse,
    nextMode?: ComposerMode
  ) {
    controller.setAgentResult(response);
    controller.setSearchResult(null);

    if (response.task) {
      applyTaskSnapshot(
        toTaskSnapshot({
          task: response.task,
          papers: response.papers,
          report: response.report,
          warnings: response.warnings,
        })
      );
    }

    if (response.import_result) {
      controller.setImportResult(response.import_result);
    }

    if (response.qa && response.task) {
      const qaMetadata = response.qa.metadata ?? {};
      const responseAnchorFigure =
        visualAnchorFigureFromResearchQAMetadata(qaMetadata);
      const paperScope = parseResearchQAPaperScope(qaMetadata);
      const scopedDocumentIds = parseResearchQADocumentIds(qaMetadata);
      const selectionWarnings =
        selectionWarningsFromResearchQAMetadata(qaMetadata);
      const fallbackPaperIds =
        paperScope.scope_mode === "all_imported"
          ? response.papers
              .filter((paper) => isImportedPaper(paper))
              .map((paper) => paper.paper_id)
          : [];
      const resolvedDocumentIds =
        scopedDocumentIds.length > 0 || paperScope.scope_mode !== "all_imported"
          ? scopedDocumentIds
          : response.task.imported_document_ids;

      controller.setAskResult({
        task_id: response.task.task_id,
        paper_ids:
          paperScope.paper_ids.length > 0
            ? paperScope.paper_ids
            : fallbackPaperIds,
        document_ids: resolvedDocumentIds,
        scope_mode: paperScope.scope_mode,
        qa: response.qa,
        report: response.report ?? null,
        todo_items: response.task.todo_items,
        warnings: selectionWarnings,
      });
      controller.setAskResultFigure(responseAnchorFigure);
    }

    if (nextMode) controller.setComposerMode(nextMode);
    else if (response.qa) controller.setComposerMode("qa");
    else if (response.task) controller.setComposerMode("qa");
  }

  function toggleSource(source: ResearchSource) {
    controller.setSources((current) =>
      current.includes(source)
        ? current.filter((item) => item !== source)
        : [...current, source]
    );
  }

  function togglePaperSelection(paperId: string) {
    controller.setSelectedPaperIds((current) =>
      current.includes(paperId)
        ? current.filter((item) => item !== paperId)
        : [...current, paperId]
    );
  }

  async function handleRunAgent(mode: ResearchAgentMode) {
    const message =
      mode === "qa" ? controller.askQuestion.trim() : controller.topic.trim();
    if (!message || !controller.sources.length) return;
    controller.setState("loading");
    controller.setError(null);
    try {
      const activeConversationId = await session.ensureConversationId();
      const response = await runResearchAgent({
        message,
        mode,
        conversation_id: activeConversationId,
        task_id:
          mode === "research"
            ? undefined
            : (controller.taskResult?.task.task_id ?? controller.lastTaskId),
        days_back: controller.daysBack,
        max_papers: controller.maxPapers,
        sources: controller.sources,
        selected_paper_ids: controller.selectedPaperIds,
        auto_import: false,
        import_top_k: 0,
        include_graph: true,
        include_embeddings: true,
        top_k: 10,
        skill_name: "research_report",
        reasoning_style: "cot",
      });
      applyAgentSnapshot(response);
      await session.loadConversationById(activeConversationId, {
        preserveAgentResult: true,
      });
      controller.setState("success");
    } catch (searchError) {
      controller.setState("error");
      controller.setError(
        searchError instanceof Error
          ? searchError.message
          : "Research agent 运行失败"
      );
    }
  }

  async function handleRefreshTask() {
    if (!controller.lastTaskId) return;
    controller.setState("loading");
    controller.setError(null);
    try {
      const activeConversationId = await session.ensureConversationId();
      const response = await runResearchAgent({
        message: controller.taskResult?.task.topic ?? controller.topic.trim(),
        mode: "research",
        conversation_id: activeConversationId,
        task_id: controller.lastTaskId,
        days_back: controller.daysBack,
        max_papers: controller.maxPapers,
        sources: controller.sources,
        auto_import: false,
        import_top_k: 0,
        skill_name: "research_report",
      });
      applyAgentSnapshot(response);
      await session.loadConversationById(activeConversationId, {
        preserveAgentResult: true,
      });
      controller.setState("success");
    } catch (taskError) {
      controller.setState("error");
      controller.setError(
        taskError instanceof Error
          ? taskError.message
          : "Research agent 继续运行失败"
      );
    }
  }

  return {
    applyTaskSnapshot,
    applyAgentSnapshot,
    toggleSource,
    togglePaperSelection,
    handleRunAgent,
    handleRefreshTask,
  };
}

export type ResearchRuntime = ReturnType<typeof useResearchRuntime>;

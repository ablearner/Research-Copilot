"use client";

import { useEffect } from "react";
import {
  analyzeResearchPaperFigure,
  listResearchPaperFigures,
  runResearchAgent,
} from "@/lib/research-api";
import { visualAnchorFigureFromResearchQAMetadata } from "@/lib/research-payloads";
import type {
  PaperCandidate,
  ResearchPaperFigurePreview,
} from "@/lib/types";
import {
  getPaperDocumentId,
  isImportedPaper,
  uniqueTrimmedStrings,
} from "./shared";
import type { ResearchDerivedState } from "./derived";
import type { ResearchRuntime } from "./runtime";
import type { ResearchConversationSession } from "./session";
import type { ResearchControllerState } from "./state";

export function useResearchQaFigureFlow(
  controller: ResearchControllerState,
  session: ResearchConversationSession,
  derived: ResearchDerivedState,
  runtime: Pick<ResearchRuntime, "applyAgentSnapshot">
) {
  const workspaceVisualAnchorFigure = derived.workspaceVisualAnchorFigure;
  const activeSelectedFigure = derived.activeSelectedFigure;
  const { dismissedFigureId, setAskAnchorDraft, setSelectedFigure } = controller;

  useEffect(() => {
    if (!workspaceVisualAnchorFigure) return;
    if (dismissedFigureId === workspaceVisualAnchorFigure.figure_id) {
      return;
    }
    setSelectedFigure((current) => {
      if (current?.figure_id === workspaceVisualAnchorFigure.figure_id) {
        return current;
      }
      return workspaceVisualAnchorFigure;
    });
  }, [dismissedFigureId, setSelectedFigure, workspaceVisualAnchorFigure]);

  useEffect(() => {
    if (!activeSelectedFigure) return;
    if (dismissedFigureId === activeSelectedFigure.figure_id) {
      return;
    }
    setAskAnchorDraft((current) => current ?? activeSelectedFigure);
  }, [activeSelectedFigure, dismissedFigureId, setAskAnchorDraft]);

  async function handleAskTaskCollection() {
    if (!controller.taskResult?.task.task_id || !controller.askQuestion.trim()) {
      return;
    }

    const importedScopePapers = derived.activePapers.filter(isImportedPaper);
    if (!importedScopePapers.length) {
      controller.setAskState("error");
      controller.setAskError("请先在左侧导入至少一篇论文，再进行研究问答。");
      return;
    }

    const scopedPapers = importedScopePapers.filter((paper) =>
      controller.selectedImportedPaperIds.includes(paper.paper_id)
    );
    if (!scopedPapers.length) {
      controller.setAskState("error");
      controller.setAskError("请先勾选至少一篇已导入论文，再进行研究问答。");
      return;
    }

    controller.setAskState("loading");
    controller.setAskError(null);

    try {
      const activeConversationId = await session.ensureConversationId();
      const anchoredFigure = derived.activeSelectedFigure;
      const anchoredVisual = derived.effectiveVisualAnchor;
      const askPayload = {
        message: controller.askQuestion.trim(),
        mode: "qa" as const,
        task_id: controller.taskResult.task.task_id,
        conversation_id: activeConversationId,
        sources: controller.sources,
        top_k: 10,
        selected_paper_ids: scopedPapers.map((paper) => paper.paper_id),
        selected_document_ids: uniqueTrimmedStrings(
          scopedPapers.map((paper) => getPaperDocumentId(paper))
        ),
        auto_import: false,
        import_top_k: 0,
        include_graph: true,
        include_embeddings: true,
        chart_image_path: anchoredVisual.image_path,
        page_id: anchoredVisual.page_id,
        page_number: anchoredVisual.page_number ?? undefined,
        chart_id: anchoredVisual.chart_id,
        skill_name: "research_report",
        reasoning_style: "cot",
        metadata: {
          ui_scope: "selected_imported_papers",
          ui_visual_anchor_supplied: Boolean(
            anchoredFigure ||
              anchoredVisual.image_path ||
              anchoredVisual.chart_id
          ),
          ui_selected_figure_id:
            anchoredFigure?.figure_id ?? anchoredVisual.figure_id ?? null,
        },
      };
      console.info("Research ask payload", askPayload);
      const response = await runResearchAgent(askPayload);
      runtime.applyAgentSnapshot(response, "qa");
      const responseAnchorFigure = visualAnchorFigureFromResearchQAMetadata(
        response.qa?.metadata
      );
      controller.setAskResultFigure(responseAnchorFigure ?? anchoredFigure ?? null);
      controller.setComposerMode("qa");
      await session.loadConversationById(activeConversationId, {
        preserveAgentResult: true,
      });
      controller.setAskState("success");
    } catch (taskError) {
      controller.setAskState("error");
      controller.setAskError(
        taskError instanceof Error
          ? taskError.message
          : "Research agent 问答失败"
      );
    }
  }

  async function handleLoadPaperFigures(paper: PaperCandidate) {
    if (!controller.taskResult?.task.task_id) return;
    controller.setFigurePanelPaperId(paper.paper_id);
    controller.setFigureState("loading");
    controller.setFigureError(null);
    try {
      const response = await listResearchPaperFigures(
        controller.taskResult.task.task_id,
        paper.paper_id
      );
      controller.setPaperFigureResult(response);
      controller.setFigureState("success");
    } catch (reason) {
      controller.setFigureState("error");
      controller.setFigureError(
        reason instanceof Error ? reason.message : "加载论文图表失败。"
      );
    }
  }

  async function handleAnalyzePaperFigure(figure: ResearchPaperFigurePreview) {
    if (!controller.taskResult?.task.task_id) return;
    controller.setFigureAnalysisState("loading");
    controller.setFigureAnalysisError(null);
    try {
      const response = await analyzeResearchPaperFigure(
        controller.taskResult.task.task_id,
        figure.paper_id,
        {
          figure_id: figure.figure_id,
          page_id: figure.page_id,
          chart_id: figure.chart_id,
          image_path: figure.image_path ?? null,
          question: "请解释这张图表在表达什么，重点看趋势、对比关系和结论边界。",
        }
      );
      controller.setFigureAnalysisResult(response);
      controller.setSelectedFigure(response.figure);
      controller.setAskAnchorDraft(response.figure);
      controller.setDismissedFigureId(null);
      controller.setFigureAnalysisState("success");
      controller.setComposerMode("qa");
    } catch (reason) {
      controller.setFigureAnalysisState("error");
      controller.setFigureAnalysisError(
        reason instanceof Error ? reason.message : "图表分析失败。"
      );
    }
  }

  return {
    handleAskTaskCollection,
    handleLoadPaperFigures,
    handleAnalyzePaperFigure,
  };
}

export type ResearchQaFigureFlow = ReturnType<typeof useResearchQaFigureFlow>;

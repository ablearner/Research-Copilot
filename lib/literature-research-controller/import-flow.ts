"use client";

import { useEffect } from "react";
import {
  getResearchJob,
  getResearchTask,
  startResearchImportJob,
} from "@/lib/research-api";
import type {
  ImportPapersResponse,
  ResearchTaskAskResponse,
  ResearchTaskResponse,
} from "@/lib/types";
import { asRecord } from "@/lib/value-coercion";
import {
  IMPORT_JOB_POLL_INTERVAL_MS,
  IMPORT_JOB_STORAGE_KEY,
} from "./shared";
import type { ResearchControllerState } from "./state";
import type { ResearchConversationSession } from "./session";
import type { ResearchRuntime } from "./runtime";

export function useResearchImportFlow(
  controller: ResearchControllerState,
  session: ResearchConversationSession,
  runtime: ResearchRuntime
) {
  async function applyImportJobSnapshot(job: NonNullable<typeof controller.activeImportJob>) {
    const output = asRecord(job.output) ?? {};
    const taskSnapshot =
      (output.task_result as ResearchTaskResponse | undefined) ?? null;
    const importSnapshot =
      (output.import_result as ImportPapersResponse | undefined) ?? null;
    const askSnapshot =
      (output.ask_result as ResearchTaskAskResponse | undefined) ?? null;
    const qaErrorMessage =
      typeof output.qa_error_message === "string"
        ? output.qa_error_message
        : null;

    if (taskSnapshot) {
      runtime.applyTaskSnapshot(taskSnapshot);
    } else if (job.task_id) {
      try {
        const refreshedTask = await getResearchTask(job.task_id);
        runtime.applyTaskSnapshot(refreshedTask);
      } catch {
        // Keep the last in-memory snapshot when refresh fails.
      }
    }

    if (importSnapshot) {
      controller.setImportResult(importSnapshot);
    }
    if (askSnapshot) {
      controller.setAskResult(askSnapshot);
      controller.setComposerMode("qa");
      controller.setAskState("success");
      controller.setAskError(null);
    } else if (qaErrorMessage) {
      controller.setAskState("error");
      controller.setAskError(qaErrorMessage);
      controller.setComposerMode("qa");
    }

    if (job.status === "completed") {
      controller.setImportState("success");
      controller.setImportError(null);
    } else {
      controller.setImportState("error");
      controller.setImportError(
        job.error_message ?? job.progress_message ?? "后台导入任务失败"
      );
    }

    if (controller.conversationId) {
      try {
        await session.loadConversationById(controller.conversationId, {
          preserveAgentResult: true,
        });
      } catch {
        // Keep current in-memory state if conversation refresh fails.
      }
    }
  }

  useEffect(() => {
    if (!controller.activeImportJob?.job_id) return;
    let cancelled = false;
    let timeoutId: number | null = null;

    const pollJob = async () => {
      try {
        const job = await getResearchJob(controller.activeImportJob!.job_id);
        if (cancelled) return;
        controller.setActiveImportJob(job);
        if (job.status === "queued" || job.status === "running") {
          timeoutId = window.setTimeout(() => {
            void pollJob();
          }, IMPORT_JOB_POLL_INTERVAL_MS);
          return;
        }
        window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
        controller.setActiveImportJob(null);
        await applyImportJobSnapshot(job);
      } catch (jobError) {
        if (cancelled) return;
        window.localStorage.removeItem(IMPORT_JOB_STORAGE_KEY);
        controller.setActiveImportJob(null);
        controller.setImportState("error");
        controller.setImportError(
          jobError instanceof Error
            ? jobError.message
            : "后台导入任务轮询失败"
        );
      }
    };

    if (
      controller.activeImportJob.status === "queued" ||
      controller.activeImportJob.status === "running"
    ) {
      void pollJob();
    }

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controller.activeImportJob?.job_id]);

  async function handleImportSelected() {
    if (!controller.selectedPaperIds.length || !controller.taskResult?.task.task_id) {
      return;
    }
    controller.setImportState("loading");
    controller.setImportError(null);
    controller.setImportResult(null);
    controller.setAskError(null);
    controller.setAskResult(null);
    try {
      const activeConversationId = await session.ensureConversationId();
      const job = await startResearchImportJob({
        task_id: controller.taskResult.task.task_id,
        conversation_id: activeConversationId,
        paper_ids: controller.selectedPaperIds,
        papers: [],
        include_graph: true,
        include_embeddings: true,
        skill_name: "research_report",
        top_k: 10,
        reasoning_style: "cot",
      });
      controller.setActiveImportJob(job);
      window.localStorage.setItem(IMPORT_JOB_STORAGE_KEY, job.job_id);
    } catch (taskError) {
      controller.setImportState("error");
      controller.setImportError(
        taskError instanceof Error
          ? taskError.message
          : "Research agent 导入失败"
      );
    }
  }

  return { handleImportSelected };
}

export type ResearchImportFlow = ReturnType<typeof useResearchImportFlow>;

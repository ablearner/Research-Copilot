"use client";

import {
  importResearchTodo,
  rerunResearchTodoSearch,
  updateResearchTodo,
} from "@/lib/research-api";
import { toTaskSnapshot } from "./shared";
import type { ResearchConversationSession } from "./session";
import type { ResearchControllerState } from "./state";
import type { ResearchRuntime } from "./runtime";

export function useResearchTodoFlow(
  controller: ResearchControllerState,
  session: ResearchConversationSession,
  runtime: ResearchRuntime
) {
  function isTodoBusy(todoId: string, action: string) {
    return controller.todoActionKey === `${todoId}:${action}`;
  }

  async function handleTodoStatus(
    todoId: string,
    nextStatus: "open" | "done" | "dismissed"
  ) {
    if (!controller.taskResult?.task.task_id) return;
    controller.setTodoActionKey(`${todoId}:${nextStatus}`);
    controller.setTodoActionError(null);
    controller.setTodoActionNotice(null);
    try {
      const response = await updateResearchTodo(
        controller.taskResult.task.task_id,
        todoId,
        nextStatus
      );
      runtime.applyTaskSnapshot(response);
      controller.setTodoActionNotice(
        nextStatus === "done"
          ? "TODO 已标记完成。"
          : nextStatus === "dismissed"
            ? "TODO 已关闭。"
            : "TODO 已恢复为打开状态。"
      );
    } catch (taskError) {
      controller.setTodoActionError(
        taskError instanceof Error
          ? taskError.message
          : "TODO 状态更新失败"
      );
    } finally {
      controller.setTodoActionKey(null);
    }
  }

  async function handleTodoSearch(todoId: string) {
    if (!controller.taskResult?.task.task_id) return;
    controller.setTodoActionKey(`${todoId}:search`);
    controller.setTodoActionError(null);
    controller.setTodoActionNotice(null);
    try {
      const activeConversationId = await session.ensureConversationId();
      const response = await rerunResearchTodoSearch(
        controller.taskResult.task.task_id,
        todoId,
        { max_papers: 5, conversation_id: activeConversationId }
      );
      runtime.applyTaskSnapshot(
        toTaskSnapshot({
          task: response.task,
          papers: response.papers,
          report: response.report,
          warnings: response.warnings,
        })
      );
      controller.setTodoActionNotice(
        `已从 TODO 重新检索，当前候选论文 ${response.papers.length} 篇。`
      );
      await session.loadConversationById(activeConversationId, {
        preserveAgentResult: true,
      });
    } catch (taskError) {
      controller.setTodoActionError(
        taskError instanceof Error
          ? taskError.message
          : "TODO 重新检索失败"
      );
    } finally {
      controller.setTodoActionKey(null);
    }
  }

  async function handleTodoImport(todoId: string) {
    if (!controller.taskResult?.task.task_id) return;
    controller.setTodoActionKey(`${todoId}:import`);
    controller.setTodoActionError(null);
    controller.setTodoActionNotice(null);
    try {
      const activeConversationId = await session.ensureConversationId();
      const response = await importResearchTodo(
        controller.taskResult.task.task_id,
        todoId,
        { max_papers: 3, conversation_id: activeConversationId }
      );
      runtime.applyTaskSnapshot(
        toTaskSnapshot({
          task: response.task,
          papers: response.papers,
          report: response.report,
          warnings: response.warnings,
        })
      );
      if (response.import_result) {
        controller.setImportResult(response.import_result);
        controller.setAskResult(null);
        controller.setTodoActionNotice(
          `已从 TODO 触发补充导入：imported=${response.import_result.imported_count} · skipped=${response.import_result.skipped_count} · failed=${response.import_result.failed_count}`
        );
      } else {
        controller.setTodoActionNotice("TODO 补充导入已执行。");
      }
      await session.loadConversationById(activeConversationId, {
        preserveAgentResult: true,
      });
    } catch (taskError) {
      controller.setTodoActionError(
        taskError instanceof Error
          ? taskError.message
          : "TODO 补充导入失败"
      );
    } finally {
      controller.setTodoActionKey(null);
    }
  }

  return {
    isTodoBusy,
    handleTodoStatus,
    handleTodoSearch,
    handleTodoImport,
  };
}

export type ResearchTodoFlow = ReturnType<typeof useResearchTodoFlow>;

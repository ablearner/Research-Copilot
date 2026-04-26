"use client";

import { AlertTriangle, CheckCheck } from "lucide-react";
import type { ResearchTodoItem } from "@/lib/types";
import { TodoWorkbenchCard } from "@/components/ResearchWorkspaceCards";

export function ResearchTodoWorkbenchSection({
  todoItems,
  todoActionError,
  todoActionNotice,
  onToggleDone,
  onToggleDismissed,
  onSearch,
  onImport,
  isBusy,
}: {
  todoItems: ResearchTodoItem[];
  todoActionError: string | null;
  todoActionNotice: string | null;
  onToggleDone: (todoId: string, status: ResearchTodoItem["status"]) => void;
  onToggleDismissed: (todoId: string, status: ResearchTodoItem["status"]) => void;
  onSearch: (todoId: string) => void;
  onImport: (todoId: string) => void;
  isBusy: (todoId: string, action: string) => boolean;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold text-gray-800">待办</div>
          <span className="text-[11px] text-gray-400">{todoItems.length}</span>
        </div>
        {todoActionError && (
          <div className="mt-2 flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] leading-5 text-red-700">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-400" />
            {todoActionError}
          </div>
        )}
        {todoActionNotice && (
          <div className="mt-2 flex items-start gap-2 rounded-lg bg-emerald-50 px-3 py-2 text-[12px] leading-5 text-emerald-700">
            <CheckCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
            {todoActionNotice}
          </div>
        )}
        <div className="mt-4 flex max-h-[36vh] flex-col gap-3 overflow-y-auto pr-1 chart-chat-scroll">
          {todoItems.length ? (
            todoItems.map((item) => (
              <TodoWorkbenchCard
                key={item.todo_id}
                item={item}
                onToggleDone={() => onToggleDone(item.todo_id, item.status)}
                onToggleDismissed={() =>
                  onToggleDismissed(item.todo_id, item.status)
                }
                onSearch={() => onSearch(item.todo_id)}
                onImport={() => onImport(item.todo_id)}
                isBusy={(action) => isBusy(item.todo_id, action)}
              />
            ))
          ) : (
            <div className="rounded-lg border border-dashed border-gray-200 px-3 py-6 text-center text-[12px] text-gray-400">
              完成研究问答后系统会生成待办。
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

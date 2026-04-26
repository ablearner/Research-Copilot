"use client";

import type { ReactNode } from "react";
import type {
  ResearchConversation,
  ResearchSource,
} from "@/lib/types";

function formatConversationTime(value: string | null | undefined): string {
  if (!value) return "刚刚";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "刚刚";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SourceToggle({
  checked,
  label,
  hint,
  icon,
  onToggle,
}: {
  checked: boolean;
  label: string;
  hint: string;
  icon: ReactNode;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={checked}
      className={`group flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2.5 transition-colors ${
        checked
          ? "border-blue-200 bg-blue-50/50"
          : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50"
      }`}
      onClick={onToggle}
    >
      <div
        className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded transition-colors ${
          checked
            ? "bg-blue-600 text-white"
            : "border border-gray-300 bg-white text-transparent group-hover:border-gray-400"
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-3 w-3"
        >
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className={`transition-colors ${
              checked ? "text-blue-600" : "text-gray-400 group-hover:text-gray-500"
            }`}
          >
            {icon}
          </span>
          <span
            className={`text-[13px] font-medium ${
              checked ? "text-gray-900" : "text-gray-700"
            }`}
          >
            {label}
          </span>
        </div>
        <div
          className={`mt-1 text-[12px] leading-5 ${
            checked ? "text-gray-500" : "text-gray-400"
          }`}
        >
          {hint}
        </div>
      </div>
    </button>
  );
}

export function SidebarStatsCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon?: ReactNode;
}) {
  return (
    <div className="flex w-full flex-col items-center rounded-lg border border-gray-200 bg-white px-2 py-2.5 text-center">
      {icon && <div className="mb-1 text-gray-400">{icon}</div>}
      <div className="text-[10px] font-medium uppercase tracking-wider text-gray-400">
        {label}
      </div>
      <div className="mt-0.5 text-[16px] font-semibold text-gray-900">{value}</div>
    </div>
  );
}

export function ResearchConversationsSection({
  model,
  actions,
}: {
  model: {
    conversations: ResearchConversation[];
    conversationId: string | null;
    currentConversation: ResearchConversation | null;
    conversationMessageCount: number;
  };
  actions: {
    onCreateConversation: () => void;
    onDeleteConversation: () => void;
    onSelectConversation: (conversationId: string) => void;
  };
}) {
  const {
    conversations,
    conversationId,
    currentConversation,
    conversationMessageCount,
  } = model;
  const {
    onCreateConversation,
    onDeleteConversation,
    onSelectConversation,
  } = actions;
  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold text-gray-800">会话</div>
          <span className="text-[11px] text-gray-400">{conversations.length}</span>
        </div>
        <div className="mt-2 flex gap-2">
          <button
            type="button"
            onClick={onCreateConversation}
            className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-[12px] font-medium text-gray-700 hover:bg-gray-50"
          >
            新建
          </button>
          <button
            type="button"
            onClick={onDeleteConversation}
            disabled={!conversationId || conversations.length <= 1}
            className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-[12px] text-gray-500 hover:border-red-200 hover:text-red-500 disabled:opacity-40"
          >
            删除
          </button>
        </div>
        <div className="mt-3 flex max-h-[28vh] flex-col gap-1.5 overflow-y-auto pr-1 chat-scroll">
          {conversations.length ? (
            conversations.map((item) => {
              const active = item.conversation_id === conversationId;
              return (
                <button
                  key={item.conversation_id}
                  type="button"
                  onClick={() => onSelectConversation(item.conversation_id)}
                  className={`rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    active
                      ? "border-blue-200 bg-blue-50/50"
                      : "border-gray-200 bg-white hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium text-gray-900">
                        {item.title}
                      </div>
                      <div className="mt-0.5 text-[11px] text-gray-500">
                        {formatConversationTime(item.updated_at)} · {item.message_count} 条记录
                      </div>
                    </div>
                    {active && (
                      <span className="rounded-full bg-gray-800 px-2 py-0.5 text-[10px] font-bold text-white">
                        当前
                      </span>
                    )}
                  </div>
                  {item.last_message_preview && (
                    <div className="mt-1.5 line-clamp-2 text-[12px] leading-5 text-gray-400">
                      {item.last_message_preview}
                    </div>
                  )}
                </button>
              );
            })
          ) : (
            <div className="rounded-lg border border-dashed border-gray-200 px-3 py-4 text-center text-[12px] text-gray-400">
              暂无会话
            </div>
          )}
        </div>
        {currentConversation && (
          <div className="mt-2 rounded-lg bg-gray-50 px-3 py-2 text-[12px] leading-5 text-gray-600">
            当前会话：{currentConversation.title} · 消息 {conversationMessageCount}
          </div>
        )}
      </div>
    </div>
  );
}

type ResearchSetupSourceOption = {
  value: ResearchSource;
  label: string;
  hint: string;
  icon: ReactNode;
};

export function ResearchSetupSection({
  model,
  actions,
}: {
  model: {
    sourceOptions: ResearchSetupSourceOption[];
    sources: ResearchSource[];
    daysBack: number;
    maxPapers: number;
  };
  actions: {
    onToggleSource: (source: ResearchSource) => void;
    onDaysBackChange: (value: number) => void;
    onMaxPapersChange: (value: number) => void;
  };
}) {
  const {
    sourceOptions,
    sources,
    daysBack,
    maxPapers,
  } = model;
  const {
    onToggleSource,
    onDaysBackChange,
    onMaxPapersChange,
  } = actions;
  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="p-3">
        <div className="text-[13px] font-semibold text-gray-800">
          检索设置
        </div>
        <div className="mt-3 grid gap-2.5">
          {sourceOptions.map((option) => (
            <SourceToggle
              key={option.value}
              checked={sources.includes(option.value)}
              label={option.label}
              hint={option.hint}
              icon={option.icon}
              onToggle={() => onToggleSource(option.value)}
            />
          ))}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <label className="group rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 transition-colors focus-within:border-gray-400 focus-within:bg-white">
            <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-gray-400 group-focus-within:text-gray-600">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3 w-3"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 6v6l4 2" />
              </svg>
              Days
            </div>
            <input
              type="number"
              min={1}
              max={3650}
              value={daysBack}
              onChange={(event) =>
                onDaysBackChange(Number(event.target.value) || 180)
              }
              className="mt-1 w-full border-0 bg-transparent p-0 text-[14px] font-medium text-gray-900 outline-none"
            />
          </label>
          <label className="group rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 transition-colors focus-within:border-gray-400 focus-within:bg-white">
            <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-gray-400 group-focus-within:text-gray-600">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3 w-3"
              >
                <path d="M4 12h16" />
                <path d="M12 4v16" />
              </svg>
              Max Papers
            </div>
            <input
              type="number"
              min={1}
              max={100}
              value={maxPapers}
              onChange={(event) =>
                onMaxPapersChange(Number(event.target.value) || 12)
              }
              className="mt-1 w-full border-0 bg-transparent p-0 text-[14px] font-medium text-gray-900 outline-none"
            />
          </label>
        </div>
      </div>
    </div>
  );
}

"use client";

import {
  ConversationMessageBubble,
  ThreadBubble,
} from "@/components/ResearchConversationMessage";
import type {
  ResearchConversation,
  ResearchMessage,
} from "@/lib/types";
import { buildListKey } from "@/lib/value-coercion";

export function ResearchThreadPreamble({
  model,
  actions,
}: {
  model: {
    currentConversation: ResearchConversation | null;
    shouldReplayConversation: boolean;
    conversationMessageCount: number;
    conversationMessages: ResearchMessage[];
    selectedPaperIds: string[];
    recommendedPaperIds: Set<string>;
    mustReadPaperIds: Set<string>;
    paperTitleById: Map<string, string>;
    hasWorkspace: boolean;
    suggestions: string[];
    error: string | null;
    replayHasError: boolean;
    activeWarnings: string[];
    replayHasWarning: boolean;
  };
  actions: {
    onTogglePaperSelection: (paperId: string) => void;
    onSelectSuggestion: (suggestion: string) => void;
  };
}) {
  const {
    currentConversation,
    shouldReplayConversation,
    conversationMessageCount,
    conversationMessages,
    selectedPaperIds,
    recommendedPaperIds,
    mustReadPaperIds,
    paperTitleById,
    hasWorkspace,
    suggestions,
    error,
    replayHasError,
    activeWarnings,
    replayHasWarning,
  } = model;
  const {
    onTogglePaperSelection,
    onSelectSuggestion,
  } = actions;
  return (
    <>
      {/* Welcome — only when no workspace yet */}
      {!hasWorkspace && !shouldReplayConversation && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="text-2xl font-semibold text-gray-900">Research-Copilot</div>
          <p className="mt-2 max-w-md text-[14px] leading-6 text-gray-500">
            输入研究主题，助手会自主检索、筛选论文并生成综述。导入论文后可进行 grounded QA。
          </p>
          <div className="mt-8 grid w-full max-w-xl gap-2 sm:grid-cols-3">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => onSelectSuggestion(suggestion)}
                className="rounded-xl border border-gray-200 bg-white px-3 py-3 text-left text-[13px] leading-5 text-gray-600 transition-colors hover:border-gray-300 hover:bg-gray-50"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Restored conversation */}
      {currentConversation && shouldReplayConversation && (
        <>
          <ThreadBubble role="assistant" title="会话已恢复">
            <p>已从 {currentConversation.title} 恢复，共 {conversationMessageCount} 条记录。</p>
          </ThreadBubble>
          {conversationMessages.map((message) => (
            <ConversationMessageBubble
              key={message.message_id}
              message={message}
              selectedPaperIds={selectedPaperIds}
              recommendedPaperIds={recommendedPaperIds}
              mustReadPaperIds={mustReadPaperIds}
              paperTitleById={paperTitleById}
              onTogglePaperSelection={onTogglePaperSelection}
            />
          ))}
        </>
      )}

      {error && (!shouldReplayConversation || !replayHasError) && (
        <ThreadBubble role="system" title="请求失败">
          <p>{error}</p>
        </ThreadBubble>
      )}

      {activeWarnings.length > 0 &&
        (!shouldReplayConversation || !replayHasWarning) && (
          <ThreadBubble role="system" title="检索告警">
            <ul>
              {activeWarnings.map((warning, index) => (
                <li
                  key={buildListKey(
                    "active-warning",
                    warning,
                    index
                  )}
                >
                  {warning}
                </li>
              ))}
            </ul>
          </ThreadBubble>
        )}
    </>
  );
}

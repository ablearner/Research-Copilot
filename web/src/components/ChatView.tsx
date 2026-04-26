import { useState, useEffect, useRef, useCallback } from 'react';
import { InputBar } from './InputBar';
import { MessageBubble } from './MessageBubble';
import type {
  ChatMessage,
  ResearchConversation,
  ResearchMessage,
  ResearchAgentRunResponse,
} from '../types';
import { getConversation, sendMessage } from '../api';

interface ChatViewProps {
  conversationId: string;
  pendingMessage: string | null;
  onPendingMessageConsumed: () => void;
  onConversationUpdated: (conv: ResearchConversation) => void;
}

// User-facing message kinds (always shown as main content)
const PRIMARY_KINDS = new Set([
  'answer',
  'report',
  'import_result',
  'welcome',
  'warning',
  'error',
]);

// Notice titles that carry actual user-facing results (not internal metadata)
const USER_FACING_NOTICE_TITLES = new Set([
  '论文分析结果',
  '图表理解结果',
  '文档理解结果',
  'Zotero 同步结果',
  '上下文压缩摘要',
  '长期兴趣论文推荐',
]);

// Internal notice titles (shown collapsed)
const INTERNAL_NOTICE_TITLES = new Set([
  'Manager 决策轨迹',
  'Agent 决策轨迹',
  'Research Workspace',
]);

function isPrimaryMessage(m: ResearchMessage): boolean {
  if (m.role === 'user') return false;
  if (PRIMARY_KINDS.has(m.kind)) return true;
  // Notices with user-facing titles are primary content
  if (m.kind === 'notice' && USER_FACING_NOTICE_TITLES.has(m.title)) return true;
  return false;
}

function isInternalNotice(m: ResearchMessage): boolean {
  if (m.kind !== 'notice') return false;
  if (INTERNAL_NOTICE_TITLES.has(m.title)) return true;
  // If not in user-facing set, treat as internal
  if (!USER_FACING_NOTICE_TITLES.has(m.title)) return true;
  return false;
}

function convertBackendMessages(messages: ResearchMessage[]): ChatMessage[] {
  const result: ChatMessage[] = [];
  let assistantGroup: ResearchMessage[] = [];

  const flushAssistant = () => {
    if (assistantGroup.length === 0) return;
    // Only show primary messages as main content
    const primaryContent = assistantGroup
      .filter(isPrimaryMessage)
      .map((m) => m.content || m.title)
      .filter(Boolean)
      .join('\n\n');
    const noticeItems = assistantGroup
      .filter(isInternalNotice)
      .map((m) => {
        const label = m.title || 'Notice';
        const meta = m.meta ? ` (${m.meta})` : '';
        return `${label}${meta}`;
      })
      .filter(Boolean);
    // Fallback: if no primary messages, show all
    const fallbackContent =
      primaryContent ||
      assistantGroup
        .map((m) => m.content || m.title)
        .filter(Boolean)
        .join('\n\n') ||
      '';

    // Extract papers from candidates messages
    const papers: import('../types').PaperCandidate[] = [];
    for (const m of assistantGroup) {
      if (m.kind === 'candidates' && m.payload?.papers) {
        for (const p of m.payload.papers as import('../types').PaperCandidate[]) {
          papers.push(p);
        }
      }
    }

    result.push({
      id: assistantGroup[0].message_id,
      role: 'assistant',
      content: fallbackContent,
      timestamp: assistantGroup[0].created_at,
      papers: papers.length > 0 ? papers : undefined,
      notices: noticeItems.length > 0 ? noticeItems : undefined,
      backendMessages: assistantGroup,
    });
    assistantGroup = [];
  };

  for (const msg of messages) {
    if (msg.role === 'user') {
      flushAssistant();
      result.push({
        id: msg.message_id,
        role: 'user',
        content: msg.content || msg.title,
        timestamp: msg.created_at,
      });
    } else {
      assistantGroup.push(msg);
    }
  }
  flushAssistant();
  return result;
}

function buildAssistantContent(response: ResearchAgentRunResponse): string {
  // 1. Only show user-facing messages as main content
  const primary = response.messages
    .filter(isPrimaryMessage)
    .map((m) => m.content || m.title)
    .filter(Boolean)
    .join('\n\n');

  if (primary) {
    return primary;
  }
  // 2. Fallback to QA answer
  if (response.qa?.answer) {
    return response.qa.answer;
  }
  // 3. Fallback to report
  if (response.report?.markdown) {
    return response.report.markdown;
  }
  // 4. If nothing matched, try ALL non-user messages as last resort
  const allText = response.messages
    .filter((m) => m.role !== 'user')
    .map((m) => m.content || m.title)
    .filter(Boolean)
    .join('\n\n');
  if (allText) {
    return allText;
  }
  if (response.status === 'failed') {
    return 'The research agent encountered an error processing your request.';
  }
  return 'Task completed.';
}

function extractNotices(response: ResearchAgentRunResponse): string[] {
  return response.messages
    .filter(isInternalNotice)
    .map((m) => {
      const label = m.title || 'Notice';
      const meta = m.meta ? ` (${m.meta})` : '';
      return `${label}${meta}`;
    })
    .filter(Boolean);
}

export function ChatView({
  conversationId,
  pendingMessage,
  onPendingMessageConsumed,
  onConversationUpdated,
}: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [taskId, setTaskId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pendingConsumed = useRef(false);

  // Load conversation history
  useEffect(() => {
    let cancelled = false;
    pendingConsumed.current = false;
    setMessages([]);
    setIsLoadingHistory(true);

    getConversation(conversationId)
      .then((data) => {
        if (cancelled) return;
        setMessages(convertBackendMessages(data.messages));
        // Restore task_id from conversation metadata
        if (data.conversation.task_id) {
          setTaskId(data.conversation.task_id);
        }
        setIsLoadingHistory(false);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('Failed to load conversation:', err);
        setIsLoadingHistory(false);
      });

    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  // Handle pending message from WelcomeScreen
  useEffect(() => {
    if (pendingMessage && !isLoadingHistory && !pendingConsumed.current) {
      pendingConsumed.current = true;
      handleSend(pendingMessage);
      onPendingMessageConsumed();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingMessage, isLoadingHistory]);

  // Auto-scroll on new messages
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const response = await sendMessage({
          message: text,
          conversationId,
          taskId: taskId || undefined,
        });

        const notices = extractNotices(response);
        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: buildAssistantContent(response),
          timestamp: new Date().toISOString(),
          papers:
            response.papers.length > 0 ? response.papers : undefined,
          report: response.report || undefined,
          qa: response.qa || undefined,
          trace:
            response.trace.length > 0 ? response.trace : undefined,
          warnings:
            response.warnings.length > 0
              ? response.warnings
              : undefined,
          notices: notices.length > 0 ? notices : undefined,
          nextActions:
            response.next_actions.length > 0
              ? response.next_actions
              : undefined,
          workspace: response.workspace,
          backendMessages: response.messages,
          status: response.status,
        };

        setMessages((prev) => [...prev, assistantMsg]);

        // Track task_id for subsequent messages
        if (response.task?.task_id) {
          setTaskId(response.task.task_id);
        }

        onConversationUpdated({
          conversation_id: conversationId,
          title: response.task?.topic || text.slice(0, 60),
          created_at: '',
          updated_at: new Date().toISOString(),
          task_id: response.task?.task_id || null,
          message_count: 0,
          last_message_preview: assistantMsg.content.slice(0, 100),
        });
      } catch (err) {
        const errorMsg: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Failed to get response: ${err instanceof Error ? err.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          isError: true,
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setIsLoading(false);
      }
    },
    [conversationId, taskId, onConversationUpdated],
  );

  return (
    <div className="h-full flex flex-col">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {isLoadingHistory ? (
            <div className="text-center py-12 text-ink-300 text-sm">
              Loading conversation…
            </div>
          ) : messages.length === 0 && !isLoading ? (
            <div className="text-center py-12 text-ink-300 text-sm font-display italic">
              Start a conversation…
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {isLoading && <TypingIndicator />}
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <InputBar onSend={handleSend} isLoading={isLoading} />
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 animate-fade-in py-2">
      <div className="flex gap-1.5">
        <div className="w-2 h-2 rounded-full bg-accent-400 typing-dot" />
        <div className="w-2 h-2 rounded-full bg-accent-400 typing-dot" />
        <div className="w-2 h-2 rounded-full bg-accent-400 typing-dot" />
      </div>
      <span className="text-xs text-ink-300">Thinking…</span>
    </div>
  );
}

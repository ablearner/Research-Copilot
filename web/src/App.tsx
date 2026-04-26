import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { WelcomeScreen } from './components/WelcomeScreen';
import {
  fetchConversations,
  createConversation,
  deleteConversation,
} from './api';
import type { ResearchConversation } from './types';
import { Menu } from 'lucide-react';

export default function App() {
  const [conversations, setConversations] = useState<ResearchConversation[]>(
    [],
  );
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);

  /* ── load conversation list ── */
  const loadConversations = useCallback(async () => {
    try {
      const data = await fetchConversations();
      setConversations(data);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  /* ── actions ── */
  const handleNewChat = useCallback(async () => {
    try {
      const res = await createConversation();
      setConversations((prev) => [res.conversation, ...prev]);
      setActiveId(res.conversation.conversation_id);
      setPendingMessage(null);
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  }, []);

  const handleSelect = useCallback((id: string) => {
    setActiveId(id);
    setPendingMessage(null);
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        setConversations((prev) =>
          prev.filter((c) => c.conversation_id !== id),
        );
        if (activeId === id) setActiveId(null);
      } catch (err) {
        console.error('Failed to delete conversation:', err);
      }
    },
    [activeId],
  );

  const handleConversationUpdated = useCallback(
    (updated: ResearchConversation) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.conversation_id === updated.conversation_id ? updated : c,
        ),
      );
    },
    [],
  );

  /* ── start chat from welcome screen ── */
  const handleStartChat = useCallback(
    async (message: string) => {
      try {
        const res = await createConversation();
        setConversations((prev) => [res.conversation, ...prev]);
        setActiveId(res.conversation.conversation_id);
        setPendingMessage(message);
      } catch (err) {
        console.error('Failed to start chat:', err);
      }
    },
    [],
  );

  const handlePendingConsumed = useCallback(() => {
    setPendingMessage(null);
  }, []);

  return (
    <div className="flex h-full bg-paper-50">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? 'w-72' : 'w-0'
        } flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden`}
      >
        <Sidebar
          conversations={conversations}
          activeConversationId={activeId}
          onNewChat={handleNewChat}
          onSelectConversation={handleSelect}
          onDeleteConversation={handleDelete}
          isLoading={loading}
        />
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-12 flex items-center px-4 border-b border-paper-200 bg-paper-50/80 backdrop-blur-sm flex-shrink-0 z-10">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="p-1.5 rounded-lg hover:bg-paper-200 transition-colors text-ink-400"
            aria-label="Toggle sidebar"
          >
            <Menu size={18} />
          </button>
          <span className="ml-3 font-display text-lg font-semibold text-ink-700 tracking-tight select-none">
            Research Copilot
          </span>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeId ? (
            <ChatView
              key={activeId}
              conversationId={activeId}
              pendingMessage={pendingMessage}
              onPendingMessageConsumed={handlePendingConsumed}
              onConversationUpdated={handleConversationUpdated}
            />
          ) : (
            <WelcomeScreen onStartChat={handleStartChat} />
          )}
        </div>
      </div>
    </div>
  );
}

import { Plus, MessageSquare, Trash2 } from 'lucide-react';
import type { ResearchConversation } from '../types';

interface SidebarProps {
  conversations: ResearchConversation[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  isLoading: boolean;
}

export function Sidebar({
  conversations,
  activeConversationId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  isLoading,
}: SidebarProps) {
  return (
    <div className="h-full flex flex-col bg-paper-100 border-r border-paper-200">
      {/* New chat */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-paper-300 hover:bg-paper-200 transition-colors text-sm font-medium text-ink-600"
        >
          <Plus size={16} />
          <span>New Research</span>
        </button>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {isLoading ? (
          <div className="px-3 py-8 text-center text-ink-300 text-sm">
            Loading…
          </div>
        ) : conversations.length === 0 ? (
          <div className="px-3 py-8 text-center text-ink-300 text-sm">
            No conversations yet
          </div>
        ) : (
          <div className="space-y-0.5">
            {conversations.map((conv) => (
              <div
                key={conv.conversation_id}
                className={`group flex items-center rounded-lg cursor-pointer transition-colors ${
                  activeConversationId === conv.conversation_id
                    ? 'bg-paper-200'
                    : 'hover:bg-paper-200/60'
                }`}
              >
                <button
                  className="flex-1 flex items-center gap-2.5 px-3 py-2.5 min-w-0 text-left"
                  onClick={() => onSelectConversation(conv.conversation_id)}
                >
                  <MessageSquare
                    size={14}
                    className="flex-shrink-0 text-ink-300"
                  />
                  <span className="truncate text-sm text-ink-600">
                    {conv.title || 'Untitled'}
                  </span>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.conversation_id);
                  }}
                  className="p-1.5 mr-1 rounded opacity-0 group-hover:opacity-100 hover:bg-paper-300 transition-all text-ink-300 hover:text-red-500"
                  title="Delete"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-paper-200">
        <div className="text-xs text-ink-300 text-center select-none">
          Research Copilot v0.1
        </div>
      </div>
    </div>
  );
}

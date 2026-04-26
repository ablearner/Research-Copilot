import type {
  ResearchAgentRunResponse,
  ResearchConversation,
  ResearchConversationResponse,
} from './types';

const API_BASE = 'http://127.0.0.1:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function fetchConversations(): Promise<ResearchConversation[]> {
  return request('/research/conversations');
}

export async function createConversation(
  title?: string,
): Promise<ResearchConversationResponse> {
  return request('/research/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: title || undefined }),
  });
}

export async function getConversation(
  id: string,
): Promise<ResearchConversationResponse> {
  return request(`/research/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${API_BASE}/research/conversations/${id}`, {
    method: 'DELETE',
  });
}

export async function renameConversation(
  id: string,
  title: string,
): Promise<ResearchConversationResponse> {
  return request(`/research/conversations/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
}

export interface SendMessageOptions {
  message: string;
  conversationId?: string;
  mode?: string;
  taskId?: string;
  selectedPaperIds?: string[];
  selectedDocumentIds?: string[];
  chartImagePath?: string;
  pageNumber?: number;
  chartId?: string;
  documentId?: string;
}

export async function sendMessage(
  opts: SendMessageOptions,
): Promise<ResearchAgentRunResponse> {
  return request('/research/agent', {
    method: 'POST',
    body: JSON.stringify({
      message: opts.message,
      mode: opts.mode || 'auto',
      conversation_id: opts.conversationId || undefined,
      task_id: opts.taskId || undefined,
      selected_paper_ids: opts.selectedPaperIds?.length ? opts.selectedPaperIds : undefined,
      selected_document_ids: opts.selectedDocumentIds?.length ? opts.selectedDocumentIds : undefined,
      chart_image_path: opts.chartImagePath || undefined,
      page_number: opts.pageNumber || undefined,
      chart_id: opts.chartId || undefined,
      document_id: opts.documentId || undefined,
    }),
  });
}

export async function checkHealth(): Promise<{ status: string }> {
  return request('/health');
}

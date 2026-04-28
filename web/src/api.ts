import type {
  ResearchAgentRunResponse,
  ResearchConversation,
  ResearchConversationResponse,
} from './types';

// Direct connection to FastAPI backend (research tasks can take 60-120s, Next.js proxy may timeout)
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
  const res = await fetch(`${API_BASE}/research/conversations/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${res.status}: ${body}`);
  }
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

export interface SSEProgressEvent {
  stage: string;
  node: string;
  status: string;
  summary: string;
  updated_at?: string;
  step_index?: number;
}

export async function sendMessageStream(
  opts: SendMessageOptions,
  onProgress: (event: SSEProgressEvent) => void,
): Promise<ResearchAgentRunResponse> {
  const res = await fetch(`${API_BASE}/research/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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

  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${res.status}: ${body}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let finalResult: ResearchAgentRunResponse | null = null;
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      if (!part.trim()) continue;
      const lines = part.split('\n');
      let eventType = '';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7);
        else if (line.startsWith('data: ')) data = line.slice(6);
      }
      if (eventType === 'progress' && data) {
        try { onProgress(JSON.parse(data)); } catch { /* ignore parse errors */ }
      } else if (eventType === 'complete' && data) {
        try { finalResult = JSON.parse(data); } catch { /* ignore */ }
      } else if (eventType === 'error' && data) {
        const err = JSON.parse(data);
        throw new Error(err.error || 'Stream error');
      }
    }
  }

  if (!finalResult) throw new Error('Stream ended without result');
  return finalResult;
}

export async function uploadDocument(
  file: File,
): Promise<{ document_id: string; filename: string; status: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`Upload failed: ${body}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  return request('/health');
}

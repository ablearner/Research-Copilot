import type {
  AnalyzeResearchPaperFigureRequest,
  AnalyzeResearchPaperFigureResponse,
  CreateResearchConversationRequest,
  CreateResearchTaskRequest,
  ImportPapersRequest,
  ImportPapersResponse,
  ResearchConversation,
  ResearchConversationResponse,
  ResearchJob,
  ResearchAgentRunRequest,
  ResearchAgentRunResponse,
  ResearchPaperFigureListResponse,
  ResearchTodoActionRequest,
  ResearchTodoActionResponse,
  ResearchTaskResponse,
  SearchPapersRequest,
  SearchPapersResponse
} from "./types";
import { normalizeResearchAgentRunResponse } from "./unified-runtime";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend";
const DEFAULT_API_KEY = process.env.NEXT_PUBLIC_API_KEY;
const DEFAULT_RESEARCH_TIMEOUT_MS = 180_000;
const DEFAULT_AGENT_TIMEOUT_MS = 300_000;
const IMPORT_TIMEOUT_FLOOR_MS = 600_000;
const IMPORT_TIMEOUT_CAP_MS = 1_200_000;

function normalizeUnknownServerError(response: Response, text: string): string | null {
  const trimmed = text.trim();
  const lower = trimmed.toLowerCase();
  const looksLikeHtml = lower.startsWith("<!doctype html") || lower.startsWith("<html");
  const looksGeneric500 = lower === "internal server error" || (response.status >= 500 && looksLikeHtml);
  if (!looksGeneric500) return null;
  return "Backend request failed. Confirm the FastAPI service is running and check the backend terminal logs for the actual error.";
}

async function readError(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) return `${response.status} ${response.statusText}`;
  const normalized = normalizeUnknownServerError(response, text);
  if (normalized) return normalized;
  try {
    const payload = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown };
    const detail = payload.detail ?? payload.error ?? payload.message;
    return typeof detail === "string" ? detail : JSON.stringify(detail ?? payload);
  } catch {
    return text;
  }
}

async function requestJson<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs = DEFAULT_RESEARCH_TIMEOUT_MS, ...requestInit } = init ?? {};
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...requestInit,
      signal: controller.signal,
      headers: {
        ...(requestInit?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(DEFAULT_API_KEY ? { "X-API-Key": DEFAULT_API_KEY } : {}),
        ...requestInit?.headers
      }
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Research request timed out in the browser. FastAPI may still be processing the request in the background; check the backend terminal and refresh the research task shortly.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function resolveResearchAgentTimeoutMs(payload: ResearchAgentRunRequest): number {
  if (payload.mode === "import") {
    const selectedPaperCount = Math.max(
      payload.selected_paper_ids?.length ?? 0,
      payload.import_top_k ?? 0,
      1,
    );
    return Math.min(
      IMPORT_TIMEOUT_CAP_MS,
      Math.max(IMPORT_TIMEOUT_FLOOR_MS, selectedPaperCount * 240_000),
    );
  }
  return DEFAULT_AGENT_TIMEOUT_MS;
}

export function searchPapers(payload: SearchPapersRequest): Promise<SearchPapersResponse> {
  return requestJson<SearchPapersResponse>("/research/papers/search", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 180_000
  });
}

export function runResearchAgent(payload: ResearchAgentRunRequest): Promise<ResearchAgentRunResponse> {
  return requestJson<ResearchAgentRunResponse>("/research/agent", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: resolveResearchAgentTimeoutMs(payload)
  }).then((response) => normalizeResearchAgentRunResponse(response));
}

export function listResearchConversations(): Promise<ResearchConversation[]> {
  return requestJson<ResearchConversation[]>("/research/conversations", {
    timeoutMs: 60_000,
  });
}

export function createResearchConversation(
  payload: CreateResearchConversationRequest
): Promise<ResearchConversationResponse> {
  return requestJson<ResearchConversationResponse>("/research/conversations", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 60_000,
  });
}

export function getResearchConversation(
  conversationId: string
): Promise<ResearchConversationResponse> {
  return requestJson<ResearchConversationResponse>(
    `/research/conversations/${conversationId}`,
    {
      timeoutMs: 60_000,
    }
  );
}

export async function deleteResearchConversation(
  conversationId: string
): Promise<void> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 60_000);
  try {
    const response = await fetch(`${API_BASE_URL}/research/conversations/${conversationId}`, {
      method: "DELETE",
      signal: controller.signal,
      headers: {
        ...(DEFAULT_API_KEY ? { "X-API-Key": DEFAULT_API_KEY } : {}),
      },
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export function createResearchTask(payload: CreateResearchTaskRequest): Promise<ResearchTaskResponse> {
  return requestJson<ResearchTaskResponse>("/research/tasks", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 180_000
  });
}

export function runResearchTask(taskId: string): Promise<ResearchTaskResponse> {
  return requestJson<ResearchTaskResponse>(`/research/tasks/${taskId}/run`, {
    method: "POST",
    timeoutMs: 180_000
  });
}

export function getResearchTask(taskId: string): Promise<ResearchTaskResponse> {
  return requestJson<ResearchTaskResponse>(`/research/tasks/${taskId}`);
}

export function importResearchPapers(payload: ImportPapersRequest): Promise<ImportPapersResponse> {
  const importTimeoutMs = IMPORT_TIMEOUT_FLOOR_MS;
  if (payload.task_id) {
    return requestJson<ImportPapersResponse>(`/research/tasks/${payload.task_id}/papers/import`, {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs: importTimeoutMs
    });
  }
  return requestJson<ImportPapersResponse>("/research/papers/import", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: importTimeoutMs
  });
}

export function startResearchImportJob(payload: ImportPapersRequest): Promise<ResearchJob> {
  const timeoutMs = 60_000;
  if (payload.task_id) {
    return requestJson<ResearchJob>(`/research/tasks/${payload.task_id}/papers/import/jobs`, {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs,
    });
  }
  return requestJson<ResearchJob>("/research/papers/import/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs,
  });
}

export function getResearchJob(jobId: string): Promise<ResearchJob> {
  return requestJson<ResearchJob>(`/research/jobs/${jobId}`, {
    timeoutMs: 60_000,
  });
}

export async function resetResearchWorkspace(): Promise<void> {
  await requestJson<{ status: string }>("/research/reset", {
    method: "POST",
    timeoutMs: 60_000,
  });
}

export function listResearchPaperFigures(
  taskId: string,
  paperId: string
): Promise<ResearchPaperFigureListResponse> {
  return requestJson<ResearchPaperFigureListResponse>(
    `/research/tasks/${taskId}/papers/${paperId}/figures`,
    { timeoutMs: 180_000 }
  );
}

export function analyzeResearchPaperFigure(
  taskId: string,
  paperId: string,
  payload: AnalyzeResearchPaperFigureRequest
): Promise<AnalyzeResearchPaperFigureResponse> {
  return requestJson<AnalyzeResearchPaperFigureResponse>(
    `/research/tasks/${taskId}/papers/${paperId}/figures/analyze`,
    {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs: 180_000,
    }
  );
}

export function updateResearchTodo(taskId: string, todoId: string, status: string): Promise<ResearchTaskResponse> {
  return requestJson<ResearchTaskResponse>(`/research/tasks/${taskId}/todos/${todoId}`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
    timeoutMs: 60_000
  });
}

export function rerunResearchTodoSearch(taskId: string, todoId: string, payload?: Partial<ResearchTodoActionRequest>): Promise<ResearchTodoActionResponse> {
  return requestJson<ResearchTodoActionResponse>(`/research/tasks/${taskId}/todos/${todoId}/search`, {
    method: "POST",
    body: JSON.stringify({
      max_papers: payload?.max_papers ?? 5,
      include_graph: payload?.include_graph ?? true,
      include_embeddings: payload?.include_embeddings ?? true,
      skill_name: payload?.skill_name ?? null,
      conversation_id: payload?.conversation_id ?? null,
    }),
    timeoutMs: 180_000
  });
}

export function importResearchTodo(taskId: string, todoId: string, payload?: Partial<ResearchTodoActionRequest>): Promise<ResearchTodoActionResponse> {
  return requestJson<ResearchTodoActionResponse>(`/research/tasks/${taskId}/todos/${todoId}/import`, {
    method: "POST",
    body: JSON.stringify({
      max_papers: payload?.max_papers ?? 3,
      include_graph: payload?.include_graph ?? true,
      include_embeddings: payload?.include_embeddings ?? true,
      skill_name: payload?.skill_name ?? null,
      conversation_id: payload?.conversation_id ?? null,
    }),
    timeoutMs: IMPORT_TIMEOUT_FLOOR_MS
  });
}

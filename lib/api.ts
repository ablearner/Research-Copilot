import type {
  AskChartRequest,
  AskChartResponse,
  AskDocumentRequest,
  AskDocumentResponse,
  AskFusedRequest,
  AskFusedResponse,
  ChartUnderstandRequest,
  ChartUnderstandResponse,
  HealthResponse,
  IndexDocumentRequest,
  IndexDocumentResponse,
  ParseDocumentRequest,
  ParseDocumentResponse,
  UploadDocumentResponse
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend";
const DEFAULT_BASE_URL = API_BASE_URL;
const DEFAULT_API_KEY = process.env.NEXT_PUBLIC_API_KEY;

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
  const { timeoutMs = 120_000, ...requestInit } = init ?? {};
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${DEFAULT_BASE_URL}${path}`, {
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
      throw new Error("Request timed out. Check the FastAPI terminal for the model call status.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function uploadDocument(file: File): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson<UploadDocumentResponse>("/documents/upload", {
    method: "POST",
    body: formData
  });
}

export function parseDocument(payload: ParseDocumentRequest): Promise<ParseDocumentResponse> {
  return requestJson<ParseDocumentResponse>("/documents/parse", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function indexDocument(payload: IndexDocumentRequest): Promise<IndexDocumentResponse> {
  return requestJson<IndexDocumentResponse>("/documents/index", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 300_000
  });
}

export function askDocument(payload: AskDocumentRequest): Promise<AskDocumentResponse> {
  return requestJson<AskDocumentResponse>("/documents/ask", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function askFused(payload: AskFusedRequest): Promise<AskFusedResponse> {
  return requestJson<AskFusedResponse>("/documents/ask/fused", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 180_000
  });
}

export function understandChart(payload: ChartUnderstandRequest): Promise<ChartUnderstandResponse> {
  return requestJson<ChartUnderstandResponse>("/charts/understand", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 180_000
  });
}

export function askChart(payload: AskChartRequest): Promise<AskChartResponse> {
  return requestJson<AskChartResponse>("/charts/ask", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 180_000
  });
}

export async function streamAskChart(
  payload: AskChartRequest,
  handlers: {
    onStart?: (sessionId: string) => void;
    onToken?: (delta: string) => void;
    onDone?: (response: AskChartResponse) => void;
    onError?: (message: string) => void;
  }
): Promise<void> {
  const response = await fetch(`${DEFAULT_BASE_URL}/charts/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(DEFAULT_API_KEY ? { "X-API-Key": DEFAULT_API_KEY } : {})
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const rawEvent of events) {
      const event = rawEvent.split("\n").find((line) => line.startsWith("event:"))?.replace("event:", "").trim();
      const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
      const data = dataLine ? JSON.parse(dataLine.replace("data:", "").trim()) as Record<string, unknown> : {};
      if (event === "start") handlers.onStart?.(String(data.session_id ?? ""));
      if (event === "token") handlers.onToken?.(String(data.delta ?? ""));
      if (event === "done") handlers.onDone?.(data as unknown as AskChartResponse);
      if (event === "error") handlers.onError?.(String(data.error ?? "Stream error"));
    }
  }
}

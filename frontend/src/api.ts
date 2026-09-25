import { SseParser } from "./sse";
import type {
  ChunkDetail,
  Citation,
  DocumentInfo,
  Health,
  SessionDetail,
  SessionSummary,
  Task,
  UploadResult,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "") + "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string = "error",
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the API. Is the backend running on port 8000?", 0, "network");
  }
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    return new ApiError(detail, response.status, body.code ?? "error");
  } catch {
    return new ApiError(`Request failed (${response.status})`, response.status);
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  listSessions: () => request<SessionSummary[]>("/sessions"),
  createSession: () => request<SessionSummary>("/sessions", json({})),
  getSession: (id: string) => request<SessionDetail>(`/sessions/${id}`),
  renameSession: (id: string, title: string) =>
    request<SessionSummary>(`/sessions/${id}`, { ...json({ title }), method: "PATCH" }),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),
  listDocuments: (sessionId: string) => request<DocumentInfo[]>(`/sessions/${sessionId}/documents`),
  uploadDocuments: (sessionId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<UploadResult>(`/sessions/${sessionId}/documents`, { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  reingestDocument: (id: string) => request<DocumentInfo>(`/documents/${id}/reingest`, { method: "POST" }),
  getChunk: (id: string) => request<ChunkDetail>(`/chunks/${id}`),
};

export interface StreamHandlers {
  onMeta?: (meta: { session_id: string; user_message_id: string; title: string }) => void;
  onSources?: (payload: { sources: unknown[] }) => void;
  onToken: (text: string) => void;
  onCitations: (payload: { answer: string; citations: Citation[] }) => void;
  onDone: (payload: { message_id: string | null }) => void;
  onError: (message: string) => void;
}

/** POST a chat turn and dispatch the Server-Sent Events as they arrive. */
export async function streamChat(
  sessionId: string,
  body: { message: string; task: Task; document_ids?: string[] },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/sessions/${sessionId}/chat`, { ...json(body), signal });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    handlers.onError("Cannot reach the API. Is the backend running?");
    return;
  }
  if (!response.ok || !response.body) {
    handlers.onError((await toApiError(response)).message);
    return;
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  let finished = false;
  try {
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const { event, data } of parser.push(value)) {
        const payload = JSON.parse(data);
        switch (event) {
          case "meta":
            handlers.onMeta?.(payload);
            break;
          case "sources":
            handlers.onSources?.(payload);
            break;
          case "token":
            handlers.onToken(payload.text);
            break;
          case "citations":
            handlers.onCitations(payload);
            break;
          case "done":
            handlers.onDone(payload);
            finished = true;
            break;
          case "error":
            handlers.onError(payload.detail ? `${payload.message} (${payload.detail})` : payload.message);
            finished = true;
            break;
        }
      }
    }
  } catch (error) {
    if ((error as Error).name !== "AbortError") handlers.onError("The connection was interrupted.");
  }
}

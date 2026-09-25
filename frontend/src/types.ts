export type Task = "qa" | "summarize" | "eli5" | "key_points" | "glossary" | "compare";

export type DocumentStatus = "pending" | "parsing" | "chunking" | "embedding" | "ready" | "failed";

export interface DocumentInfo {
  id: string;
  session_id: string;
  original_name: string;
  mime: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: DocumentStatus;
  progress: number;
  stage_detail: string | null;
  error: string | null;
  created_at: string;
}

export interface Citation {
  marker: number;
  document_id: string;
  document_name: string;
  page: number | null;
  chunk_id: string;
  chunk_index: number | null;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  task: string;
  citations: Citation[];
  document_ids?: string[] | null;
  created_at: string;
  /** Client-side only: true while the answer is streaming. */
  pending?: boolean;
  /** Client-side only: error text when the answer failed. */
  error?: string;
  /** Client-side only: number of passages used while streaming. */
  sourceCount?: number;
}

export interface SessionSummary {
  id: string;
  title: string;
  document_count: number;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail extends SessionSummary {
  documents: DocumentInfo[];
  messages: ChatMessage[];
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  model: string;
  embedding_model: string;
  components: { name: string; ok: boolean; detail: string | null }[];
  missing_config: string[];
}

export interface UploadResult {
  documents: DocumentInfo[];
  duplicates: string[];
  rejected: { detail: string; code: string }[];
}

export interface ChunkDetail {
  id: string;
  document_id: string;
  document_name: string;
  chunk_index: number;
  page: number | null;
  heading: string | null;
  content: string;
}

export const BUSY_STATUSES: DocumentStatus[] = ["pending", "parsing", "chunking", "embedding"];

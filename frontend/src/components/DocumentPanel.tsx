import { formatBytes } from "../format";
import { BUSY_STATUSES, type DocumentInfo, type Task } from "../types";

interface Props {
  documents: DocumentInfo[];
  selected: Set<string>;
  disabled: boolean;
  open: boolean;
  onToggle: (id: string) => void;
  onTask: (task: Task, documentIds: string[]) => void;
  onDelete: (document: DocumentInfo) => void;
  onRetry: (document: DocumentInfo) => void;
  onUploadClick: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  parsing: "Reading",
  chunking: "Splitting",
  embedding: "Indexing",
  ready: "Ready",
  failed: "Failed",
};

const ACTIONS: { task: Task; label: string; title: string }[] = [
  { task: "summarize", label: "Summarize", title: "Summarize this document" },
  { task: "eli5", label: "Explain simply", title: "Explain like I'm new to the topic" },
  { task: "key_points", label: "Key points", title: "Extract the key points" },
  { task: "glossary", label: "Glossary", title: "Build a glossary of terms" },
];

export default function DocumentPanel({
  documents,
  selected,
  disabled,
  open,
  onToggle,
  onTask,
  onDelete,
  onRetry,
  onUploadClick,
}: Props) {
  const ready = documents.filter((doc) => doc.status === "ready");
  const scopeIds = selected.size ? [...selected] : ready.map((doc) => doc.id);

  return (
    <aside className={`doc-panel ${open ? "open" : ""}`} aria-label="Documents in this chat">
      <div className="panel-head">
        <h2>Documents</h2>
        <button className="btn small" onClick={onUploadClick}>
          Upload
        </button>
      </div>
      <p className="muted small">
        {selected.size
          ? `Questions are limited to ${selected.size} selected document${selected.size === 1 ? "" : "s"}.`
          : "Questions search every ready document in this chat. Tick documents to narrow the scope."}
      </p>

      {documents.length === 0 && (
        <div className="empty-docs muted small">No documents yet. Drop files onto the chat or use Upload.</div>
      )}

      <ul className="doc-list">
        {documents.map((doc) => {
          const busy = BUSY_STATUSES.includes(doc.status);
          return (
            <li key={doc.id} className={`doc-item status-${doc.status}`}>
              <div className="doc-row">
                <input
                  type="checkbox"
                  aria-label={`Limit questions to ${doc.original_name}`}
                  checked={selected.has(doc.id)}
                  disabled={doc.status !== "ready"}
                  onChange={() => onToggle(doc.id)}
                />
                <div className="doc-main">
                  <span className="doc-name" title={doc.original_name}>
                    {doc.original_name}
                  </span>
                  <span className="muted small">
                    {formatBytes(doc.size_bytes)}
                    {doc.status === "ready" &&
                      ` · ${doc.page_count || 1} page${doc.page_count === 1 ? "" : "s"} · ${doc.chunk_count} chunks`}
                  </span>
                </div>
                <span className={`badge ${doc.status}`}>{STATUS_LABEL[doc.status]}</span>
              </div>
              {busy && (
                <div className="progress" title={doc.stage_detail ?? ""}>
                  <div className="progress-bar" style={{ width: `${Math.round(doc.progress * 100)}%` }} />
                </div>
              )}
              {busy && doc.stage_detail && <span className="muted small">{doc.stage_detail}</span>}
              {doc.status === "failed" && <p className="error-text small">{doc.error}</p>}
              <div className="doc-actions">
                {doc.status === "ready" &&
                  ACTIONS.map((action) => (
                    <button
                      key={action.task}
                      className="btn tiny"
                      title={action.title}
                      disabled={disabled}
                      onClick={() => onTask(action.task, [doc.id])}
                    >
                      {action.label}
                    </button>
                  ))}
                {doc.status === "failed" && (
                  <button className="btn tiny" onClick={() => onRetry(doc)}>
                    Retry
                  </button>
                )}
                <button
                  className="btn tiny danger"
                  disabled={busy}
                  onClick={() => window.confirm(`Delete ${doc.original_name}?`) && onDelete(doc)}
                >
                  Delete
                </button>
              </div>
            </li>
          );
        })}
      </ul>

      {ready.length > 1 && (
        <div className="multi-actions">
          <span className="muted small">
            Across {selected.size ? "selected" : "all"} documents ({scopeIds.length})
          </span>
          <div className="doc-actions">
            <button className="btn tiny" disabled={disabled} onClick={() => onTask("summarize", scopeIds)}>
              Summarize all
            </button>
            <button className="btn tiny" disabled={disabled} onClick={() => onTask("compare", scopeIds)}>
              Compare
            </button>
            <button className="btn tiny" disabled={disabled} onClick={() => onTask("glossary", scopeIds)}>
              Combined glossary
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

import { relativeTime } from "../format";
import type { Health, SessionSummary } from "../types";

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  health: Health | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}

export default function Sidebar({ sessions, activeId, health, open, onSelect, onNew, onDelete, onRename }: Props) {
  const rename = (session: SessionSummary) => {
    const title = window.prompt("Rename chat", session.title)?.trim();
    if (title && title !== session.title) onRename(session.id, title);
  };

  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Chats">
      <div className="brand">
        <img src="/favicon.svg" alt="" width={28} height={28} />
        <div>
          <strong>Document Understanding Buddy</strong>
          <span>Chat with your documents</span>
        </div>
      </div>
      <button className="btn primary block" onClick={onNew}>
        + New chat
      </button>
      <nav className="session-list">
        {sessions.length === 0 && <p className="muted small">No chats yet.</p>}
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(session.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => event.key === "Enter" && onSelect(session.id)}
          >
            <div className="session-text">
              <span className="session-title" title={session.title}>
                {session.title}
              </span>
              <span className="muted small">
                {session.document_count} doc{session.document_count === 1 ? "" : "s"} ·{" "}
                {relativeTime(session.updated_at)}
              </span>
            </div>
            <div className="session-actions">
              <button
                className="icon-btn"
                title="Rename"
                onClick={(event) => {
                  event.stopPropagation();
                  rename(session);
                }}
              >
                ✎
              </button>
              <button
                className="icon-btn danger"
                title="Delete chat"
                onClick={(event) => {
                  event.stopPropagation();
                  if (window.confirm(`Delete "${session.title}" and its documents?`)) onDelete(session.id);
                }}
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </nav>
      {health && (
        <footer className="sidebar-footer small">
          <span className={`dot ${health.status === "ok" ? "ok" : "warn"}`} />
          {health.model} · v{health.version}
        </footer>
      )}
    </aside>
  );
}

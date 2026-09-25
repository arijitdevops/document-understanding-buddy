import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, streamChat } from "./api";
import ChatView from "./components/ChatView";
import DocumentPanel from "./components/DocumentPanel";
import HealthBanner from "./components/HealthBanner";
import Sidebar from "./components/Sidebar";
import { BUSY_STATUSES, type ChatMessage, type DocumentInfo, type Health, type SessionSummary, type Task } from "./types";

const TASK_PROMPTS: Record<Exclude<Task, "qa">, (names: string) => string> = {
  summarize: (names) => `Summarize ${names}.`,
  eli5: (names) => `Explain ${names} like I'm new to the topic.`,
  key_points: (names) => `What are the key points of ${names}?`,
  glossary: (names) => `Build a glossary of the important terms in ${names}.`,
  compare: (names) => `Compare ${names}: main themes, agreements and differences.`,
};

function localId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [title, setTitle] = useState("New chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [streaming, setStreaming] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [panel, setPanel] = useState<"none" | "sessions" | "docs">("none");
  const abort = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const flash = useCallback((text: string) => {
    setNotice(text);
    window.setTimeout(() => setNotice((current) => (current === text ? null : current)), 6000);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (error) {
      setHealthError((error as Error).message);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      /* surfaced by the health banner */
    }
  }, []);

  const openSession = useCallback(async (id: string) => {
    abort.current?.abort();
    setActiveId(id);
    setSelected(new Set());
    setPanel("none");
    try {
      const detail = await api.getSession(id);
      setTitle(detail.title);
      setMessages(detail.messages);
      setDocuments(detail.documents);
    } catch (error) {
      flash((error as Error).message);
    }
  }, [flash]);

  const newSession = useCallback(async (): Promise<string | null> => {
    try {
      const created = await api.createSession();
      await refreshSessions();
      setActiveId(created.id);
      setTitle(created.title);
      setMessages([]);
      setDocuments([]);
      setSelected(new Set());
      setPanel("none");
      return created.id;
    } catch (error) {
      flash((error as Error).message);
      return null;
    }
  }, [flash, refreshSessions]);

  // Initial load: health + sessions, then open the most recent chat.
  useEffect(() => {
    void refreshHealth();
    (async () => {
      try {
        const list = await api.listSessions();
        setSessions(list);
        if (list.length) await openSession(list[0].id);
      } catch {
        /* health banner explains */
      }
    })();
    const timer = window.setInterval(refreshHealth, 30000);
    return () => window.clearInterval(timer);
  }, [refreshHealth, openSession]);

  // Poll document status while anything is still being ingested.
  const busy = documents.some((doc) => BUSY_STATUSES.includes(doc.status));
  useEffect(() => {
    if (!busy || !activeId) return;
    const timer = window.setInterval(async () => {
      try {
        setDocuments(await api.listDocuments(activeId));
      } catch {
        /* retry on next tick */
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [busy, activeId]);

  const upload = async (files: File[]) => {
    const sessionId = activeId ?? (await newSession());
    if (!sessionId) return;
    try {
      const result = await api.uploadDocuments(sessionId, files);
      setDocuments(await api.listDocuments(sessionId));
      const notes: string[] = [];
      if (result.documents.length) notes.push(`Processing ${result.documents.length} file(s)...`);
      if (result.duplicates.length) notes.push(`Already in this chat: ${result.duplicates.join(", ")}`);
      result.rejected.forEach((item) => notes.push(item.detail));
      if (notes.length) flash(notes.join(" "));
      void refreshSessions();
    } catch (error) {
      flash(error instanceof ApiError ? error.message : "Upload failed.");
    }
  };

  const send = async (text: string, task: Task = "qa", documentIds?: string[]) => {
    if (!activeId || streaming) return;
    const scope = documentIds ?? (selected.size ? [...selected] : undefined);
    const userMessage: ChatMessage = {
      id: localId("user"),
      role: "user",
      content: text,
      task,
      citations: [],
      created_at: new Date().toISOString(),
    };
    const assistantId = localId("assistant");
    const assistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      task,
      citations: [],
      created_at: new Date().toISOString(),
      pending: true,
    };
    setMessages((current) => [...current, userMessage, assistant]);
    setStreaming(true);
    const controller = new AbortController();
    abort.current = controller;
    const patch = (update: (message: ChatMessage) => ChatMessage) =>
      setMessages((current) => current.map((message) => (message.id === assistantId ? update(message) : message)));

    await streamChat(
      activeId,
      { message: text, task, document_ids: scope },
      {
        onMeta: (meta) => {
          setTitle(meta.title);
          void refreshSessions();
        },
        onSources: (payload) => patch((message) => ({ ...message, sourceCount: payload.sources.length })),
        onToken: (delta) => patch((message) => ({ ...message, content: message.content + delta })),
        onCitations: (payload) =>
          patch((message) => ({ ...message, content: payload.answer, citations: payload.citations })),
        onDone: (payload) =>
          patch((message) => ({ ...message, id: payload.message_id ?? message.id, pending: false })),
        onError: (error) => patch((message) => ({ ...message, pending: false, error })),
      },
      controller.signal,
    );
    patch((message) => ({ ...message, pending: false }));
    setStreaming(false);
    abort.current = null;
  };

  const runTask = (task: Task, documentIds: string[]) => {
    const names = documents
      .filter((doc) => documentIds.includes(doc.id))
      .map((doc) => `"${doc.original_name}"`)
      .join(", ");
    if (task === "qa") return;
    void send(TASK_PROMPTS[task](names || "the documents"), task, documentIds);
  };

  const deleteDocument = async (doc: DocumentInfo) => {
    try {
      await api.deleteDocument(doc.id);
      setDocuments((current) => current.filter((item) => item.id !== doc.id));
      setSelected((current) => {
        const next = new Set(current);
        next.delete(doc.id);
        return next;
      });
      void refreshSessions();
    } catch (error) {
      flash((error as Error).message);
    }
  };

  const retryDocument = async (doc: DocumentInfo) => {
    try {
      await api.reingestDocument(doc.id);
      if (activeId) setDocuments(await api.listDocuments(activeId));
    } catch (error) {
      flash((error as Error).message);
    }
  };

  const deleteSession = async (id: string) => {
    try {
      await api.deleteSession(id);
      const remaining = sessions.filter((session) => session.id !== id);
      setSessions(remaining);
      if (id === activeId) {
        if (remaining.length) await openSession(remaining[0].id);
        else {
          setActiveId(null);
          setMessages([]);
          setDocuments([]);
          setTitle("New chat");
        }
      }
    } catch (error) {
      flash((error as Error).message);
    }
  };

  const renameSession = async (id: string, newTitle: string) => {
    try {
      await api.renameSession(id, newTitle);
      if (id === activeId) setTitle(newTitle);
      void refreshSessions();
    } catch (error) {
      flash((error as Error).message);
    }
  };

  const toggle = (id: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        health={health}
        open={panel === "sessions"}
        onSelect={openSession}
        onNew={newSession}
        onDelete={deleteSession}
        onRename={renameSession}
      />
      <main className="main">
        <div className="mobile-bar">
          <button className="btn small" onClick={() => setPanel(panel === "sessions" ? "none" : "sessions")}>
            Chats
          </button>
          <button className="btn small" onClick={() => setPanel(panel === "docs" ? "none" : "docs")}>
            Documents ({documents.length})
          </button>
        </div>
        <HealthBanner health={health} error={healthError} />
        {notice && (
          <div className="banner info" role="status" onClick={() => setNotice(null)}>
            {notice}
          </div>
        )}
        <ChatView
          title={activeId ? title : "Document Understanding Buddy"}
          messages={messages}
          documents={documents}
          streaming={streaming}
          onSend={(text) => void send(text)}
          onStop={() => abort.current?.abort()}
          onFiles={(files) => void upload(files)}
          fileInputRef={fileInputRef}
        />
      </main>
      <DocumentPanel
        documents={documents}
        selected={selected}
        disabled={streaming}
        open={panel === "docs"}
        onToggle={toggle}
        onTask={runTask}
        onDelete={deleteDocument}
        onRetry={retryDocument}
        onUploadClick={() => fileInputRef.current?.click()}
      />
      {panel !== "none" && <div className="scrim" onClick={() => setPanel("none")} />}
    </div>
  );
}

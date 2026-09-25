import { useEffect, useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import { ACCEPTED_TYPES } from "../format";
import type { ChatMessage, DocumentInfo } from "../types";
import MessageBubble from "./MessageBubble";

interface Props {
  title: string;
  messages: ChatMessage[];
  documents: DocumentInfo[];
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onFiles: (files: File[]) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
}

const SUGGESTIONS = [
  "What is this document about?",
  "List the most important numbers and dates.",
  "What are the main risks or open questions?",
];

export default function ChatView({ title, messages, documents, streaming, onSend, onStop, onFiles, fileInputRef }: Props) {
  const [text, setText] = useState("");
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const hasReady = documents.some((doc) => doc.status === "ready");

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const value = text.trim();
    if (!value || streaming) return;
    onSend(value);
    setText("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const onDragEnter = (event: DragEvent) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };
  const onDragLeave = () => {
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length) onFiles(files);
  };

  return (
    <section
      className="chat"
      onDragEnter={onDragEnter}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <header className="chat-head">
        <h1 title={title}>{title}</h1>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <h2>Ask anything about your documents</h2>
            <p className="muted">
              Drop PDF, Word, PowerPoint, text, Markdown, CSV or image files here. Answers only use your
              documents and cite the exact passages.
            </p>
            {!documents.length && (
              <button className="btn primary" onClick={() => fileInputRef.current?.click()}>
                Upload documents
              </button>
            )}
            {hasReady && (
              <div className="suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button key={suggestion} className="btn ghost" onClick={() => onSend(suggestion)}>
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={bottom} />
      </div>

      <form className="composer" onSubmit={submit}>
        <button
          type="button"
          className="icon-btn attach"
          title="Attach documents"
          onClick={() => fileInputRef.current?.click()}
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPTED_TYPES}
          hidden
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length) onFiles(files);
            event.target.value = "";
          }}
        />
        <textarea
          value={text}
          rows={1}
          placeholder={hasReady ? "Ask a question about your documents..." : "Upload a document to start"}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Message"
        />
        {streaming ? (
          <button type="button" className="btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn primary" disabled={!text.trim()}>
            Send
          </button>
        )}
      </form>

      {dragging && (
        <div className="drop-overlay">
          <div>
            <strong>Drop files to add them to this chat</strong>
            <span className="muted">PDF · DOCX · PPTX · TXT · MD · CSV · PNG/JPG</span>
          </div>
        </div>
      )}
    </section>
  );
}

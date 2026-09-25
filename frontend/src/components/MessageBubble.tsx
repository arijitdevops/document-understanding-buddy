import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { citationMarker, linkifyCitations } from "../citations";
import { TASK_LABELS } from "../format";
import type { ChatMessage } from "../types";
import CitationChip from "./CitationChip";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const byMarker = useMemo(
    () => new Map(message.citations.map((citation) => [citation.marker, citation])),
    [message.citations],
  );
  const markdown = useMemo(
    () => linkifyCitations(message.content, new Set(byMarker.keys())),
    [message.content, byMarker],
  );

  if (message.role === "user") {
    return (
      <div className="message user">
        {message.task !== "qa" && <span className="task-tag">{TASK_LABELS[message.task] ?? message.task}</span>}
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="message assistant">
      <div className="avatar" aria-hidden>
        DB
      </div>
      <div className="bubble">
        {message.pending && !message.content && (
          <div className="thinking">
            <span className="spinner" />
            {message.sourceCount ? `Reading ${message.sourceCount} passages...` : "Searching your documents..."}
          </div>
        )}
        <div className="markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const marker = citationMarker(href);
                const citation = marker !== null ? byMarker.get(marker) : undefined;
                if (citation) return <CitationChip citation={citation} />;
                return (
                  <a href={href} target="_blank" rel="noreferrer">
                    {children}
                  </a>
                );
              },
            }}
          >
            {markdown}
          </ReactMarkdown>
          {message.pending && message.content && <span className="cursor" />}
        </div>
        {message.error && <div className="error-text">⚠ {message.error}</div>}
        {message.citations.length > 0 && (
          <div className="sources">
            <span className="muted small">Sources</span>
            <div className="source-chips">
              {message.citations.map((citation) => (
                <CitationChip key={citation.marker} citation={citation} compact={false} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { citationLabel } from "../citations";
import type { Citation } from "../types";

interface Props {
  citation: Citation;
  compact?: boolean;
}

/** A small `[n]` chip; clicking it shows where the claim came from. */
export default function CitationChip({ citation, compact = true }: Props) {
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  const loadFull = async () => {
    setLoading(true);
    try {
      setFull((await api.getChunk(citation.chunk_id)).content);
    } catch {
      setFull("The passage is no longer available (was the document deleted?).");
    } finally {
      setLoading(false);
    }
  };

  return (
    <span className="citation" ref={ref}>
      <button
        type="button"
        className={compact ? "chip" : "chip wide"}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        title={citationLabel(citation)}
      >
        {compact ? citation.marker : `[${citation.marker}] ${citationLabel(citation)}`}
      </button>
      {open && (
        <span className="popover" role="dialog">
          <span className="popover-head">
            <strong>[{citation.marker}]</strong> {citationLabel(citation)}
          </span>
          <span className="popover-body">{full ?? citation.snippet}</span>
          {full === null && (
            <button type="button" className="link-btn" onClick={loadFull} disabled={loading}>
              {loading ? "Loading..." : "Show full passage"}
            </button>
          )}
        </span>
      )}
    </span>
  );
}

import type { Citation } from "./types";

const MARKER = /\[(\d{1,3})\](?!\()/g;

/**
 * Turn `[n]` citation markers into Markdown links (`[n](#cite-n)`) so the
 * Markdown renderer hands them to our custom link component, which renders
 * a citation chip. Markers inside inline code or fenced code blocks are left
 * untouched, as are markers that do not correspond to a known citation.
 */
export function linkifyCitations(markdown: string, known: Set<number>): string {
  const parts = markdown.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return parts
    .map((part, index) =>
      index % 2 === 1
        ? part
        : part.replace(MARKER, (match, value: string) =>
            known.has(Number(value)) ? `[${value}](#cite-${value})` : match,
          ),
    )
    .join("");
}

/** Parse the marker number out of a `#cite-n` href. */
export function citationMarker(href: string | undefined): number | null {
  const match = /^#cite-(\d+)$/.exec(href ?? "");
  return match ? Number(match[1]) : null;
}

/** Human-readable location such as "report.pdf · p. 4 · chunk 12". */
export function citationLabel(citation: Citation): string {
  const bits = [citation.document_name];
  if (citation.page) bits.push(`p. ${citation.page}`);
  if (citation.chunk_index !== null && citation.chunk_index !== undefined) {
    bits.push(`chunk ${citation.chunk_index + 1}`);
  }
  return bits.join(" · ");
}

/** Unique documents cited by an answer, in first-citation order. */
export function citedDocuments(citations: Citation[]): string[] {
  return [...new Set(citations.map((c) => c.document_name))];
}

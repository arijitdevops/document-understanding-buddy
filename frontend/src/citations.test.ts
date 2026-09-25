import { describe, expect, it } from "vitest";
import { citationLabel, citationMarker, citedDocuments, linkifyCitations } from "./citations";
import type { Citation } from "./types";

const citation = (marker: number, name = "report.pdf"): Citation => ({
  marker,
  document_id: "d1",
  document_name: name,
  page: 4,
  chunk_id: `c${marker}`,
  chunk_index: 11,
  snippet: "text",
});

describe("linkifyCitations", () => {
  it("links known markers only", () => {
    expect(linkifyCitations("Fact [1]. Other [3].", new Set([1]))).toBe("Fact [1](#cite-1). Other [3].");
  });

  it("leaves code untouched", () => {
    const text = "Use `arr[1]` and\n```\nx[1]\n```\nsee [1]";
    expect(linkifyCitations(text, new Set([1]))).toBe("Use `arr[1]` and\n```\nx[1]\n```\nsee [1](#cite-1)");
  });

  it("does not double-link existing markdown links", () => {
    expect(linkifyCitations("[1](http://x)", new Set([1]))).toBe("[1](http://x)");
  });
});

describe("citation helpers", () => {
  it("parses markers from hrefs", () => {
    expect(citationMarker("#cite-12")).toBe(12);
    expect(citationMarker("https://example.com")).toBeNull();
  });

  it("builds a readable label", () => {
    expect(citationLabel(citation(1))).toBe("report.pdf · p. 4 · chunk 12");
  });

  it("lists cited documents once", () => {
    expect(citedDocuments([citation(1), citation(2), citation(3, "b.docx")])).toEqual(["report.pdf", "b.docx"]);
  });
});

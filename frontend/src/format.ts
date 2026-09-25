export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function relativeTime(iso: string, now: Date = new Date()): string {
  // The API returns naive UTC timestamps; treat them as UTC.
  const then = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  const seconds = Math.max(0, Math.round((now.getTime() - then.getTime()) / 1000));
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return then.toLocaleDateString();
}

export const ACCEPTED_TYPES = ".pdf,.docx,.pptx,.txt,.md,.markdown,.csv,.png,.jpg,.jpeg,.webp";

export const TASK_LABELS: Record<string, string> = {
  qa: "Question",
  summarize: "Summary",
  eli5: "Explain like I'm new",
  key_points: "Key points",
  glossary: "Glossary",
  compare: "Comparison",
};

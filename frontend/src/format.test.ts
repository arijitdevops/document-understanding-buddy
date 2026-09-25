import { describe, expect, it } from "vitest";
import { formatBytes, relativeTime } from "./format";

describe("format", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });

  it("treats naive API timestamps as UTC", () => {
    const now = new Date("2026-01-01T12:10:00Z");
    expect(relativeTime("2026-01-01T12:00:00", now)).toBe("10 min ago");
    expect(relativeTime("2026-01-01T12:09:50Z", now)).toBe("just now");
  });
});

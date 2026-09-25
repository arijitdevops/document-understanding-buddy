import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses complete events", () => {
    const parser = new SseParser();
    const events = parser.push('event: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {}\n\n');
    expect(events).toEqual([
      { event: "token", data: '{"text":"Hi"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("buffers events split across chunks", () => {
    const parser = new SseParser();
    expect(parser.push("event: tok")).toEqual([]);
    expect(parser.push('en\ndata: {"text":"a"}\n')).toEqual([]);
    expect(parser.push("\nevent: done\r\ndata: {}\r\n\r\n")).toEqual([
      { event: "token", data: '{"text":"a"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("ignores comments and defaults the event name", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\n\ndata: x\n\n")).toEqual([{ event: "message", data: "x" }]);
  });
});

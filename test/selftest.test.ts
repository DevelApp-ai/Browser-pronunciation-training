import { describe, expect, it } from "vitest";
import { buildLog, detectBrowser, logToText, summarize, type TestEntry } from "../src/selftest/selftest.ts";

describe("detectBrowser", () => {
  it("detects Edge despite the Chrome token in its UA", () => {
    const ua =
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0";
    expect(detectBrowser(ua)).toEqual({ name: "Edge", version: "127.0.0.0" });
  });

  it("detects Firefox", () => {
    const ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0";
    expect(detectBrowser(ua)).toEqual({ name: "Firefox", version: "130.0" });
  });

  it("falls back to Chrome when only the Chrome token is present", () => {
    const ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/127.0.0.0 Safari/537.36";
    expect(detectBrowser(ua).name).toBe("Chrome");
  });
});

describe("summarize / log rendering", () => {
  const entries: TestEntry[] = [
    { name: "a", status: "pass", detail: "ok" },
    { name: "b", status: "fail", detail: "boom" },
    { name: "c", status: "warn", detail: "hmm" },
    { name: "d", status: "skip", detail: "n/a" },
  ];
  it("counts statuses", () => {
    expect(summarize(entries)).toEqual({ pass: 1, fail: 1, warn: 1, skip: 1, total: 4 });
  });
  it("renders a readable text log with every status line", () => {
    const text = logToText({
      app: "browser-pronunciation-training",
      timestamp: "2026-10-02T12:00:00.000Z",
      url: "https://example.test/",
      userAgent: "ua",
      platform: "test",
      browser: { name: "Edge", version: "127" },
      entries,
      summary: summarize(entries),
    });
    expect(text).toContain("Browser: Edge 127");
    expect(text).toContain("[PASS] a — ok");
    expect(text).toContain("[FAIL] b — boom");
    expect(text).toContain("[WARN] c — hmm");
    expect(text).toContain("[SKIP] d — n/a");
    expect(text).toContain("1 pass, 1 fail, 1 warn, 1 skip (4 total)");
  });
  it("buildLog shapes a complete log object", () => {
    const log = buildLog(entries);
    expect(log.summary.total).toBe(4);
    expect(log.entries).toHaveLength(4);
  });
});

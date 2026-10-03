import { describe, expect, it } from "vitest";
import { DownloadTracker, formatMB, modelFilesCached } from "../src/model/download.ts";

describe("DownloadTracker (issue #18)", () => {
  it("aggregates progress across files", () => {
    const tr = new DownloadTracker();
    tr.update({ file: "a.onnx", loaded: 10, total: 100 });
    const agg = tr.update({ file: "b.onnx", loaded: 50, total: 100 });
    expect(agg.files).toBe(2);
    expect(agg.loadedBytes).toBe(60);
    expect(agg.totalBytes).toBe(200);
    expect(agg.percent).toBe(30);
    expect(agg.complete).toBe(false);
  });

  it("ignores events without a known total", () => {
    const tr = new DownloadTracker();
    const agg = tr.update({ file: "a.onnx", loaded: 10, total: 0 });
    expect(agg.files).toBe(0);
    expect(agg.percent).toBe(0);
    expect(agg.complete).toBe(false);
  });

  it("clamps loaded bytes to the file total", () => {
    const tr = new DownloadTracker();
    const agg = tr.update({ file: "a.onnx", loaded: 999, total: 100 });
    expect(agg.loadedBytes).toBe(100);
    expect(agg.percent).toBe(100);
    expect(agg.complete).toBe(true);
  });

  it("is only complete when every byte of every file has arrived", () => {
    const tr = new DownloadTracker();
    tr.update({ file: "a.onnx", loaded: 100, total: 100 });
    const agg = tr.update({ file: "b.onnx", loaded: 50, total: 100 });
    expect(agg.complete).toBe(false);
  });
});

describe("formatMB", () => {
  it("rounds to whole megabytes", () => {
    expect(formatMB(0)).toBe("0 MB");
    expect(formatMB(52 * 1024 * 1024)).toBe("52 MB");
    expect(formatMB(1500 * 1024)).toBe("1 MB");
  });
});

describe("modelFilesCached", () => {
  it("reports false outside the browser (no Cache Storage in Node)", async () => {
    expect(await modelFilesCached()).toBe(false);
  });
});

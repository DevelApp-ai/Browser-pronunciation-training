import { describe, expect, it } from "vitest";
import { percentile, statsToMarkdown, syntheticUtterances, type LatencyStats } from "../src/benchmark/harness.ts";

describe("percentile", () => {
  it("computes p50/p95 on sorted samples", () => {
    const sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(sorted, 50)).toBe(5);
    expect(percentile(sorted, 95)).toBe(10);
    expect(percentile([42], 50)).toBe(42);
    expect(percentile([], 95)).toBeNaN();
  });
});

describe("syntheticUtterances", () => {
  it("is deterministic, correctly sized, and distinct per utterance", () => {
    const a = syntheticUtterances(2, 1.5);
    const b = syntheticUtterances(2, 1.5);
    expect(a).toHaveLength(2);
    expect(a[0]).toHaveLength(24000); // 1.5 s × 16 kHz
    expect(Array.from(a[0]!)).toEqual(Array.from(b[0]!));
    expect(Array.from(a[0]!)).not.toEqual(Array.from(a[1]!));
  });
});

describe("statsToMarkdown", () => {
  it("renders the model × device × backend × latency table", () => {
    const stats: LatencyStats[] = [
      { backend: "wasm", model: "espeak-phoneme", deviceTier: "high", p50Ms: 420, p95Ms: 640, n: 15 },
      { backend: "webgpu", model: "espeak-phoneme", deviceTier: "high", p50Ms: 500, p95Ms: 700, n: 15, webgpuSlower: true },
    ];
    const md = statsToMarkdown(stats);
    expect(md).toContain("| espeak-phoneme | high | wasm | 420 | 640 | 15 |");
    expect(md).toContain("| espeak-phoneme | high | webgpu | 500 | 700 | 15 | WebGPU **slower** than WASM — keep WASM pinned |");
  });
});

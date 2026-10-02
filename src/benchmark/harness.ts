/**
 * In-app benchmark harness — issue #7 (answers TDS Open Question 1).
 *
 * Runs each loaded model on fixed test utterances and measures p50/p95
 * latency per backend (WASM / WebGPU), feature-detecting `navigator.gpu`.
 * Results are recorded in docs/backend-benchmarks.md, which pins the
 * default backend per model. Known risk: wav2vec2 conv stacks can be
 * SLOWER on WebGPU (ONNX Runtime issue #21618) — the harness exists to
 * catch exactly that.
 *
 * Run it in the app via `?benchmark=1` (see src/main.ts) and paste the
 * generated markdown table into docs/backend-benchmarks.md.
 */
import type { PhonemeModel } from "../model/runtime.ts";

export interface LatencyStats {
  backend: "wasm" | "webgpu";
  model: string;
  deviceTier: string;
  p50Ms: number;
  p95Ms: number;
  n: number;
  /** True when WebGPU ran but lost to WASM. */
  webgpuSlower?: boolean;
}

export function detectDeviceTier(): string {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const hasWebgpu = !!gpu;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  const cores = navigator.hardwareConcurrency ?? 4;
  if (!hasWebgpu) return "no-webgpu";
  if (mobile) return "mobile";
  return cores >= 8 ? "high" : "integrated";
}

export async function webgpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

export function percentile(sorted: number[], p: number): number {
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)]!;
}

/**
 * Deterministic pseudo-PCM (seeded LCG mixed with a sine) for fixed test
 * utterances — latency benchmarks need reproducible input, not silence.
 */
export function syntheticUtterance(seconds = 2, sampleRate = 16000, seed = 42): Float32Array {
  const n = Math.round(seconds * sampleRate);
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const noise = ((s / 0xffffffff) * 2 - 1) * 0.1;
    out[i] = 0.5 * Math.sin((2 * Math.PI * 220 * i) / sampleRate) + noise;
  }
  return out;
}

/** Fixed utterance set for a benchmark run (distinct seeds per utterance). */
export function syntheticUtterances(count = 3, seconds = 2, sampleRate = 16000): Float32Array[] {
  return Array.from({ length: count }, (_, i) => syntheticUtterance(seconds, sampleRate, 42 + i * 7));
}

/**
 * Benchmark one model on a fixed utterance set against the chosen backend(s).
 * Warm-up runs are excluded. Re-run whenever a new model artifact lands (M2
 * pipeline hook) and paste the table into docs/backend-benchmarks.md.
 */
export async function benchmarkModel(
  model: PhonemeModel,
  utterances: Float32Array[],
  opts: { backends?: Array<"wasm" | "webgpu">; runsPerUtterance?: number; modelName?: string } = {},
): Promise<LatencyStats[]> {
  const backends = opts.backends ?? ["wasm"];
  const runs = opts.runsPerUtterance ?? 5;
  const stats: LatencyStats[] = [];
  if (utterances.length === 0) return stats;

  for (const backend of backends) {
    if (backend === "webgpu" && !(await webgpuAvailable())) continue;
    // warm-up (JIT / shader compilation / first WASM run)
    await model.logitFrames(utterances[0]!, backend);
    const samples: number[] = [];
    for (const pcm of utterances) {
      for (let i = 0; i < runs; i++) {
        const { latencyMs } = await model.logitFrames(pcm, backend);
        samples.push(latencyMs);
      }
    }
    const sorted = [...samples].sort((a, b) => a - b);
    stats.push({
      backend,
      model: opts.modelName ?? "espeak-phoneme",
      deviceTier: detectDeviceTier(),
      p50Ms: Math.round(percentile(sorted, 50)),
      p95Ms: Math.round(percentile(sorted, 95)),
      n: samples.length,
    });
  }
  const wasm = stats.find((s) => s.backend === "wasm");
  const gpu = stats.find((s) => s.backend === "webgpu");
  if (wasm && gpu) gpu.webgpuSlower = gpu.p50Ms > wasm.p50Ms;
  return stats;
}

/**
 * Markdown table for docs/backend-benchmarks.md
 * (model × device tier × backend × latency), ready to paste.
 */
export function statsToMarkdown(stats: LatencyStats[]): string {
  const lines = [
    "| Model | Device tier | Backend | p50 (ms) | p95 (ms) | n | Notes |",
    "| --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const s of stats) {
    const notes = s.webgpuSlower ? "WebGPU **slower** than WASM — keep WASM pinned" : "";
    lines.push(`| ${s.model} | ${s.deviceTier} | ${s.backend} | ${s.p50Ms} | ${s.p95Ms} | ${s.n} | ${notes} |`);
  }
  return lines.join("\n");
}

/** Pinned default backend per model — edit via benchmark results only. */
export const DEFAULT_BACKENDS: Record<string, "wasm" | "webgpu"> = {
  "espeak-phoneme": "wasm", // pinned until docs/backend-benchmarks.md says otherwise
};

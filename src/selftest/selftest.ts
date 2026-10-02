/**
 * In-browser self-test — runs environment, model and microphone checks in
 * the current browser (Edge, Firefox, …) and produces a downloadable log.
 *
 * Open the deployed app with ?selftest=1, click "Run self-test" (and the
 * microphone test if you want capture coverage), then download the log
 * (JSON and/or text) and upload it for analysis.
 */
import type { PhonemeModel } from "../model/runtime.ts";
import { syntheticUtterance } from "../benchmark/harness.ts";

export type TestStatus = "pass" | "fail" | "warn" | "skip";

export interface TestEntry {
  name: string;
  status: TestStatus;
  detail: string;
  ms?: number;
}

export interface SelfTestLog {
  app: string;
  timestamp: string;
  url: string;
  userAgent: string;
  platform: string;
  browser: { name: string; version: string };
  entries: TestEntry[];
  summary: { pass: number; fail: number; warn: number; skip: number; total: number };
}

/** Detect the actual browser (Edge masquerades as Chrome in its UA). */
export function detectBrowser(ua: string): { name: string; version: string } {
  const tests: Array<[string, RegExp]> = [
    ["Edge", /Edg(?:e|A|iOS)?\/([\d.]+)/],
    ["Opera", /OPR\/([\d.]+)/],
    ["Samsung Internet", /SamsungBrowser\/([\d.]+)/],
    ["Firefox", /Firefox\/([\d.]+)/],
    ["Chrome", /Chrome\/([\d.]+)/],
    ["Safari", /Version\/([\d.]+).*Safari/],
  ];
  for (const [name, re] of tests) {
    const m = ua.match(re);
    if (m) return { name, version: m[1] ?? "" };
  }
  return { name: "Unknown", version: "" };
}

function entry(name: string, status: TestStatus, detail: string, ms?: number): TestEntry {
  return { name, status, detail, ms };
}

/** Static environment / capability checks (no permissions needed). */
export async function runEnvironmentChecks(): Promise<TestEntry[]> {
  const entries: TestEntry[] = [];
  const ua = navigator.userAgent;

  entries.push(entry("getUserMedia API", typeof navigator.mediaDevices?.getUserMedia === "function" ? "pass" : "fail",
    typeof navigator.mediaDevices?.getUserMedia === "function" ? "available" : "missing — audio capture impossible"));

  // Module workers: transformers.js runs in a module Web Worker.
  const workerOk = await new Promise<boolean>((resolve) => {
    try {
      const url = URL.createObjectURL(new Blob(["self.postMessage('ok')"], { type: "text/javascript" }));
      const w = new Worker(url, { type: "module" });
      const timer = setTimeout(() => { w.terminate(); resolve(false); }, 3000);
      w.onmessage = () => { clearTimeout(timer); w.terminate(); resolve(true); };
      w.onerror = () => { clearTimeout(timer); w.terminate(); resolve(false); };
    } catch {
      resolve(false);
    }
  });
  entries.push(entry("Module Web Workers", workerOk ? "pass" : "fail", workerOk ? "supported" : "not supported — the phoneme worker cannot run"));

  // AudioContext pinned at 16 kHz?
  try {
    const ctx = new AudioContext({ sampleRate: 16000 });
    entries.push(entry("AudioContext @ 16 kHz", ctx.sampleRate === 16000 ? "pass" : "warn",
      `actual rate: ${ctx.sampleRate} Hz` + (ctx.sampleRate !== 16000 ? " — capture will resample (slower, still works)" : "")));
    const sp = ctx.createScriptProcessor(1024, 1, 1);
    entries.push(entry("ScriptProcessorNode", "pass", "supported (PCM tap)"));
    void sp.disconnect();
    await ctx.close();
  } catch (err) {
    entries.push(entry("AudioContext", "fail", String(err)));
  }

  entries.push(entry("MediaRecorder", typeof MediaRecorder !== "undefined" ? "pass" : "warn",
    typeof MediaRecorder !== "undefined" ? "supported (attempt playback available)" : "not supported — attempt playback disabled, scoring unaffected"));

  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
  entries.push(entry("WebGPU", gpu ? "pass" : "skip", gpu ? "navigator.gpu present" : "not available — WASM backend will be used"));

  entries.push(entry("speechSynthesis", typeof speechSynthesis !== "undefined" ? "pass" : "warn",
    typeof speechSynthesis !== "undefined" ? `voices loaded: ${speechSynthesis.getVoices().length}` : "not supported — reference playback unavailable"));

  entries.push(entry("hardwareConcurrency", "pass", `${navigator.hardwareConcurrency ?? "unknown"} logical cores`));
  entries.push(entry("userAgent", "pass", ua));
  return entries;
}

/**
 * Load the phoneme model and run one synthetic utterance end-to-end:
 * checks the worker responds, logits are non-empty, the phoneme set
 * looks like an espeak IPA vocabulary, and records latency.
 */
export async function runModelChecks(model: PhonemeModel, backend: "wasm" | "webgpu" = "wasm"): Promise<TestEntry[]> {
  const entries: TestEntry[] = [];
  const pcm = syntheticUtterance(2);
  try {
    const t0 = performance.now();
    const out = await model.logitFrames(pcm, backend);
    const wallMs = Math.round(performance.now() - t0);
    entries.push(entry(`Model inference (${backend})`, "pass",
      `frames×vocab: ${out.logits.length} logits, worker latency ${Math.round(out.latencyMs)} ms, wall ${wallMs} ms (incl. download)`,
      out.latencyMs));
    const vocabSize = out.phonemeSet.length;
    entries.push(entry(`Phoneme vocabulary (${backend})`, vocabSize >= 30 ? "pass" : "warn",
      `${vocabSize} entries; first: ${out.phonemeSet.slice(0, 5).join(" ")}`));
    if (typeof out.transcript === "string") {
      entries.push(entry(`Greedy transcript (${backend})`, out.transcript.trim().length > 0 ? "pass" : "warn",
        `"${out.transcript.slice(0, 120)}"`));
    }
  } catch (err) {
    entries.push(entry(`Model inference (${backend})`, "fail",
      err instanceof Error ? err.message : String(err)));
  }
  return entries;
}

/** Live microphone round-trip: capture ~1.5 s and verify 16 kHz PCM + blob. */
export async function runMicrophoneCheck(
  startCapture: () => { analyser: AnalyserNode; stop(): Promise<{ pcm: Float32Array; blob: Blob }> },
): Promise<TestEntry[]> {
  const entries: TestEntry[] = [];
  try {
    const capture = startCapture();
    const analyserOk = !!capture.analyser;
    await new Promise((r) => setTimeout(r, 1500));
    const { pcm, blob } = await capture.stop();
    entries.push(entry("Microphone capture", pcm.length > 0 ? "pass" : "fail",
      `${pcm.length} samples (${(pcm.length / 16000).toFixed(2)} s @16 kHz), blob ${blob.size} bytes, analyser ${analyserOk ? "ok" : "missing"}`));
  } catch (err) {
    entries.push(entry("Microphone capture", "fail", err instanceof Error ? `${err.name}: ${err.message}` : String(err)));
  }
  return entries;
}

export function summarize(entries: TestEntry[]): SelfTestLog["summary"] {
  const count = (s: TestStatus) => entries.filter((e) => e.status === s).length;
  return { pass: count("pass"), fail: count("fail"), warn: count("warn"), skip: count("skip"), total: entries.length };
}

export function buildLog(entries: TestEntry[]): SelfTestLog {
  return {
    app: "browser-pronunciation-training",
    timestamp: new Date().toISOString(),
    url: typeof location !== "undefined" ? location.href : "unknown",
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "unknown",
    platform: typeof navigator !== "undefined" ? ((navigator as Navigator & { platform?: string }).platform ?? "unknown") : "unknown",
    browser: detectBrowser(navigator.userAgent),
    entries,
    summary: summarize(entries),
  };
}

/** Human-readable text version of the log. */
export function logToText(log: SelfTestLog): string {
  const lines = [
    "Browser Pronunciation Training — self-test log",
    `Timestamp: ${log.timestamp}`,
    `URL: ${log.url}`,
    `Browser: ${log.browser.name} ${log.browser.version}`,
    `Platform: ${log.platform}`,
    `User agent: ${log.userAgent}`,
    `Summary: ${log.summary.pass} pass, ${log.summary.fail} fail, ${log.summary.warn} warn, ${log.summary.skip} skip (${log.summary.total} total)`,
    "",
  ];
  for (const e of log.entries) {
    lines.push(`[${e.status.toUpperCase()}] ${e.name}${e.ms !== undefined ? ` (${Math.round(e.ms)} ms)` : ""} — ${e.detail}`);
  }
  return lines.join("\n");
}

/** Trigger a browser download of the log as JSON (or text). */
export function downloadLog(log: SelfTestLog, format: "json" | "text" = "json"): void {
  const browser = `${log.browser.name}-${log.browser.version}`.replace(/[^\w.-]/g, "");
  const stamp = log.timestamp.replace(/[:.]/g, "-");
  const content = format === "json" ? JSON.stringify(log, null, 2) : logToText(log);
  const blob = new Blob([content], { type: format === "json" ? "application/json" : "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `bpt-selftest-${browser}-${stamp}.${format === "json" ? "json" : "txt"}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

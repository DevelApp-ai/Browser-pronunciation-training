/**
 * App entry — wires capture (#4) → model logits (#5) → GOP scorer (#6)
 * → phoneme-pack tips (#8) → feedback UI (#9) → prosody merge (#16)
 * → eval collection (#19).
 *
 * Query params:
 *   lang=en|da|ne — UI + phoneme-pack language
 *   benchmark=1   — run the WebGPU-vs-WASM harness (#7) and print the
 *                   markdown table for docs/backend-benchmarks.md
 *   selftest=1    — run in-browser self-tests (environment, model,
 *                   microphone) and download a test log for upload
 *   eval=1        — show the evaluation-data controls (CSV download for
 *                   human-rater correlation, issue #19)
 */
import { listInputDevices, startCapture } from "./audio/capture.ts";
import { benchmarkModel, DEFAULT_BACKENDS, statsToMarkdown, syntheticUtterances } from "./benchmark/harness.ts";
import { downloadEvalCsv, toEvalRecord, toEvalCsv, type EvalRecord } from "./eval/collect.ts";
import { loadPhonemeModel } from "./model/runtime.ts";
import { loadPack } from "./packs/loader.ts";
import { graphemesToPhonemes } from "./packs/schema.ts";
import { withProsody } from "./prosody/merge.ts";
import { scoreUtterance } from "./scoring/gop.ts";
import { drawWaveform, playBlob, renderPhonemeRibbon, renderScoreCard, speakReference } from "./ui/feedback.ts";
import { t, type Lang } from "./ui/i18n.ts";
import {
  buildLog,
  detectBrowser,
  downloadLog,
  logToText,
  runEnvironmentChecks,
  runMicrophoneCheck,
  runModelChecks,
  type SelfTestLog,
  type TestEntry,
} from "./selftest/selftest.ts";

const EXERCISES: Record<string, string[]> = {
  en: ["Hello world", "I think three things", "They usually measure it", "The pronunciation practice"],
};

function renderSelfTestPage(): void {
  const app = document.querySelector<HTMLElement>("#app")!;
  const entries: TestEntry[] = [];
  let log: SelfTestLog = buildLog(entries);

  app.innerHTML = [
    "<h1>Self-test</h1>",
    "<p id='stIntro'>Run the checks in this browser, then download the log (JSON and/or text) and upload it.</p>",
    "<div id='controls'>",
    "  <button id='stRun'>Run self-test (environment + model)</button>",
    "  <button id='stMic'>Run microphone test</button>",
    "  <button id='stJson' disabled>Download log (JSON)</button>",
    "  <button id='stTxt' disabled>Download log (TXT)</button>",
    "  <button id='stCopy' disabled>Copy text summary</button>",
    "</div>",
    "<ul id='stResults'></ul>",
  ].join("");

  const results = app.querySelector<HTMLElement>("#stResults")!;
  const jsonBtn = app.querySelector<HTMLButtonElement>("#stJson")!;
  const txtBtn = app.querySelector<HTMLButtonElement>("#stTxt")!;
  const copyBtn = app.querySelector<HTMLButtonElement>("#stCopy")!;

  function record(list: TestEntry[]): void {
    for (const e of list) {
      entries.push(e);
      const li = document.createElement("li");
      const mark = { pass: "✅", fail: "❌", warn: "⚠️", skip: "⏭️" }[e.status];
      li.textContent = `${mark} ${e.name} — ${e.detail}`;
      results.appendChild(li);
    }
    log = buildLog(entries);
    for (const b of [jsonBtn, txtBtn, copyBtn]) b.disabled = entries.length === 0;
  }

  app.querySelector<HTMLButtonElement>("#stRun")!.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget as HTMLButtonElement;
    btn.disabled = true;
    btn.textContent = "Running… (first run downloads the model)";
    try {
      record(await runEnvironmentChecks());
      const model = loadPhonemeModel();
      record(await runModelChecks(model, "wasm"));
      model.terminate();
    } catch (err) {
      record([{ name: "self-test", status: "fail", detail: String(err) }]);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run self-test (environment + model)";
    }
  });

  app.querySelector<HTMLButtonElement>("#stMic")!.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget as HTMLButtonElement;
    btn.disabled = true;
    btn.textContent = "Recording 1.5 s…";
    try {
      record(await runMicrophoneCheck(() => startCapture()));
    } finally {
      btn.disabled = false;
      btn.textContent = "Run microphone test";
    }
  });

  jsonBtn.addEventListener("click", () => downloadLog(log, "json"));
  txtBtn.addEventListener("click", () => downloadLog(log, "text"));
  copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(logToText(log));
    copyBtn.textContent = "Copied!";
    setTimeout(() => (copyBtn.textContent = "Copy text summary"), 1500);
  });

  const b = detectBrowser(navigator.userAgent);
  app.querySelector<HTMLElement>("#stIntro")!.textContent +=
    ` Detected: ${b.name} ${b.version}.`;
}

async function main() {
  const params = new URLSearchParams(location.search);
  const app = document.querySelector<HTMLElement>("#app")!;

  if (params.has("selftest")) {
    renderSelfTestPage();
    return;
  }

  const lang = (params.get("lang") ?? "en") as Lang;
  const s = t(lang);
  const evalMode = params.has("eval");
  const pack = await loadPack(lang);
  const model = loadPhonemeModel();

  // Benchmark mode (issue #7): run the harness on fixed synthetic utterances
  // and print the table to paste into docs/backend-benchmarks.md.
  if (params.has("benchmark")) {
    app.innerHTML = "<h1>Backend benchmark</h1><pre id='benchOut'>running…</pre>";
    const stats = await benchmarkModel(model, syntheticUtterances(), {
      backends: ["wasm", "webgpu"],
      modelName: "espeak-phoneme",
    });
    app.querySelector<HTMLElement>("#benchOut")!.textContent =
      stats.length > 0 ? statsToMarkdown(stats) : "No backend ran (WebGPU unavailable and WASM skipped?)";
    return;
  }

  const sentence = EXERCISES[lang]?.[0] ?? "Hello world";
  app.innerHTML = `
    <h1>${sentence}</h1>
    <div id="controls">
      <button id="refBtn">${s.playReference}</button>
      <button id="slowBtn">${s.slow}</button>
      <button id="recBtn">${s.record}</button>
      <button id="playBtn" disabled>${s.play}</button>
      ${evalMode ? '<button id="evalBtn" disabled>Download eval CSV</button>' : ""}
    </div>
    <canvas id="wave" width="600" height="120"></canvas>
    <div id="scoreCard"><p>${s.scoreNone}</p></div>
    <div id="ribbon"></div>
  `;

  const refLang = lang === "en" ? "en-US" : lang;
  app.querySelector<HTMLButtonElement>("#refBtn")!.addEventListener("click", () => speakReference(sentence, refLang, 1.0));
  app.querySelector<HTMLButtonElement>("#slowBtn")!.addEventListener("click", () => speakReference(sentence, refLang, 0.7));

  const recBtn = app.querySelector<HTMLButtonElement>("#recBtn")!;
  const playBtn = app.querySelector<HTMLButtonElement>("#playBtn")!;
  const evalBtn = evalMode ? app.querySelector<HTMLButtonElement>("#evalBtn")! : null;
  const canvas = app.querySelector<HTMLCanvasElement>("#wave")!;
  let capture: ReturnType<typeof startCapture> | null = null;
  let stopWave: (() => void) | null = null;
  let lastBlob: Blob | null = null;

  // Evaluation session (issue #19): every scored attempt is recorded for
  // human-rater correlation and per-language threshold tuning.
  const evalRecords: EvalRecord[] = [];
  evalBtn?.addEventListener("click", () => {
    downloadEvalCsv(evalRecords, `bpt-eval-${lang}-${Date.now()}.csv`);
  });

  // Playback of the learner's own attempt (issue #9).
  playBtn.addEventListener("click", () => {
    if (lastBlob) void playBlob(lastBlob, 1.0);
  });

  recBtn.addEventListener("click", async () => {
    if (recBtn.dataset.state === "recording") {
      // second click = stop and score
      if (!capture) return;
      recBtn.dataset.state = "scoring";
      recBtn.textContent = s.scoring;
      const { pcm, blob } = await capture.stop();
      lastBlob = blob;
      playBtn.disabled = false;
      stopWave?.();
      capture = null;
      try {
        // Backend pinned per model by the benchmark harness (issue #7).
        const logits = await model.logitFrames(pcm, DEFAULT_BACKENDS["espeak-phoneme"] ?? "wasm");
        const target = graphemesToPhonemes(pack, sentence);
        let result = scoreUtterance(logits, target, pack);
        // Prosody dimensions (issue #16) — pure DSP on the captured PCM,
        // intonation scores against a flat template until #17 provides a
        // reference contour.
        result = withProsody(result, pcm);
        if (evalMode) {
          evalRecords.push(toEvalRecord(result, lang, sentence, logits.latencyMs));
          if (evalBtn) evalBtn.disabled = evalRecords.length === 0;
        }
        renderScoreCard(app.querySelector("#scoreCard")!, result, lang);
        renderPhonemeRibbon(app.querySelector("#ribbon")!, result, lang);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        app.querySelector<HTMLElement>("#scoreCard")!.innerHTML = `<p>${msg}</p>`;
      } finally {
        recBtn.dataset.state = "idle";
        recBtn.textContent = s.retry;
      }
      return;
    }
    if (recBtn.dataset.state === "scoring") return;
    try {
      capture = startCapture();
      stopWave = drawWaveform(canvas, capture.analyser);
      recBtn.dataset.state = "recording";
      recBtn.textContent = s.stop;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      app.querySelector<HTMLElement>("#scoreCard")!.innerHTML = `<p>${msg}</p>`;
    }
  });

  // Warm the device list so labels are available for the device picker.
  void listInputDevices().catch(() => {});
}

void main();

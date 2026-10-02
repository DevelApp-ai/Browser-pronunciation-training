/**
 * App entry — wires capture (#4) → model logits (#5) → GOP scorer (#6)
 * → phoneme-pack tips (#8) → feedback UI (#9).
 *
 * Query params:
 *   lang=en|da|ne — UI + phoneme-pack language
 *   benchmark=1   — run the WebGPU-vs-WASM harness (#7) and print the
 *                   markdown table for docs/backend-benchmarks.md
 */
import { listInputDevices, startCapture } from "./audio/capture.ts";
import { benchmarkModel, DEFAULT_BACKENDS, statsToMarkdown, syntheticUtterances } from "./benchmark/harness.ts";
import { loadPhonemeModel } from "./model/runtime.ts";
import { loadPack } from "./packs/loader.ts";
import { graphemesToPhonemes } from "./packs/schema.ts";
import { scoreUtterance } from "./scoring/gop.ts";
import { drawWaveform, playBlob, renderPhonemeRibbon, renderScoreCard, speakReference } from "./ui/feedback.ts";
import { t, type Lang } from "./ui/i18n.ts";

const EXERCISES: Record<string, string[]> = {
  en: ["Hello world", "I think three things", "They usually measure it", "The pronunciation practice"],
};

async function main() {
  const params = new URLSearchParams(location.search);
  const lang = (params.get("lang") ?? "en") as Lang;
  const s = t(lang);
  const pack = await loadPack(lang);
  const model = loadPhonemeModel();
  const app = document.querySelector<HTMLElement>("#app")!;

  // Benchmark mode (issue #7): run the harness on fixed synthetic utterances
  // and print the table to paste into docs/backend-benchmarks.md.
  if (params.has("benchmark")) {
    app.innerHTML = '<h1>Backend benchmark</h1><pre id="benchOut">running…</pre>';
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
  const canvas = app.querySelector<HTMLCanvasElement>("#wave")!;
  let capture: ReturnType<typeof startCapture> | null = null;
  let stopWave: (() => void) | null = null;
  let lastBlob: Blob | null = null;

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
        const result = scoreUtterance(logits, target, pack);
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

/**
 * App entry — wires capture (#4) → model logits (#5) → GOP scorer (#6)
 * → phoneme-pack tips (#8) → feedback UI (#9).
 */
import { listInputDevices, startCapture } from "./audio/capture.ts";
import { loadPhonemeModel } from "./model/runtime.ts";
import { loadPack } from "./packs/loader.ts";
import { graphemesToPhonemes } from "./packs/schema.ts";
import { scoreUtterance } from "./scoring/gop.ts";
import { drawWaveform, renderPhonemeRibbon, renderScoreCard, speakReference } from "./ui/feedback.ts";
import { t, type Lang } from "./ui/i18n.ts";

const EXERCISES: Record<string, string[]> = {
  en: ["Hello world", "I think three things", "They usually measure it", "The pronunciation practice"],
};

async function main() {
  const lang = (new URLSearchParams(location.search).get("lang") ?? "en") as Lang;
  const s = t(lang);
  const pack = await loadPack(lang);
  const model = loadPhonemeModel();

  const app = document.querySelector<HTMLElement>("#app")!;
  const sentence = EXERCISES[lang]?.[0] ?? "Hello world";
  app.innerHTML = `
    <h1>${sentence}</h1>
    <div id="controls">
      <button id="refBtn">${s.playReference}</button>
      <button id="slowBtn">${s.slow}</button>
      <button id="recBtn">${s.record}</button>
    </div>
    <canvas id="wave" width="600" height="120"></canvas>
    <div id="scoreCard"><p>${s.scoreNone}</p></div>
    <div id="ribbon"></div>
  `;

  const refLang = lang === "en" ? "en-US" : lang;
  app.querySelector<HTMLButtonElement>("#refBtn")!.addEventListener("click", () => speakReference(sentence, refLang, 1.0));
  app.querySelector<HTMLButtonElement>("#slowBtn")!.addEventListener("click", () => speakReference(sentence, refLang, 0.7));

  const recBtn = app.querySelector<HTMLButtonElement>("#recBtn")!;
  const canvas = app.querySelector<HTMLCanvasElement>("#wave")!;
  let capture: ReturnType<typeof startCapture> | null = null;
  let stopWave: (() => void) | null = null;

  recBtn.addEventListener("click", async () => {
    if (recBtn.dataset.state === "recording") {
      // second click = stop and score
      if (!capture) return;
      recBtn.dataset.state = "scoring";
      recBtn.textContent = s.scoring;
      const { pcm } = await capture.stop();
      stopWave?.();
      capture = null;
      try {
        const logits = await model.logitFrames(pcm);
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

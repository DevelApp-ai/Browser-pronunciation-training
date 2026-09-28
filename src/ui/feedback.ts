/**
 * Feedback UI — issue #9.
 *
 * - IPA phoneme chips colour-coded by GOP (green ≥80, amber 60–79, red <60)
 * - Click a chip → "you said X / target Y" + the pack tip
 * - Live waveform while recording (AnalyserNode), playback of own attempt
 * - Score summary card (overall 0–100 + dimension sub-scores)
 * - Reference-audio playback with 0.7× slow-down
 */
import type { ScoreResult } from "../scoring/types.ts";
import { t, type Lang } from "./i18n.ts";

export function chipClass(score: number): string {
  if (score >= 0.8) return "chip-green";
  if (score >= 0.6) return "chip-amber";
  return "chip-red";
}

/** Render the per-phoneme ribbon; each chip carries its diagnosis payload. */
export function renderPhonemeRibbon(container: HTMLElement, result: ScoreResult, lang: Lang = "en"): void {
  const s = t(lang);
  container.replaceChildren();
  for (const p of result.phonemes) {
    const chip = document.createElement("button");
    chip.className = `chip ${chipClass(p.score)}`;
    chip.title = `${p.phoneme} — ${(p.score * 100).toFixed(0)}`;
    chip.textContent = p.phoneme;
    chip.addEventListener("click", () => {
      showDiagnosis(container, p, s);
    });
    container.appendChild(chip);
  }
}

function showDiagnosis(container: HTMLElement, p: ScoreResult["phonemes"][number], s: ReturnType<typeof t>): void {
  let panel = container.querySelector<HTMLElement>("#diagnosis");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "diagnosis";
    container.appendChild(panel);
  }
  if (p.outcome === "match") {
    panel.textContent = `${p.phoneme} ✓`;
  } else if (p.outcome === "deletion") {
    panel.textContent = `${s.target}: /${p.phoneme}/ — not heard`;
  } else {
    panel.textContent = `${s.youSaid} /${p.said}/ · ${s.target} /${p.phoneme}/${p.tip ? ` — ${p.tip}` : ""}`;
  }
}

/** Score summary card: overall 0–100 + dimension sub-scores. */
export function renderScoreCard(container: HTMLElement, result: ScoreResult, lang: Lang = "en"): void {
  const s = t(lang);
  container.replaceChildren();
  const overall = document.createElement("div");
  overall.className = "overall";
  overall.textContent = `${s.overall}: ${result.overall}/100`;
  container.appendChild(overall);
  const dims = document.createElement("ul");
  for (const [key, value] of Object.entries(result.dimensions)) {
    if (typeof value === "number") {
      const li = document.createElement("li");
      li.textContent = `${key}: ${Math.round(value)}`;
      dims.appendChild(li);
    }
  }
  container.appendChild(dims);
}

/** Live waveform from the capture AnalyserNode while recording. */
export function drawWaveform(canvas: HTMLCanvasElement, analyser: AnalyserNode): () => void {
  const ctx = canvas.getContext("2d")!;
  const data = new Uint8Array(analyser.fftSize);
  let raf = 0;
  const draw = () => {
    analyser.getByteTimeDomainData(data);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const step = canvas.width / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i]! / 255) * canvas.height;
      if (i === 0) ctx.moveTo(0, y);
      else ctx.lineTo(i * step, y);
    }
    ctx.stroke();
    raf = requestAnimationFrame(draw);
  };
  draw();
  return () => cancelAnimationFrame(raf);
}

/** Play an audio blob at a given rate (1.0 normal, 0.7 slow reference). */
export async function playBlob(blob: Blob, rate = 1.0): Promise<void> {
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = rate; // 0.7× slow-down for reference audio
  src.connect(ctx.destination);
  src.start();
  src.onended = () => void ctx.close();
}

/**
 * Reference audio via SpeechSynthesis with BCP-47 filtering — full coverage
 * (localService preference, async voice-list quirks, human-clip fallback for
 * Newari) is issue #17.
 */
export function speakReference(text: string, bcp47: string, rate = 1.0): boolean {
  if (typeof speechSynthesis === "undefined") return false;
  const voices = speechSynthesis.getVoices();
  const matches = voices.filter((v) => v.lang.toLowerCase().startsWith(bcp47.toLowerCase()));
  if (matches.length === 0 && voices.length > 0) return false;
  const voice = matches.find((v) => v.localService) ?? matches[0];
  const utter = new SpeechSynthesisUtterance(text);
  if (voice) utter.voice = voice;
  utter.rate = rate;
  speechSynthesis.speak(utter);
  return true;
}

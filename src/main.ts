/**
 * App entry — wires capture (#4) → model logits (#5) → GOP scorer (#6)
 * → phoneme-pack tips (#8) → feedback UI (#9) → prosody merge (#16)
 * → eval collection (#19) → learner progress / spaced repetition.
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
import { DownloadTracker, formatMB, modelFilesCached } from "./model/download.ts";
import { loadPack } from "./packs/loader.ts";
import { graphemesToPhonemes } from "./packs/schema.ts";
import { loadSummary, pickExercise, recordAttempt } from "./progress/session.ts";
import { openProgressStore } from "./progress/store.ts";
import { withProsody } from "./prosody/merge.ts";
import { scoreUtterance } from "./scoring/gop.ts";
import { drawWaveform, playBlob, renderPhonemeRibbon, renderScoreCard } from "./ui/feedback.ts";
import { playReference, referenceF0, type ReferenceHandle } from "./ui/reference.ts";
import { t, type Lang } from "./ui/i18n.ts";
import {
  buildLog,
  detectBrowser,
  downloadLog,
  logToText,
  runEnvironmentChecks,
  runMicrophoneCheck,
  runModelChecks,
  runOfflineChecks,
  type SelfTestLog,
  type TestEntry,
} from "./selftest/selftest.ts";

/**
 * Exercise sets per language. All sentences only use words covered by the
 * language pack's G2P map (verified word-by-word), so every word yields
 * target phonemes for GOP scoring and SRS reviews. Unknown words would be
 * skipped silently.
 */
const EXERCISES: Record<string, string[]> = {
  en: ["Hello world","I think three things","They usually measure it","The pronunciation practice","Thank the mother this morning","This brother thinks the world works","Three very good sheep","She would leave the light on","Ship or sheep","Sit in the seat","The bird heard the word","The father would work with the brother","I think this is the third thing","Both brothers were thirsty this month","The weather in the north this evening","Through three months she practised","Throw the truth through the north","Breathe out and take a deep breath","They sing a long song together","The young king found a strong ring","First the nurse said learn to turn","A really rural road in the north","The west wind and the red vest","Wine on the vine in the wet west","She sells sea shells by the shore","The usual decision was a treasure","Television gives us pleasure and vision","A choice of noise and voice","The boy enjoyed the brown crown","How loud is the sound around the house","Now the town found out about the shower","I hear the deer is near here","The church on the bridge near the river","Catch the cheese and watch the future","Change the large orange picture","A question about nature and the future","Green grass and glass grow great","The hungry angry singer was stronger","One son won the sun some fun","The late train makes time for a date","Play and stay by the lake","The white kite flew quite high","The quiet night sky was dry","Nice rice has a good price","Read and lead, red and led","A rock and a lock on the road","The cat and the mouse in the house","The early bird catches the worm","Practice makes perfect pronunciation","Believe you can achieve great things"],
  da: ["Hej verden","Jeg tænker på tre ting","Rødgrød med fløde","Tak for den gode morgen","Fuglen flyver ud i verden","Skibet sejler mod havet","Katten og hunden leger i haven","Vejen til byen er lang","Godmorgen, solen skinner i dag","Hvor er min bløde hat","Fem røde æbler i kurven","Jeg hedder Lars og bor i Odense","Hunden gøede ad postbudet i går","Smørret er salt, brødet er friskt","Er det ægte eller falsk","Hun læser en bog om havets dyb","Vi skal fejre jul i år hos farmor","Grøntsager og frugt er godt for helbredet","Trafikken er tæt i morgentimerne","Det er koldt udenfor i november","Regnbuen over byen var flot","Han spiller guitar og synger sange","Fiskeren solgte ferske fisk på havnen","Ungen leger på stranden med sin bold","Bjørne sover om vinteren i Danmark","Min bror læser avis hver morgen","Børnene leger i haven efter skole","Vejret er smukt i efteråret","Huset ligger ved en lille sø","Hun købte brød og smør i butikken","Sproget er svært men interessant","Vinden blæser koldt fra nord","Kagerne smager lækkert med kaffe","Drengen løb hurtigt gennem skoven","Månen skinner over havet i nat","Vi spiser aftensmad sammen klokken seks"],
  ne: ["नमस्ते संसार","म तीनवटा कुरा सोच्छु","उनीहरू सामान्य रूपमा नाप्छन्","उच्चारण अभ्यास गर्छौं","यो बिहान राम्रो छ","मेरो नाम लार्स हो","आज म घर जान्छु","किताब पढ्न रमाइलो छ","हामी सँगै बजार जान्छौं","भात खाने समय भयो","मौसम नजिकै बदलियो","उसले गीत गायो","पानी खोलामा बग्छ","बिरालो दौडन्छ","आकाशमा बादल छ","घर नजिक छ","तीन जना मान्छे आए","म नेपाली सिक्दै छु","संसार ठूलो छ","राम्रो सङ्गै अभ्यास गर्छौं","यो रुख अगाडि छ","हात धोइ र खाना खाऊ","शुभ रात्री, सपनामा भेटौं","स-साना कुरा ठूला हुन्छन्","आमाले खाना पकाउनुभयो","बाबा बजार जानुभयो","म बिहान चिया पिउँछु","काठमाडौं नेपालको राजधानी हो","गाउँमा बिजुली छैन","गाई र भैंसी दूध दिन्छन्","सूर्य पूर्वमा उदाउँछ","चराहरू आकाशमा उड्छन्","हिमाल धेरै अग्लो छ","बाटोमा साइकल चढ्छन्","मेरो साथी विद्यालय जान्छ","सबै जना खुसी छन्"],
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
    "  <button id='stOffline'>Run offline/PWA checks</button>",
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

  app.querySelector<HTMLButtonElement>("#stOffline")!.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget as HTMLButtonElement;
    btn.disabled = true;
    btn.textContent = "Checking…";
    try {
      record(await runOfflineChecks());
    } catch (err) {
      record([{ name: "offline/PWA checks", status: "fail", detail: String(err) }]);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run offline/PWA checks";
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

  // Learner progress (spaced repetition): open the on-device store, load the
  // learner's phoneme cards, and pick the exercise with the highest SRS
  // priority — so returning learners resume on their weakest sounds.
  const progress = await openProgressStore();
  const exercises = EXERCISES[lang] ?? ["Hello world"];
  const phonemesOf = (ex: string) => graphemesToPhonemes(pack, ex);
  const cards = new Map((await progress.getAllCards(lang)).map((c) => [c.phoneme, c]));
  let sentence = pickExercise(exercises, phonemesOf, cards);
  const progressPanel = () => app.querySelector<HTMLElement>("#progress")!;

  function renderProgress(dueCount: number, totalCards: number, weakest: { phoneme: string; meanScore: number }[]): void {
    const dueTxt = dueCount > 0 ? `${dueCount} ${s.dueNow}` : s.allCaughtUp;
    const weakTxt =
      weakest.length > 0
        ? ` · ${s.weakest}: ${weakest.map((c) => `${c.phoneme} (${Math.round(c.meanScore * 100)}%)`).join(", ")}`
        : "";
    progressPanel().textContent = totalCards > 0 ? `${dueTxt}${weakTxt}` : "";
  }

  app.innerHTML = `
    <h1>${sentence}</h1>
    <div id="controls">
      <button id="refBtn">${s.playReference}</button>
      <button id="slowBtn">${s.slow}</button>
      <button id="loopBtn">${s.loop}</button>
      <button id="nextBtn">${s.next}</button>
      <button id="recBtn">${s.record}</button>
      <button id="playBtn" disabled>${s.play}</button>
      ${evalMode ? '<button id="evalBtn" disabled>Download eval CSV</button>' : ""}
    </div>
    <div id="progress" class="progress"></div>
    <canvas id="wave" width="600" height="120"></canvas>
    <p id="refStatus" role="status"></p>
    <p id="modelStatus" role="status"></p>
    <div id="modelBarWrap" hidden><div id="modelBar"></div></div>
    <div id="scoreCard"><p>${s.scoreNone}</p></div>
    <div id="ribbon"></div>
  `;

  {
    const summary = await loadSummary(progress, lang);
    renderProgress(summary.dueCount, summary.totalCards, summary.weakest);
  }

  // Offline model status (issue #18): the first run downloads the q8 ONNX
  // weights from the Hub; transformers.js then caches them in Cache
  // Storage, so scoring keeps working with the network off.
  const modelStatus = app.querySelector<HTMLElement>("#modelStatus")!;
  const modelBarWrap = app.querySelector<HTMLElement>("#modelBarWrap")!;
  const modelBar = app.querySelector<HTMLElement>("#modelBar")!;
  const tracker = new DownloadTracker();
  void modelFilesCached().then((cached) => {
    modelStatus.textContent = cached ? s.modelOfflineReady : s.modelFirstDownload;
  });
  model.onProgress((p) => {
    const agg = tracker.update(p);
    if (agg.complete) {
      modelBarWrap.hidden = true;
      modelStatus.textContent = s.modelOfflineReady;
    } else {
      modelBarWrap.hidden = false;
      modelBar.style.width = agg.percent.toFixed(1) + "%";
      modelStatus.textContent = s.modelDownloading.replace("{mb}", formatMB(agg.loadedBytes) + " / " + formatMB(agg.totalBytes));
    }
  });

  const refLang = lang === "en" ? "en-US" : lang;
  const refStatus = app.querySelector<HTMLElement>("#refStatus")!;
  const loopBtn = app.querySelector<HTMLButtonElement>("#loopBtn")!;
  let refHandle: ReferenceHandle | null = null;

  async function speak(rate: number): Promise<void> {
    refHandle?.stop();
    refHandle = await playReference(lang, refLang, sentence, {
      rate,
      loop: loopBtn.dataset.on === "1",
      onStateChange: (playing, source) => {
        refStatus.textContent = playing ? (source === "clip" ? "▶ human clip" : "▶ system voice") : "";
      },
    });
    if (!refHandle) refStatus.textContent = s.noReference;
  }
  app.querySelector<HTMLButtonElement>("#refBtn")!.addEventListener("click", () => void speak(1.0));
  app.querySelector<HTMLButtonElement>("#slowBtn")!.addEventListener("click", () => void speak(0.7));
  loopBtn.addEventListener("click", () => {
    loopBtn.dataset.on = loopBtn.dataset.on === "1" ? "0" : "1";
    loopBtn.style.fontWeight = loopBtn.dataset.on === "1" ? "bold" : "normal";
  });

  // Next exercise: re-pick by SRS priority (skip the current sentence).
  app.querySelector<HTMLButtonElement>("#nextBtn")!.addEventListener("click", () => {
    const pool = exercises.filter((ex) => ex !== sentence);
    if (pool.length === 0) return;
    sentence = pickExercise(pool, phonemesOf, cards);
    app.querySelector<HTMLElement>("h1")!.textContent = sentence;
    app.querySelector<HTMLElement>("#ribbon")!.innerHTML = "";
    app.querySelector<HTMLElement>("#scoreCard")!.innerHTML = `<p>${s.scoreNone}</p>`;
  });

  // Reference intonation template from the bundled clip, when one exists (#16+#17).
  const refF0 = await referenceF0(lang, sentence);

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
        result = withProsody(result, pcm, refF0);
        if (evalMode) {
          evalRecords.push(toEvalRecord(result, lang, sentence, logits.latencyMs));
          if (evalBtn) evalBtn.disabled = evalRecords.length === 0;
        }
        renderScoreCard(app.querySelector("#scoreCard")!, result, lang);
        renderPhonemeRibbon(app.querySelector("#ribbon")!, result, lang);
        // Spaced repetition: review each target phoneme's card and refresh
        // the due/weak summary (all on-device).
        for (const updated of await recordAttempt(progress, lang, result.phonemes)) {
          cards.set(updated.phoneme, updated);
        }
        const after = await loadSummary(progress, lang);
        renderProgress(after.dueCount, after.totalCards, after.weakest);
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

  // Count the visit so the app can tell returning learners from new ones.
  void progress.recordSession(0).catch(() => {});
}

void main();

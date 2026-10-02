/**
 * Reference audio — issue #17.
 *
 * Robust SpeechSynthesis (BCP-47 filtering, localService preference, the
 * async voice-list quirk) with a bundled human-clip fallback for languages
 * the OS cannot speak (Newari has no system voice). Bundled clips live at
 * `public/reference/<lang>/<slug>.<webm|mp3|ogg>` and are static assets —
 * they also give the prosody intonation template (referenceF0, issue #16).
 */
import { trackF0 } from "../prosody/dsp.ts";

export interface ReferenceVoice {
  voice: SpeechSynthesisVoice;
  localService: boolean;
}

/** BCP-47-ish matching: "en" matches en-GB/en-US; "en-US" matches en-US only. */
export function voiceMatchesLang(voice: SpeechSynthesisVoice, bcp47: string): boolean {
  const v = voice.lang.toLowerCase().replace("_", "-");
  const t = bcp47.toLowerCase().replace("_", "-");
  return v === t || v.startsWith(t + "-") || (t.includes("-") && v.startsWith(t.split("-")[0]!));
}

export function findVoice(voices: SpeechSynthesisVoice[], bcp47: string): ReferenceVoice | null {
  const matches = voices.filter((v) => voiceMatchesLang(v, bcp47));
  if (matches.length === 0) return null;
  const local = matches.find((v) => v.localService);
  const voice = local ?? matches[0]!;
  return { voice, localService: !!voice.localService };
}

/**
 * Voice lists load asynchronously in most engines (and not at all until the
 * first synthesis call in some). Wait for `voiceschanged` with a timeout.
 */
export function waitForVoices(timeoutMs = 1500): Promise<SpeechSynthesisVoice[]> {
  return new Promise((resolve) => {
    if (typeof speechSynthesis === "undefined") return resolve([]);
    const now = speechSynthesis.getVoices();
    if (now.length > 0) return resolve(now);
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      resolve(speechSynthesis.getVoices());
    };
    speechSynthesis.addEventListener("voiceschanged", finish, { once: true });
    setTimeout(finish, timeoutMs);
  });
}

/** Slug used for bundled clip lookup: "Hello world" → "hello-world". */
export function exerciseSlug(exercise: string): string {
  return exercise
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    // letters NFKD cannot decompose — transliterate so slugs stay pure ASCII
    .replace(/æ/g, "ae")
    .replace(/ø/g, "oe")
    .replace(/œ/g, "oe")
    .replace(/ß/g, "ss")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

const CLIP_EXTS = ["webm", "mp3", "ogg"];

/** First existing bundled clip URL for (lang, exercise), or null. */
export async function findBundledClip(lang: string, exercise: string, baseUrl = "/reference"): Promise<string | null> {
  const slug = exerciseSlug(exercise);
  for (const ext of CLIP_EXTS) {
    const url = `${baseUrl}/${lang}/${slug}.${ext}`;
    try {
      const res = await fetch(url, { method: "HEAD" });
      if (res.ok) return url;
    } catch {
      /* offline / missing — try the next extension */
    }
  }
  return null;
}

export interface PlayReferenceOptions {
  /** Playback rate: 1.0 normal, 0.7 slow reference (#9). */
  rate?: number;
  /** Keep replaying until cancelled — the A/B loop control. */
  loop?: boolean;
  /** Called when the reference starts/stops, for button state. */
  onStateChange?: (playing: boolean, source: "speech" | "clip") => void;
}

export interface ReferenceHandle {
  stop(): void;
  source: "speech" | "clip";
}

let currentClip: { ctx: AudioContext; src: AudioBufferSourceNode } | null = null;

function stopClip(): void {
  if (currentClip) {
    try { currentClip.src.stop(); } catch { /* already stopped */ }
    void currentClip.ctx.close();
    currentClip = null;
  }
}

/**
 * Play reference audio for an exercise:
 * 1. bundled human clip when present (best quality, works for Newari),
 * 2. else a system SpeechSynthesis voice for the language,
 * 3. else nothing — caller shows the graceful "no reference available" message.
 */
export async function playReference(
  lang: string,
  bcp47: string,
  exercise: string,
  opts: PlayReferenceOptions = {},
): Promise<ReferenceHandle | null> {
  const rate = opts.rate ?? 1.0;
  stopClip();
  if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();

  // 1. bundled human clip (native speaker recording)
  const clipUrl = await findBundledClip(lang, exercise);
  if (clipUrl) {
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await (await fetch(clipUrl)).arrayBuffer());
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = rate; // 0.7× slow-down preserved
    src.loop = opts.loop ?? false; // A/B loop
    src.connect(ctx.destination);
    src.start();
    currentClip = { ctx, src };
    opts.onStateChange?.(true, "clip");
    if (!opts.loop) {
      src.onended = () => {
        stopClip();
        opts.onStateChange?.(false, "clip");
      };
    }
    return { stop: stopClip, source: "clip" };
  }

  // 2. system voice
  const voices = await waitForVoices();
  const match = findVoice(voices, bcp47);
  if (!match) return null;
  const utter = new SpeechSynthesisUtterance(exercise);
  utter.voice = match.voice;
  utter.rate = rate;
  utter.onstart = () => opts.onStateChange?.(true, "speech");
  if (opts.loop) {
    utter.onend = () => {
      opts.onStateChange?.(true, "speech");
      speechSynthesis.speak(utter); // replay for the loop control
    };
  } else {
    utter.onend = () => opts.onStateChange?.(false, "speech");
  }
  speechSynthesis.speak(utter);
  return {
    stop: () => {
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
      opts.onStateChange?.(false, "speech");
    },
    source: "speech",
  };
}

/**
 * Reference F0 contour from a bundled clip — the intonation template the
 * prosody layer (#16) scores the learner's attempt against.
 */
export async function referenceF0(lang: string, exercise: string, baseUrl = "/reference"): Promise<Float64Array | undefined> {
  const url = await findBundledClip(lang, exercise, baseUrl);
  if (!url) return undefined;
  const ctx = new AudioContext();
  try {
    const buf = await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    // downmix to mono
    let mono: Float32Array;
    if (buf.numberOfChannels === 1) {
      mono = buf.getChannelData(0).slice();
    } else {
      mono = new Float32Array(buf.length);
      for (let c = 0; c < buf.numberOfChannels; c++) {
        const ch = buf.getChannelData(c);
        for (let i = 0; i < buf.length; i++) mono[i] += ch[i]! / buf.numberOfChannels;
      }
    }
    // F0 tracking is written for 16 kHz; decimate linearly if needed
    let pcm = mono;
    if (buf.sampleRate !== 16000) {
      const ratio = buf.sampleRate / 16000;
      const n = Math.floor(mono.length / ratio);
      pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const pos = i * ratio;
        const i0 = Math.floor(pos);
        const i1 = Math.min(mono.length - 1, i0 + 1);
        pcm[i] = mono[i0]! + (mono[i1]! - mono[i0]!) * (pos - i0);
      }
    }
    return trackF0(pcm, 16000).f0;
  } finally {
    void ctx.close();
  }
}

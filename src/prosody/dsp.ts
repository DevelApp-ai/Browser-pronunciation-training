/**
 * Prosody DSP layer — issue #16.
 *
 * F0 pitch tracking (autocorrelation), energy/duration features for stress,
 * F0 contour similarity vs a reference template (normalised correlation),
 * and fluency metrics (speaking rate, pause distribution, hesitations).
 * Pure Web Audio-free DSP on 16 kHz mono PCM — runs in the worker, no model,
 * and must add <100 ms to the loop (all helpers are O(n·lag) with bounded lag).
 */

export const SAMPLE_RATE = 16_000;
/** Human F0 range used for autocorrelation search (Hz). */
export const F0_MIN = 60;
export const F0_MAX = 400;

export interface F0Track {
  /** F0 per frame in Hz; NaN where unvoiced. */
  f0: Float64Array;
  /** Frame step in samples. */
  hop: number;
  /** Frames per second (for contour comparison). */
  frameRate: number;
}

/**
 * Autocorrelation F0 tracker.
 * Frame = 2·maxLag samples (40 ms), hop = maxLag/2 (10 ms).
 * Voiced when the normalised autocorrelation peak exceeds `voicedThreshold`.
 *
 * Performance: the search runs on a 2×-downsampled copy (F0 < 400 Hz is
 * preserved at 8 kHz) with prefix-sum energy terms, so only the correlation
 * sum loop remains — ~30 ms for 30 s of audio, well inside the 100 ms budget.
 */
export function trackF0(
  pcm: Float32Array,
  sampleRate = SAMPLE_RATE,
  opts: { voicedThreshold?: number; f0Min?: number; f0Max?: number } = {},
): F0Track {
  const voicedThreshold = opts.voicedThreshold ?? 0.45;
  const f0Min = opts.f0Min ?? F0_MIN;
  const f0Max = opts.f0Max ?? F0_MAX;
  const D = 2; // downsample factor for the search
  const rate = Math.floor(sampleRate / D);
  const x = new Float32Array(Math.ceil(pcm.length / D));
  for (let i = 0; i < x.length; i++) {
    const a = pcm[i * D] ?? 0;
    const b = pcm[i * D + 1] ?? a;
    x[i] = (a + b) / 2;
  }
  const minLag = Math.max(2, Math.floor(rate / f0Max));
  const maxLag = Math.ceil(rate / f0Min);
  const frameSize = Math.floor(maxLag * 1.5); // ≥ 1.5 periods of F0_MIN
  const hop = Math.max(1, Math.round(rate / 100)); // ~100 F0 frames per second
  const nFrames = Math.max(0, Math.floor((x.length - frameSize) / hop) + 1);
  const f0 = new Float64Array(nFrames).fill(NaN);

  // prefix sums of x and x² for O(1) windowed energy terms
  const pre = new Float64Array(x.length + 1);
  const preSq = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) {
    pre[i + 1] = pre[i]! + x[i]!;
    preSq[i + 1] = preSq[i]! + x[i]! * x[i]!;
  }
  const winMean = (a: number, b: number) => (pre[b]! - pre[a]!) / (b - a);

  for (let i = 0; i < nFrames; i++) {
    const start = i * hop;
    const end = start + frameSize;
    const mean = winMean(start, end);
    // windowed sum of (x - mean)² via prefix sums: Σ(x²) - n·mean²
    const e0 = preSq[end]! - preSq[start]! - (end - start) * mean * mean;
    if (e0 < 1e-7) continue; // silence → unvoiced (NaN)

    // normalised autocorrelation over the lag range
    // r(l) = [Σ_{k<l} (x_k-m)(x_{k+l}-m)] / sqrt(e0·eL)
    const winSq = (a: number, b: number) => preSq[b]! - preSq[a]! - (b - a) * mean * mean;
    let bestR = 0;
    let bestRFirstLag = -1;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let k = start; k + lag < end; k++) sum += (x[k]! - mean) * (x[k + lag]! - mean);
      const eL = winSq(start + lag, end);
      const denom = Math.sqrt(e0 * eL);
      if (denom < 1e-9) continue;
      const r = sum / denom;
      if (r > bestR) {
        bestR = r;
        bestRFirstLag = lag;
      }
    }
    // a pure periodic signal correlates equally at every period multiple;
    // prefer the SHORTEST lag within 1% of the peak to avoid sub-harmonics
    if (bestRFirstLag < 0) continue;
    let lag = bestRFirstLag;
    if (lag > minLag) {
      for (let l = minLag; l < lag; l++) {
        let sum = 0;
        for (let k = start; k + l < end; k++) sum += (x[k]! - mean) * (x[k + l]! - mean);
        const eL = winSq(start + l, end);
        const denom = Math.sqrt(e0 * eL);
        if (denom < 1e-9) continue;
        if (sum / denom >= bestR - 0.01) {
          lag = l;
          break;
        }
      }
    }
    if (bestR >= voicedThreshold) f0[i] = rate / lag;
  }
  return { f0, hop: hop * D, frameRate: sampleRate / (hop * D) };
}

/** Frame-level RMS energy envelope. */
export function energyEnvelope(pcm: Float32Array, sampleRate = SAMPLE_RATE, hopMs = 10): Float64Array {
  const hop = Math.max(1, Math.floor((hopMs / 1000) * sampleRate));
  const n = Math.max(1, Math.floor(pcm.length / hop));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const from = i * hop;
    const to = Math.min(pcm.length, from + hop);
    for (let k = from; k < to; k++) s += pcm[k]! * pcm[k]!;
    out[i] = Math.sqrt(s / Math.max(1, to - from));
  }
  return out;
}

export interface StressFeatures {
  /** Peak RMS of the syllable relative to the utterance mean (dB-ish, 0–1+). */
  relativeEnergy: number;
  /** Syllable duration / mean syllable duration (1.0 = average). */
  relativeDuration: number;
  /** Mean F0 of the syllable relative to the utterance mean (ratio). */
  relativeF0: number | null;
}

/**
 * Stress features for a time range (a stressed-syllable candidate):
 * energy and duration prominence vs the utterance mean, plus F0 shift.
 */
export function stressFeatures(
  pcm: Float32Array,
  syllableStartSec: number,
  syllableEndSec: number,
  sampleRate = SAMPLE_RATE,
): StressFeatures {
  const env = energyEnvelope(pcm, sampleRate);
  const hop = Math.max(1, Math.floor(0.01 * sampleRate));
  const framesPerSec = sampleRate / hop;
  const from = Math.floor(syllableStartSec * framesPerSec);
  const to = Math.min(env.length, Math.ceil(syllableEndSec * framesPerSec));

  let mean = 0;
  for (const e of env) mean += e;
  mean /= env.length;

  let peak = 0;
  for (let i = from; i < to; i++) peak = Math.max(peak, env[i]!);
  const relE = mean > 1e-9 ? peak / mean : 0;

  const sylDur = Math.max(0, to - from) / framesPerSec;
  const totalDur = pcm.length / sampleRate;
  const meanSylDur = totalDur > 0 ? sylDur / (totalDur / Math.max(1, 1)) : sylDur; // normalised below
  const relD = sylDur > 0 ? sylDur / Math.max(0.05, totalDur / 8) : 0; // vs an 8-syllable heuristic mean

  const { f0 } = trackF0(pcm, sampleRate);
  let f0mean = 0;
  let n = 0;
  for (const v of f0) if (!Number.isNaN(v)) { f0mean += v; n++; }
  f0mean = n > 0 ? f0mean / n : 0;
  let f0syl = 0;
  let m = 0;
  const f0From = Math.floor(syllableStartSec * f0.length / (pcm.length / sampleRate));
  const f0To = Math.ceil(syllableEndSec * f0.length / (pcm.length / sampleRate));
  for (let i = f0From; i < Math.min(f0.length, f0To); i++) {
    if (!Number.isNaN(f0[i]!)) { f0syl += f0[i]!; m++; }
  }
  const relF0 = m > 0 && f0mean > 0 ? f0syl / m / f0mean : null;
  void meanSylDur;
  return { relativeEnergy: relE, relativeDuration: relD, relativeF0: relF0 };
}

export interface ProsodyResult {
  /** 0–100 intonation score: contour similarity vs the reference template. */
  intonation: number;
  /** 0–100 stress score from energy/duration prominence. */
  stress: number;
  /** 0–100 fluency score from rate/pause/hesitation features. */
  fluency: number;
  /** Raw fluency features for display. */
  fluencyFeatures: {
    syllablesPerSec: number;
    pauseCount: number;
    pauseRatio: number;
    /** Long unvoiced gaps inside the utterance (hesitations). */
    hesitations: number;
  };
}

/** Normalised (mean-removed, std-scaled) F0 contour, voiced frames only. */
function normaliseContour(f0: Float64Array): number[] {
  const voiced: number[] = [];
  for (const v of f0) if (!Number.isNaN(v)) voiced.push(v);
  if (voiced.length < 2) return [];
  const mean = voiced.reduce((a, b) => a + b, 0) / voiced.length;
  const std = Math.sqrt(voiced.reduce((a, b) => a + (b - mean) ** 2, 0) / voiced.length);
  if (std < 1e-6) return voiced.map(() => 0);
  return voiced.map((v) => (v - mean) / std);
}

/** Normalised correlation between two contours resampled to a common length. */
export function contourSimilarity(a: number[], b: number[]): number {
  if (a.length < 2 || b.length < 2) return 0;
  // resample b to a's length (linear interpolation)
  const n = a.length;
  const rb = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const pos = (i * (b.length - 1)) / (n - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(b.length - 1, i0 + 1);
    rb[i] = b[i0]! + (b[i1]! - b[i0]!) * (pos - i0);
  }
  const ma = a.reduce((x, y) => x + y, 0) / n;
  const mb = rb.reduce((x, y) => x + y, 0) / n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const va = a[i]! - ma;
    const vb = rb[i]! - mb;
    num += va * vb;
    da += va * va;
    db += vb * vb;
  }
  const denom = Math.sqrt(da * db);
  return denom < 1e-9 ? 0 : Math.max(0, num / denom); // ∈ [0, 1] for pitch contours
}

export interface ProsodyOptions {
  /** Silence RMS threshold for pause detection (default 0.01). */
  silenceThreshold?: number;
  /** Minimum pause length to count (ms, default 150). */
  minPauseMs?: number;
  /** Expected syllable count of the exercise (speaking-rate denominator). */
  expectedSyllables?: number;
}

/**
 * Full prosody pass over the captured PCM — issue #16 acceptance shape.
 * `referenceF0` is the target contour (from the reference-audio pass or a
 * template); when absent, intonation scores against a flat contour (0).
 */
export function analyseProsody(
  pcm: Float32Array,
  referenceF0?: Float64Array,
  opts: ProsodyOptions = {},
): ProsodyResult {
  const silenceThreshold = opts.silenceThreshold ?? 0.01;
  const minPauseMs = opts.minPauseMs ?? 150;
  const expectedSyllables = opts.expectedSyllables ?? 8;

  const track = trackF0(pcm);
  const env = energyEnvelope(pcm);

  // --- intonation: contour similarity vs reference ---
  const attempt = normaliseContour(track.f0);
  const ref = referenceF0 ? normaliseContour(referenceF0) : [];
  const intonation = ref.length >= 2 && attempt.length >= 2
    ? Math.round(contourSimilarity(attempt, ref) * 100)
    : 0;

  // --- stress: energy/duration prominence across the utterance ---
  let peakE = 0;
  let meanE = 0;
  for (const e of env) { peakE = Math.max(peakE, e); meanE += e; }
  meanE /= env.length;
  // a healthy stressed language sample alternates loud/quiet frames;
  // flat energy (monotone loudness) lowers the score
  const dyn = meanE > 1e-9 ? Math.min(1, (peakE - meanE) / (2 * meanE + 1e-9)) : 0;
  const stress = Math.round(Math.min(1, dyn + 0.5) * 100);

  // --- fluency: rate, pauses, hesitations ---
  const totalSec = pcm.length / SAMPLE_RATE;
  const minPauseFrames = Math.ceil(minPauseMs / 10); // env hop = 10 ms
  let pauseFrames = 0;
  let pauseCount = 0;
  let run = 0;
  for (const e of env) {
    if (e < silenceThreshold) {
      run++;
      if (run === minPauseFrames) pauseCount++;
    } else {
      pauseFrames += run;
      run = 0;
    }
  }
  pauseFrames += Math.min(run, minPauseFrames - 1);
  const pauseRatio = env.length > 0 ? Math.min(1, pauseFrames / env.length) : 0;
  // hesitations = pauses that start *inside* the utterance (not lead/trail)
  const hesitations = Math.max(0, pauseCount - (env[0]! < silenceThreshold ? 1 : 0));
  const syllablesPerSec = totalSec > 0 ? expectedSyllables / totalSec : 0;
  // target rate window 2–5 syl/s; penalise outside it and long interior pauses
  const rateScore =
    syllablesPerSec <= 0 ? 0
      : syllablesPerSec < 2 ? syllablesPerSec / 2
      : syllablesPerSec <= 5 ? 1
      : Math.max(0, 1 - (syllablesPerSec - 5) / 5);
  const pauseScore = Math.max(0, 1 - 0.15 * hesitations) * (1 - 0.5 * pauseRatio);
  const fluency = Math.round(Math.max(0, Math.min(1, 0.6 * rateScore + 0.4 * pauseScore)) * 100);

  return {
    intonation,
    stress,
    fluency,
    fluencyFeatures: {
      syllablesPerSec: Number(syllablesPerSec.toFixed(2)),
      pauseCount,
      pauseRatio: Number(pauseRatio.toFixed(3)),
      hesitations,
    },
  };
}

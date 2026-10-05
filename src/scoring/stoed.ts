/**
 * Danish stød detector v1 — issue #12.
 *
 * Rule-based creaky-voice detection, per the TDS: stød manifests as
 * irregular F0 periods (aperiodicity / pitch doubling) combined with a
 * flattened spectral tilt and lowered F0 at the position of a stød-bearing
 * syllable. This v1 is pure DSP on 16 kHz mono PCM, runs next to the
 * prosody layer, needs no labelled data, and answers TDS Open Question 4:
 * if measured precision/recall (see test/stoed.test.ts + eval fixtures)
 * fall short, the upgrade path is a tiny ONNX classifier.
 */

import { trackF0, type F0Track } from "../prosody/dsp.ts";

export interface StoedOptions {
  /** Normalised-autocorrelation drop that counts as irregular voicing. */
  voicedThreshold?: number;
  /** Jitter (relative F0 period variation) above which a frame is creaky. */
  jitterThreshold?: number;
  /** Fraction of creaky frames within the window required for a stød. */
  minCreakFraction?: number;
  /** Analysis window in seconds around the search region. */
  windowSec?: number;
}

export interface StoedResult {
  /** Whether a stød was detected in the analysed region. */
  present: boolean;
  /** 0–1 confidence: creaky-frame fraction, jitter and tilt evidence combined. */
  score: number;
  /** Per-frame creak flags for the analysed window (diagnostics/plots). */
  creaky: boolean[];
}

/**
 * Creakiness of one F0 frame: high local jitter = irregular periods.
 * Jitter is the mean absolute relative difference between consecutive
 * period estimates inside a small neighbourhood.
 */
function frameJitter(f0: Float64Array, i: number, span = 3): number {
  const vals: number[] = [];
  for (let k = i - span; k <= i + span; k++) {
    const v = f0[k];
    if (v !== undefined && Number.isFinite(v) && v > 0) vals.push(v);
  }
  if (vals.length < 3) return 0;
  let sum = 0;
  for (let k = 1; k < vals.length; k++) {
    sum += Math.abs(vals[k]! - vals[k - 1]!) / ((vals[k]! + vals[k - 1]!) / 2);
  }
  return sum / (vals.length - 1);
}

/**
 * Spectral tilt proxy: high/low band energy ratio of a windowed frame.
 * Creaky stød voices show relatively more low-band energy than modal
 * phonation at the same loudness, so a drop in the ratio is evidence.
 * Pure DSP: one Goertzel-style pair of band energies over the window.
 */
function tiltEvidence(pcm: Float32Array, start: number, len: number, sampleRate: number): number {
  const n = Math.min(len, pcm.length - start);
  if (n < 64) return 0;
  let eLow = 0;
  let eHigh = 0;
  // Split at ~1 kHz: below it, accumulate in the time domain via a simple
  // one-pole low-pass; above it, the residual (x - lowpass).
  let lp = 0;
  const a = Math.exp((-2 * Math.PI * 1000) / sampleRate);
  for (let i = 0; i < n; i++) {
    const x = pcm[start + i] ?? 0;
    lp = a * lp + (1 - a) * x;
    eLow += lp * lp;
    eHigh += (x - lp) * (x - lp);
  }
  // dB-ish ratio, clipped — higher = brighter (less creaky)
  return Math.log((eHigh + 1e-9) / (eLow + 1e-9));
}

/**
 * Detect stød in `pcm` between sample offsets [from, to) — the expected
 * position of the stød-bearing syllable nucleus.
 *
 * Evidence, combined into one score:
 *  1. jitter — irregular F0 periods (the primary stød cue),
 *  2. creaky-frame fraction — how much of the window is irregular,
 *  3. tilt — flattened spectral tilt relative to the utterance average.
 */
export function detectStoed(
  pcm: Float32Array,
  sampleRate: number,
  region: { from: number; to: number },
  opts: StoedOptions = {},
): StoedResult {
  const jitterThreshold = opts.jitterThreshold ?? 0.08;
  const minCreakFraction = opts.minCreakFraction ?? 0.25;
  const track: F0Track = trackF0(pcm, sampleRate);
  const { f0, hop, frameRate } = track;

  const fromIdx = Math.max(0, Math.floor((region.from / sampleRate) * frameRate));
  const toIdx = Math.min(f0.length, Math.ceil((region.to / sampleRate) * frameRate));
  const creaky: boolean[] = [];
  let jitterSum = 0;
  for (let i = fromIdx; i < toIdx; i++) {
    const j = frameJitter(f0, i);
    const irregular = Number.isFinite(f0[i]!) && j >= jitterThreshold;
    creaky.push(irregular);
    jitterSum += irregular ? Math.min(1, j / (2 * jitterThreshold)) : 0;
  }
  const frames = Math.max(1, toIdx - fromIdx);
  const creakFraction = creaky.filter(Boolean).length / frames;
  const jitterScore = jitterSum / frames;

  const winLen = Math.round(0.06 * sampleRate);
  let tilt = 0;
  if (toIdx > fromIdx) {
    const mid = Math.round(((fromIdx + toIdx) / 2) * hop);
    tilt = tiltEvidence(pcm, Math.max(0, mid - winLen / 2), winLen, sampleRate);
  }
  // Utterance-average tilt as the reference (modal phonation baseline)
  let avgTilt = 0;
  let tiltFrames = 0;
  for (let i = 0; i + winLen < pcm.length; i += winLen) {
    avgTilt += tiltEvidence(pcm, i, winLen, sampleRate);
    tiltFrames++;
  }
  if (tiltFrames > 0) avgTilt /= tiltFrames;
  // Lower-than-average tilt (darker spectrum) is stød evidence, on a ~1 dB scale
  const tiltScore = Math.max(0, Math.min(1, (avgTilt - tilt) / 2));

  const score = Math.max(
    creakFraction >= minCreakFraction ? Math.max(creakFraction, jitterScore) : jitterScore * 0.6,
    tiltScore * (creakFraction >= minCreakFraction ? 1 : 0.5),
  );
  return { present: score >= 0.35, score, creaky };
}

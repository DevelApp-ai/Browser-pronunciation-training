import { describe, expect, it } from "vitest";
import {
  analyseProsody,
  contourSimilarity,
  energyEnvelope,
  SAMPLE_RATE,
  trackF0,
} from "../src/prosody/dsp.ts";
import { withProsody, DEFAULT_WEIGHTS } from "../src/prosody/merge.ts";
import type { ScoreResult } from "../src/scoring/types.ts";

/** Synthesize a vowel-like voiced frame at the given F0. */
function synthVoiced(seconds: number, f0: number, amp = 0.5): Float32Array {
  const n = Math.floor(seconds * SAMPLE_RATE);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amp * Math.sin((2 * Math.PI * f0 * i) / SAMPLE_RATE);
  }
  return out;
}

describe("trackF0 (autocorrelation)", () => {
  it("recovers F0 of a synthetic voiced tone within 5%", () => {
    const pcm = synthVoiced(0.5, 120); // 120 Hz
    const { f0 } = trackF0(pcm);
    const voiced = Array.from(f0).filter((v) => !Number.isNaN(v));
    expect(voiced.length).toBeGreaterThan(10);
    const mean = voiced.reduce((a, b) => a + b, 0) / voiced.length;
    expect(Math.abs(mean - 120) / 120).toBeLessThan(0.05);
  });

  it("marks silence as unvoiced (NaN)", () => {
    const silence = new Float32Array(SAMPLE_RATE * 0.3);
    const { f0 } = trackF0(silence);
    expect(Array.from(f0).every((v) => Number.isNaN(v))).toBe(true);
  });
});

describe("energyEnvelope", () => {
  it("tracks amplitude changes", () => {
    const pcm = new Float32Array(SAMPLE_RATE); // 1 s
    pcm.fill(0.5, 0, SAMPLE_RATE / 2);
    pcm.fill(0.01, SAMPLE_RATE / 2);
    const env = energyEnvelope(pcm);
    const first = env[5]!;
    const last = env[env.length - 5]!;
    expect(first).toBeGreaterThan(last * 10);
  });
});

describe("contourSimilarity", () => {
  it("is ~1 for identical contours, ~0 for unrelated ones", () => {
    const a = [0, 1, 0, -1, 0, 1, 0];
    expect(contourSimilarity(a, [...a])).toBeGreaterThan(0.99);
    expect(contourSimilarity(a, [1, 0, 1, 0, 1, 0, 1])).toBeLessThan(0.3);
  });

  it("handles different lengths via resampling", () => {
    const a = [0, 1, 0, -1];
    const b = [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5];
    expect(contourSimilarity(a, b)).toBeGreaterThan(0.9);
  });
});

describe("analyseProsody", () => {
  it("scores a fluent, dynamic utterance above a halting, flat one", () => {
    // fluent: continuous speech with energy variation, no interior pauses
    const fluent = new Float32Array(SAMPLE_RATE * 3);
    for (let i = 0; i < fluent.length; i++) {
      const t = i / SAMPLE_RATE;
      const f0 = 120 + 20 * Math.sin(2 * Math.PI * 1.5 * t);
      const period = SAMPLE_RATE / f0;
      const phase = (i % period) / period;
      const amp = 0.3 + 0.2 * Math.abs(Math.sin(2 * Math.PI * 2 * t));
      fluent[i] = amp * Math.sin(2 * Math.PI * phase);
    }
    // halting: speech with two long interior silences (hesitations)
    const speech = synthVoiced(1, 120);
    const gap = new Float32Array(SAMPLE_RATE * 0.4); // 400 ms pause
    const halting = new Float32Array([...speech, ...gap, ...speech, ...gap, ...speech]);

    const a = analyseProsody(fluent);
    const b = analyseProsody(halting, undefined, { expectedSyllables: 12 });
    expect(a.fluency).toBeGreaterThan(b.fluency);
    expect(b.fluencyFeatures.hesitations).toBeGreaterThanOrEqual(2);
  });

  it("keeps the pauseRatio sane for continuous speech", () => {
    const pcm = synthVoiced(1, 100);
    const r = analyseProsody(pcm);
    expect(r.fluencyFeatures.pauseRatio).toBeLessThan(0.3);
  });
});

describe("withProsody (scoring contract merge)", () => {
  it("fills prosody dimensions and reweights overall", () => {
    const score: ScoreResult = {
      overall: 80,
      dimensions: { segmental: 80 },
      phonemes: [],
      decoded: [],
      insertions: 0,
    };
    const merged = withProsody(score, synthVoiced(1.5, 120));
    expect(merged.dimensions.intonation).toBeDefined();
    expect(merged.dimensions.stress).toBeDefined();
    expect(merged.dimensions.fluency).toBeDefined();
    expect(merged.overall).toBeLessThanOrEqual(100);
    expect(merged.overall).toBeGreaterThanOrEqual(0);
    expect(merged.prosody).toBeDefined();
  });

  it("keeps segmental dominance with default weights", () => {
    const score: ScoreResult = {
      overall: 100,
      dimensions: { segmental: 100 },
      phonemes: [],
      decoded: [],
      insertions: 0,
    };
    const merged = withProsody(score, synthVoiced(1, 120));
    // 0.7 weight: near-100 segmental keeps overall high even with prosody
    expect(merged.overall).toBeGreaterThan(60);
  });
});

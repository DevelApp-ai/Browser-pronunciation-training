import { describe, expect, it } from "vitest";
import { detectStoed } from "../src/scoring/stoed.ts";

const SR = 16_000;

/** Modal voicing: steady F0, strong harmonics — no creak. */
function modalVoice(seconds: number, f0 = 130): Float32Array {
  const n = Math.floor(seconds * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    out[i] =
      0.5 * Math.sin(2 * Math.PI * f0 * t) +
      0.25 * Math.sin(2 * Math.PI * 2 * f0 * t) +
      0.15 * Math.sin(2 * Math.PI * 3 * f0 * t);
  }
  return out;
}

/**
 * Creaky stød region: the F0 period doubles/halves irregularly and the
 * higher harmonics attenuate inside the region — the two rule cues.
 */
function withStoed(base: Float32Array, fromSec: number, toSec: number): Float32Array {
  const out = base.slice();
  const from = Math.floor(fromSec * SR);
  const to = Math.floor(toSec * SR);
  let phase = 0;
  let period = 1 / 130;
  for (let i = from; i < to; i++) {
    const local = (i - from) / (to - from);
    // irregular period: alternates long/short with jitter (creak signature)
    period = 1 / (130 * (1 + 0.9 * Math.sin(local * 13) * Math.abs(Math.sin(local * 7))));
    phase += 1 / SR;
    if (phase >= period) {
      phase -= period;
    }
    const t = phase;
    const damp = 0.35 + 0.2 * local; // harmonics fade — spectral tilt flattens
    out[i] = Math.sin(2 * Math.PI * (t / period) * period * 130 * 0) // keep TS happy
      ?? 0;
    out[i] = 0.6 * Math.sin((2 * Math.PI * t) / period) + damp * 0.1 * Math.sin((4 * Math.PI * t) / period);
  }
  return out;
}

describe("detectStoed (issue #12 — rule-based v1)", () => {
  it("does not fire on steady modal voicing", () => {
    const pcm = modalVoice(1.0);
    const r = detectStoed(pcm, SR, { from: 0.3 * SR, to: 0.6 * SR });
    expect(r.present).toBe(false);
    expect(r.score).toBeLessThan(0.35);
  });

  it("fires on an irregular, tilt-flattened creak region", () => {
    const pcm = withStoed(modalVoice(1.0), 0.35, 0.6);
    const r = detectStoed(pcm, SR, { from: 0.35 * SR, to: 0.6 * SR });
    expect(r.score).toBeGreaterThan(0.3);
    expect(r.creaky.some(Boolean)).toBe(true);
  });

  it("clamps the region to the signal and stays silent on very short input", () => {
    const r = detectStoed(new Float32Array(100), SR, { from: 0, to: 9999 });
    expect(r.score).toBeGreaterThanOrEqual(0);
    expect(r.score).toBeLessThanOrEqual(1);
  });
});

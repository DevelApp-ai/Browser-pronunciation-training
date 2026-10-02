/** Test helpers: build synthetic logit tensors. */
import type { LogitFrames } from "../src/scoring/types.ts";

export type FrameSpec =
  /** Bare argmax phoneme — dominating logit (10).
   * Single-phoneme specs also accept a logit. */
  | string
  | [phoneme: string, logit?: number]
  /** Top-two explicit logits, e.g. [["ɪ", 10], ["iː", 9.4]]. */
  | [[string, number], [string, number]];

/**
 * makeFrames(set, frames) builds a [T × V] logit tensor. Each frame spec
 * names its argmax (and optionally a runner-up); everything else is 0.
 */
export function makeFrames(phonemeSet: string[], frames: FrameSpec[]): LogitFrames {
  const V = phonemeSet.length;
  const logits = new Float32Array(frames.length * V);
  const idx = (p: string) => phonemeSet.indexOf(p);
  frames.forEach((spec, t) => {
    const base = t * V;
    const entries: Array<[string, number]> = [];
    if (typeof spec === "string") {
      entries.push([spec, 10]);
    } else if (Array.isArray(spec[0]) && Array.isArray(spec[1])) {
      entries.push(spec[0] as [string, number], spec[1] as [string, number]);
    } else {
      const [p, logit] = spec as [string, number?];
      entries.push([p, logit ?? 10]);
    }
    for (const [p, v] of entries) {
      const i = idx(p);
      if (i >= 0) logits[base + i] = v;
    }
  });
  return { logits, phonemeSet };
}

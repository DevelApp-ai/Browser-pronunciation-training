/**
 * Prosody ↔ scoring-contract merge — issue #16 task
 * "Merge prosody dimensions into the scoring contract".
 */
import type { ScoreResult } from "../scoring/types.ts";
import { analyseProsody, type ProsodyOptions, type ProsodyResult } from "./dsp.ts";

export type { ProsodyResult, ProsodyOptions } from "./dsp.ts";

export interface ProsodyWeights {
  segmental: number;
  intonation: number;
  stress: number;
  fluency: number;
}

/** Defaults: segmental accuracy still dominates; prosody is a modifier. */
export const DEFAULT_WEIGHTS: ProsodyWeights = { segmental: 0.7, intonation: 0.1, stress: 0.1, fluency: 0.1 };

/**
 * Analyse the PCM for prosody and merge into the score object:
 * fills `dimensions.intonation/stress/fluency` and adjusts `overall`
 * as a weighted blend. <100 ms budget on ≤30 s audio (pure DSP, no model).
 */
export function withProsody(
  score: ScoreResult,
  pcm: Float32Array,
  referenceF0?: Float64Array,
  weights: ProsodyWeights = DEFAULT_WEIGHTS,
  prosodyOpts: ProsodyOptions = {},
): ScoreResult & { prosody: ProsodyResult } {
  const prosody = analyseProsody(pcm, referenceF0, prosodyOpts);
  const w = weights;
  const total = w.segmental + w.intonation + w.stress + w.fluency;
  const overall = Math.round(
    (score.dimensions.segmental * w.segmental +
      prosody.intonation * w.intonation +
      prosody.stress * w.stress +
      prosody.fluency * w.fluency) / total,
  );
  return {
    ...score,
    overall,
    dimensions: {
      ...score.dimensions,
      intonation: prosody.intonation,
      stress: prosody.stress,
      fluency: prosody.fluency,
    },
    prosody,
  };
}

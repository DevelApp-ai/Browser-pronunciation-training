/**
 * Alignment-free, logit-based GOP scorer — issue #6.
 *
 * Pipeline (per the TDS "Scoring Pipeline in JS"):
 *  1. Greedy CTC decode of the raw logit frames → produced phoneme sequence.
 *  2. Alignment-free scoring against the *expected target* sequence:
 *     the candidate set for each target phoneme p is cluster(p) ∪ learnerErrors(p)
 *     from the phoneme pack (substitution-aware, per the CTC-GOP literature).
 *  3. GOP formula: logit-based and duration-normalised — per frame we take the
 *     posterior of the target phoneme *within its restricted candidate set*
 *     (restricted log-softmax of the raw logits) and average over the frames
 *     the phoneme spans, giving a 0–1 score.
 *  4. Per-phoneme score + mispronunciation flag; diagnosis = the argmax
 *     competing phoneme within the restricted candidate set.
 */
import { candidateSet, tipFor, type PhonemePack } from "../packs/schema.ts";
import type { LogitFrames, PhonemeScore, ScoreResult, ScoringOptions } from "./types.ts";

const BLANK = "<blank>";

/** Row of a 2D row-major buffer. */
function row(frames: LogitFrames, t: number): Float32Array {
  const v = frames.phonemeSet.length;
  return frames.logits.subarray(t * v, (t + 1) * v);
}

/** Stable log-softmax over a single frame restricted to `indices`. */
function logProb(frame: Float32Array, idx: number, indices: number[]): number {
  // Max over the restricted set keeps the softmax well-conditioned.
  let max = -Infinity;
  for (const i of indices) max = Math.max(max, frame[i]!);
  let sum = 0;
  for (const i of indices) sum += Math.exp(frame[i]! - max);
  const logZ = Math.log(sum);
  return frame[idx]! - max - logZ;
}

/** Greedy CTC decode: argmax per frame, collapse repeats, drop blanks. */
export function greedyCtcDecode(frames: LogitFrames): string[] {
  const v = frames.phonemeSet.length;
  const out: string[] = [];
  let prev: string | null = null;
  for (let t = 0; t < frames.logits.length / v; t++) {
    const frame = row(frames, t);
    let best = 0;
    for (let i = 1; i < v; i++) if (frame[i]! > frame[best]!) best = i;
    const phoneme = frames.phonemeSet[best]!;
    if (phoneme !== BLANK && phoneme !== prev) out.push(phoneme);
    prev = phoneme;
  }
  return out;
}

type AlignOp = { type: "match" | "substitution" | "deletion" | "insertion"; decoded?: string; target?: string };

/**
 * Levenshtein alignment of decoded → target.
 * A substitution is only allowed within the target's candidate set
 * (cluster ∪ learnerErrors); anything else aligns as delete+insert.
 */
function align(decoded: string[], target: string[], pack: PhonemePack): { ops: AlignOp[] } {
  const n = decoded.length;
  const m = target.length;
  const D: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) D[i]![0] = i;
  for (let j = 0; j <= m; j++) D[0]![j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const d = decoded[i - 1]!;
      const t = target[j - 1]!;
      const substitutable = d === t || candidateSet(pack, t).includes(d);
      D[i]![j] = Math.min(
        D[i - 1]![j]! + 1, // deletion (of a target phoneme's realization)
        D[i]![j - 1]! + 1, // insertion
        D[i - 1]![j - 1]! + (substitutable ? 0 : 1),
      );
    }
  }
  // backtrack
  const ops: AlignOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0) {
      const d = decoded[i - 1]!;
      const t = target[j - 1]!;
      const substitutable = d === t || candidateSet(pack, t).includes(d);
      const cost = substitutable ? 0 : 1;
      if (D[i]![j]! === D[i - 1]![j - 1]! + cost) {
        ops.push({ type: d === t ? "match" : "substitution", decoded: d, target: t });
        i--;
        j--;
        continue;
      }
    }
    if (j > 0 && D[i]![j] === D[i]![j - 1]! + 1) {
      ops.push({ type: "deletion", target: target[j - 1]! });
      j--;
      continue;
    }
    ops.push({ type: "insertion", decoded: decoded[i - 1]! });
    i--;
  }
  return { ops: ops.reverse() };
}

/** Frame ranges per decoded phoneme from greedy decoding. */
function decodedFrameRanges(frames: LogitFrames): { phoneme: string; from: number; to: number }[] {
  const v = frames.phonemeSet.length;
  const ranges: { phoneme: string; from: number; to: number }[] = [];
  let prev: string | null = null;
  for (let t = 0; t < frames.logits.length / v; t++) {
    const frame = row(frames, t);
    let best = 0;
    for (let i = 1; i < v; i++) if (frame[i]! > frame[best]!) best = i;
    const phoneme = frames.phonemeSet[best]!;
    if (phoneme !== BLANK) {
      if (phoneme !== prev) ranges.push({ phoneme, from: t, to: t + 1 });
      else ranges[ranges.length - 1]!.to = t + 1;
    }
    prev = phoneme;
  }
  return ranges;
}

function indexMap(phonemeSet: string[]): Map<string, number> {
  return new Map(phonemeSet.map((p, i) => [p, i]));
}

/**
 * Score a read-aloud utterance against the expected target phoneme sequence.
 */
export function scoreUtterance(
  frames: LogitFrames,
  target: string[],
  pack: PhonemePack,
  options: ScoringOptions = {},
): ScoreResult {
  const threshold = options.mispronunciationThreshold ?? 0.6;
  const vocab = indexMap(frames.phonemeSet);
  const decoded = greedyCtcDecode(frames);
  const ranges = decodedFrameRanges(frames);
  const { ops } = align(decoded, target, pack);

  const phonemes: PhonemeScore[] = [];
  let insertions = 0;
  // Walk ops in sync with `ranges` (each aligned op consumes one decoded range).
  let r = 0;
  for (const op of ops) {
    if (op.type === "insertion") {
      insertions++;
      if (r < ranges.length) r++;
      continue;
    }
    const t = op.target!;
    const range = ranges[r];
    r++;
    const cands = candidateSet(pack, t);
    const indices = cands.map((c) => vocab.get(c)).filter((x): x is number => x !== undefined);
    const targetIdx = vocab.get(t);
    if (targetIdx === undefined || indices.length === 0) {
      // Phoneme outside the model vocabulary — cannot score it.
      phonemes.push({ phoneme: t, score: 0, mispronounced: true, durationFrames: 0, outcome: op.type });
      continue;
    }
    const from = range ? range.from : 0;
    const to = Math.max(from + 1, range ? range.to : 1);
    // Duration-normalised logit-based GOP: the per-frame posterior of the
    // target phoneme *within the restricted candidate set* (computed from raw
    // logits via a restricted log-softmax), averaged over the phoneme's frames.
    // A perfectly realised phoneme dominates its cluster → score → 1.
    let sum = 0;
    for (let f = from; f < to; f++) {
      const frame = row(frames, f);
      const lp = logProb(frame, targetIdx, indices);
      sum += Math.exp(lp); // restricted posterior of the target
    }
    const score = sum / (to - from); // ∈ (0, 1]
    const isMatch = op.type === "match";
    phonemes.push({
      phoneme: t,
      score,
      mispronounced: score < threshold,
      durationFrames: to - from,
      said: isMatch ? undefined : op.decoded,
      tip: isMatch ? undefined : tipFor(pack, t, op.decoded ?? ""),
      outcome: op.type === "deletion" ? "deletion" : isMatch ? "match" : "substitution",
    });
  }

  const scored = phonemes.filter((p) => p.outcome !== "deletion");
  const mean = scored.length ? scored.reduce((s, p) => s + p.score, 0) / scored.length : 0;
  const penalty = target.length ? insertions / target.length : 0;
  const segmental = Math.round(Math.max(0, mean - penalty) * 100);
  return {
    overall: segmental,
    dimensions: { segmental },
    phonemes,
    decoded,
    insertions,
  };
}

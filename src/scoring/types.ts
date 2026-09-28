/**
 * Scoring contract types — issues #6 and #16 (prosody hooks).
 * The score object shape follows the TDS "scoring contract":
 * overall 0–100 + dimension sub-scores + per-phoneme detail.
 */

/** One frame of raw model logits over the phoneme vocabulary (pre-softmax). */
export interface LogitFrames {
  /** frames × vocab logits, row-major. */
  logits: Float32Array;
  /** Vocabulary order — must match the model's tokenizer (espeak IPA set). */
  phonemeSet: string[];
}

export interface PhonemeScore {
  /** Target phoneme (IPA). */
  phoneme: string;
  /** 0–1 per-phoneme GOP score. */
  score: number;
  /** True when score < threshold (mispronunciation flag). */
  mispronounced: boolean;
  /** Frames assigned to this phoneme (for duration display). */
  durationFrames: number;
  /** What the learner actually said (diagnosis), when it differs. */
  said?: string;
  /** Pack tip for the diagnosed confusion, when available. */
  tip?: string;
  /** Alignment outcome for this target position. */
  outcome: "match" | "substitution" | "deletion";
}

export interface DimensionScores {
  /** Segmental accuracy 0–100 (mean GOP over target phonemes). */
  segmental: number;
  /** Prosody dimensions land in M3 (issue #16) — hooks reserved. */
  intonation?: number;
  stress?: number;
  fluency?: number;
  /** Danish stød flag arrives with issue #12. */
  stoed?: { present: boolean; score: number };
}

export interface ScoreResult {
  /** Overall 0–100. */
  overall: number;
  dimensions: DimensionScores;
  phonemes: PhonemeScore[];
  /** Decoded phoneme sequence the learner actually produced. */
  decoded: string[];
  /** Number of inserted (extra) phonemes not aligned to any target. */
  insertions: number;
}

export interface ScoringOptions {
  /** GOP below this flags a mispronunciation (default 0.6). */
  mispronunciationThreshold?: number;
}

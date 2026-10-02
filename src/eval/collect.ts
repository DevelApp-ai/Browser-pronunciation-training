/**
 * Evaluation data collection — issue #19 (human-rater correlation &
 * per-language threshold tuning).
 *
 * Every scored attempt is appended to an in-session EvalRecord list. The
 * CSV export includes a blank `rater_` column set: a human rater fills
 * them in (0–100 per row), then the correlation between GOP scores and
 * human ratings can be computed offline (Pearson/Spearman per dimension).
 * Threshold tuning: collect the same CSV across languages once per-language
 * models exist (#11–#14) and pick per-language mispronunciation thresholds.
 */
import type { ScoreResult } from "../scoring/types.ts";

export interface EvalRecord {
  timestamp: string;
  language: string;
  exercise: string;
  overall: number;
  segmental: number;
  intonation: number | undefined;
  stress: number | undefined;
  fluency: number | undefined;
  /** Mean per-phoneme GOP and count of mispronounced phonemes. */
  phonemeMean: number;
  mispronounced: number;
  /** Model inference latency (ms) — context for the rater correlation. */
  latencyMs: number;
  /** Raw per-phoneme detail: phoneme:score pairs. */
  phonemes: string;
}

export function toEvalRecord(
  result: ScoreResult & { prosody?: { intonation: number; stress: number; fluency: number } },
  language: string,
  exercise: string,
  latencyMs: number,
): EvalRecord {
  const scored = result.phonemes.filter((p) => p.outcome !== "deletion");
  const phonemeMean = scored.length
    ? scored.reduce((s, p) => s + p.score, 0) / scored.length
    : 0;
  return {
    timestamp: new Date().toISOString(),
    language,
    exercise,
    overall: result.overall,
    segmental: result.dimensions.segmental,
    intonation: result.dimensions.intonation,
    stress: result.dimensions.stress,
    fluency: result.dimensions.fluency,
    phonemeMean: Number(phonemeMean.toFixed(4)),
    mispronounced: result.phonemes.filter((p) => p.mispronounced).length,
    latencyMs: Math.round(latencyMs),
    phonemes: result.phonemes.map((p) => `${p.phoneme}:${p.score.toFixed(2)}`).join(" "),
  };
}

const CSV_COLUMNS = [
  "timestamp", "language", "exercise", "overall", "segmental",
  "intonation", "stress", "fluency", "phonemeMean", "mispronounced",
  "latencyMs", "phonemes", "rater_segmental", "rater_prosody", "rater_overall",
];

function csvCell(v: string | number | undefined): string {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

/** CSV for rater annotation: empty `rater_*` columns are filled by humans. */
export function toEvalCsv(records: EvalRecord[]): string {
  const rows = [CSV_COLUMNS.join(",")];
  for (const r of records) {
    rows.push(CSV_COLUMNS.map((c) => csvCell((r as unknown as Record<string, string | number | undefined>)[c])).join(","));
  }
  return rows.join("\n");
}

export function downloadEvalCsv(records: EvalRecord[], filename = "bpt-eval.csv"): void {
  const blob = new Blob([toEvalCsv(records)], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

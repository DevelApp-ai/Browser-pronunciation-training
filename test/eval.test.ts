import { describe, expect, it } from "vitest";
import { toEvalCsv, toEvalRecord, type EvalRecord } from "../src/eval/collect.ts";
import type { ScoreResult } from "../src/scoring/types.ts";

const score: ScoreResult = {
  overall: 72,
  dimensions: { segmental: 72, intonation: 55, stress: 80, fluency: 60 },
  phonemes: [
    { phoneme: "θ", score: 0.9, mispronounced: false, durationFrames: 5, outcome: "match" },
    { phoneme: "r", score: 0.4, mispronounced: true, durationFrames: 4, said: "l", outcome: "substitution" },
    { phoneme: "iː", score: 0.0, mispronounced: true, durationFrames: 0, outcome: "deletion" },
  ],
  decoded: ["θ", "l"],
  insertions: 0,
};

describe("toEvalRecord", () => {
  it("aggregates phoneme stats and drops deletions from the mean", () => {
    const r = toEvalRecord(score, "en", "three", 950);
    expect(r.language).toBe("en");
    expect(r.exercise).toBe("three");
    expect(r.segmental).toBe(72);
    expect(r.mispronounced).toBe(2);
    // mean over non-deleted phonemes: (0.9 + 0.4) / 2
    expect(r.phonemeMean).toBeCloseTo(0.65, 3);
    expect(r.latencyMs).toBe(950);
    expect(r.phonemes).toContain("θ:0.90");
  });
});

describe("toEvalCsv", () => {
  it("renders header with rater columns and one row per record", () => {
    const records: EvalRecord[] = [
      toEvalRecord(score, "en", "three", 950),
      toEvalRecord({ ...score, overall: 90, dimensions: { ...score.dimensions, segmental: 90 } }, "en", "three", 800),
    ];
    const csv = toEvalCsv(records);
    const lines = csv.split("\n");
    expect(lines).toHaveLength(3);
    expect(lines[0]).toContain("rater_segmental,rater_prosody,rater_overall");
    expect(lines[1]).toContain("en,three,72,72");
    expect(lines[2]).toContain("en,three,90,90");
  });

  it("escapes commas and quotes in the exercise column", () => {
    const r = toEvalRecord(score, "en", 'they, said "hi"', 100);
    const csv = toEvalCsv([r]);
    expect(csv).toContain('"they, said ""hi"""');
  });
});

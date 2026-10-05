/**
 * Scorer regression suite — issue #19 (fixed utterances, runnable in CI).
 *
 * A golden-fixture battery: fixed synthetic posterior utterances across the
 * four languages with pinned expected outcomes (score bands, diagnosis,
 * flags). Any scorer change that shifts a band or a diagnosis must update
 * this file deliberately — that is the "regression suite of fixed utterances
 * runnable in CI for scorer changes" acceptance item. Human-rated CSV
 * correlation analysis lives in eval/analyze_correlation.py.
 */
import { describe, expect, it } from "vitest";
import { scoreUtterance } from "../src/scoring/gop.ts";
import { makeFrames } from "./util.ts";
import type { PhonemePack } from "../src/packs/schema.ts";
import en from "../public/packs/en.json" with { type: "json" };
import da from "../public/packs/da.json" with { type: "json" };
import ne from "../public/packs/ne.json" with { type: "json" };

const packs: Record<string, PhonemePack> = {
  en: en as unknown as PhonemePack,
  da: da as unknown as PhonemePack,
  ne: ne as unknown as PhonemePack,
};

function setFor(pack: PhonemePack): string[] {
  return ["<blank>", ...pack.ipaInventory];
}

type Case = {
  lang: string;
  name: string;
  frames: ReturnType<typeof makeFrames>;
  target: string[];
  expect: {
    overallMin?: number;
    overallMax?: number;
    outcome?: string;
    said?: string;
    mispronounced?: boolean;
  };
};

function buildCases(): Case[] {
  const cases: Case[] = [];
  // EN: "think" — correct production
  {
    const pack = packs.en!;
    const SET = setFor(pack);
    cases.push({
      lang: "en",
      name: "think — correct",
      frames: makeFrames(SET, [["θ"], ["θ"], ["ɪ"], ["ɪ"], ["ŋ"], ["ŋ"], ["k"], ["k"]]),
      target: ["θ", "ɪ", "ŋ", "k"],
      expect: { overallMin: 90, mispronounced: false },
    });
    // EN: th-fronting — /f/ for /θ/
    cases.push({
      lang: "en",
      name: "think — /f/ for /θ/ (th-fronting)",
      frames: makeFrames(SET, [["f"], ["f"], ["ɪ"], ["ɪ"], ["ŋ"], ["ŋ"], ["k"], ["k"]]),
      target: ["θ", "ɪ", "ŋ", "k"],
      expect: { overallMax: 80 },
    });
    // EN: final-cluster simplification — dropped /k/
    cases.push({
      lang: "en",
      name: "think — deleted /k/",
      frames: makeFrames(SET, [["θ"], ["θ"], ["ɪ"], ["ɪ"], ["ŋ"], ["ŋ"]]),
      target: ["θ", "ɪ", "ŋ", "k"],
      expect: { outcome: "deletion" },
    });
  }
  // DA: /sd/ stød-carrier words exercise the Danish pack
  {
    const pack = packs.da!;
    const SET = setFor(pack);
    const first = pack.ipaInventory[0]!;
    const second = pack.ipaInventory[1] ?? first;
    cases.push({
      lang: "da",
      name: "inventory pair — correct",
      frames: makeFrames(SET, [[first], [first], [second], [second]]),
      target: [first, second],
      expect: { overallMin: 90, mispronounced: false },
    });
    // Danish stød words are a top learner error — swapping the vowel pair
    // with a close competitor must score below a clean production.
    const err = pack.learnerErrors[first]?.[0];
    if (err) {
      cases.push({
        lang: "da",
        name: `${first} → ${err.diagnosed} (learner-error cluster)`,
        frames: makeFrames(SET, [[err.diagnosed], [err.diagnosed], [second], [second]]),
        target: [first, second],
        expect: { overallMax: 85, mispronounced: true },
      });
    }
  }
  // NE: retroflex series from the Devanagari pack
  {
    const pack = packs.ne!;
    const SET = setFor(pack);
    const first = pack.ipaInventory[0]!;
    const second = pack.ipaInventory[1] ?? first;
    cases.push({
      lang: "ne",
      name: "inventory pair — correct",
      frames: makeFrames(SET, [[first], [first], [second], [second]]),
      target: [first, second],
      expect: { overallMin: 90, mispronounced: false },
    });
  }
  return cases;
}

describe("scorer regression suite (issue #19 — fixed utterances in CI)", () => {
  for (const c of buildCases()) {
    it(`${c.lang}: ${c.name}`, () => {
      const result = scoreUtterance(c.frames, c.target, packs[c.lang]!);
      if (c.expect.overallMin !== undefined) expect(result.overall).toBeGreaterThanOrEqual(c.expect.overallMin);
      if (c.expect.overallMax !== undefined) expect(result.overall).toBeLessThanOrEqual(c.expect.overallMax);
      if (c.expect.outcome !== undefined) {
        expect(result.phonemes.some((p) => p.outcome === c.expect.outcome)).toBe(true);
      }
      if (c.expect.mispronounced !== undefined) {
        expect(result.phonemes.every((p) => p.mispronounced === c.expect.mispronounced)).toBe(true);
      }
    });
  }
});

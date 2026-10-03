import { describe, expect, it } from "vitest";
import { greedyCtcDecode, scoreUtterance } from "../src/scoring/gop.ts";
import { makeFrames } from "./util.ts";
import en from "../public/packs/en.json" with { type: "json" };
import type { PhonemePack } from "../src/packs/schema.ts";

const pack = en as unknown as PhonemePack;
const SET = ["<blank>", ...pack.ipaInventory];

describe("greedyCtcDecode", () => {
  it("collapses repeats and drops blanks", () => {
    // one spec per frame: θ θ ɪ ɪ ŋ ŋ k k → decoded θ ɪ ŋ k
    const f = makeFrames(SET, ["θ", "θ", "ɪ", "ɪ", "ŋ", "ŋ", "k", "k"]);
    expect(greedyCtcDecode(f)).toEqual(["θ", "ɪ", "ŋ", "k"]);
  });
});

describe("scoreUtterance (synthetic posterior suite — issue #6 acceptance)", () => {
  it("scores a correct utterance near 100 with no flags", () => {
    const f = makeFrames(SET, [
      ["θ", 10], ["θ", 10], ["ɪ", 10], ["ɪ", 10], ["ŋ", 10], ["ŋ", 10], ["k", 10], ["k", 10],
    ]);
    const result = scoreUtterance(f, ["θ", "ɪ", "ŋ", "k"], pack);
    expect(result.decoded).toEqual(["θ", "ɪ", "ŋ", "k"]);
    expect(result.overall).toBeGreaterThanOrEqual(95);
    expect(result.phonemes.every((p) => !p.mispronounced)).toBe(true);
  });

  it("detects a substitution inside the learner-error cluster with a tip", () => {
    // learner says /s/ instead of /θ/
    const f = makeFrames(SET, [
      ["s", 10], ["s", 10], ["ɪ", 10], ["ɪ", 10], ["ŋ", 10], ["ŋ", 10], ["k", 10], ["k", 10],
    ]);
    const result = scoreUtterance(f, ["θ", "ɪ", "ŋ", "k"], pack);
    const theta = result.phonemes[0]!;
    expect(theta.outcome).toBe("substitution");
    expect(theta.said).toBe("s");
    expect(theta.tip).toContain("/s/");
    expect(theta.mispronounced).toBe(true);
    expect(result.overall).toBeLessThanOrEqual(75);
  });

  it("detects a deletion (target phoneme never produced)", () => {
    const f = makeFrames(SET, [
      ["θ", 10], ["ŋ", 10], ["ŋ", 10], ["k", 10], ["k", 10],
    ]);
    const result = scoreUtterance(f, ["θ", "ɪ", "ŋ", "k"], pack);
    expect(result.phonemes.map((p) => p.outcome)).toContain("deletion");
  });

  it("detects an insertion and penalises the overall score", () => {
    // extra /e/ inserted between ɪ and ŋ
    const f = makeFrames(SET, [
      ["θ", 10], ["θ", 10], ["ɪ", 10], ["ɪ", 10], ["e", 10], ["e", 10], ["ŋ", 10], ["ŋ", 10], ["k", 10], ["k", 10],
    ]);
    const result = scoreUtterance(f, ["θ", "ɪ", "ŋ", "k"], pack);
    expect(result.insertions).toBeGreaterThanOrEqual(1);
    expect(result.overall).toBeLessThan(100);
  });

  it("gives partial credit when a cluster competitor is close", () => {
    // /ɪ/ wins its restricted set but /iː/ is right behind
    const f = makeFrames(SET, [
      ["θ", 10], ["θ", 10],
      [["ɪ", 10], ["iː", 9.4]], [["ɪ", 10], ["iː", 9.4]],
      ["ŋ", 10], ["ŋ", 10], ["k", 10], ["k", 10],
    ]);
    const result = scoreUtterance(f, ["θ", "ɪ", "ŋ", "k"], pack);
    expect(result.overall).toBeGreaterThan(80);
    expect(result.overall).toBeLessThan(100);
  });
});

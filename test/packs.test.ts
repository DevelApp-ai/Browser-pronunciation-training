import { describe, expect, it } from "vitest";
import { candidateSet, graphemesToPhonemes, tipFor, type PhonemePack } from "../src/packs/schema.ts";
import en from "../src/packs/en.json" with { type: "json" };
import da from "../src/packs/da.json" with { type: "json" };
import ne from "../src/packs/ne.json" with { type: "json" };
import neww from "../src/packs/new.json" with { type: "json" };

const pack = en as unknown as PhonemePack;

describe("phoneme pack schema", () => {
  it("all four language packs validate", () => {
    for (const p of [en, da, ne, neww] as unknown as PhonemePack[]) {
      expect(p.language).toBeTruthy();
      expect(Array.isArray(p.ipaInventory)).toBe(true);
      expect(typeof p.g2p).toBe("object");
      expect(typeof p.learnerErrors).toBe("object");
      expect(typeof p.prosodyRules).toBe("object");
    }
  });

  it("candidate set includes the phoneme and its learner errors", () => {
    expect(candidateSet(pack, "θ")).toEqual(["θ", "s", "t", "f"]);
    expect(candidateSet(pack, "p")).toEqual(["p"]);
  });

  it("tips are found for diagnosed confusions", () => {
    expect(tipFor(pack, "θ", "s")).toContain("/s/");
    expect(tipFor(pack, "θ", "z")).toBeUndefined();
  });

  it("G2P resolves known words", () => {
    expect(graphemesToPhonemes(pack, "think")).toEqual(["θ", "ɪ", "ŋ", "k"]);
    expect(graphemesToPhonemes(pack, "Hello, world!")).toEqual([
      "h", "ə", "l", "əʊ", "w", "ɜː", "l", "d",
    ]);
  });

  it("learner error targets are valid IPA inventory entries", () => {
    for (const [target, errors] of Object.entries(pack.learnerErrors)) {
      expect(pack.ipaInventory).toContain(target);
      for (const e of errors as Array<{ diagnosed: string }>) {
        expect(e.diagnosed).toBeTruthy();
      }
    }
  });
});

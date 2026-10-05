import { describe, expect, it } from "vitest";
import { greedyRnntDecode, type RnntInputs } from "../src/scoring/rnnt_decode.ts";

const VOCAB = ["<blank>", "h", "e", "l", "o"];

/** Build joint logits [T × U × V] with a chosen winner per (t, u). */
function joint(
  T: number,
  U: number,
  winner: (t: number, u: number) => number,
  vocabSize = VOCAB.length,
): Float32Array {
  const out = new Float32Array(T * U * vocabSize);
  for (let t = 0; t < T; t++) {
    for (let u = 0; u < U; u++) {
      out[(t * U + u) * vocabSize + winner(t, u)] = 10;
    }
  }
  return out;
}

function inputs(T: number, U: number, winner: (t: number, u: number) => number): RnntInputs {
  return {
    T,
    U,
    joint: joint(T, U, winner),
    vocab: VOCAB,
  };
}

describe("greedyRnntDecode (issue #15 spike — minimal transducer decode)", () => {
  it("emits multiple tokens on one frame when the joint keeps firing", () => {
    // frame 0: emit h then e, then blank; later frames blank
    const r = greedyRnntDecode(inputs(3, 4, (t, u) => (t === 0 && u === 0 ? 1 : t === 0 && u === 1 ? 2 : 0)));
    expect(r.tokens).toEqual(["h", "e"]);
  });

  it("advances the encoder frame on blank and stays coupled to the prefix state", () => {
    // emits "l" at frame 1 for prefix 0, "o" at frame 2 for prefix 1
    const r = greedyRnntDecode(inputs(3, 3, (t, u) => (t === 1 && u === 0 ? 3 : t === 2 && u === 1 ? 4 : 0)));
    expect(r.tokens).toEqual(["l", "o"]);
    expect(r.steps).toBeGreaterThanOrEqual(5); // 3 frames + re-evals after emits
  });

  it("respects the maxTokens safety bound", () => {
    // always emits "l" for every (t, u)
    const r = greedyRnntDecode({ ...inputs(2, 8, () => 3), maxTokens: 3 });
    expect(r.tokens).toEqual(["l", "l", "l"]);
  });

  it("returns empty on a degenerate joint tensor", () => {
    const r = greedyRnntDecode({ T: 0, U: 1, joint: new Float32Array(0), vocab: VOCAB });
    expect(r.tokens).toEqual([]);
    expect(r.steps).toBe(0);
  });
});

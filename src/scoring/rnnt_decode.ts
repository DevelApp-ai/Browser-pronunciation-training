/**
 * Minimal greedy RNN-T transducer decoder — issue #15 spike.
 *
 * This exists to make the decoder-choice comparison concrete and
 * measurable: the CTC path (`greedyCtcDecode`, ~30 stateless lines) versus
 * the stateful transducer loop required for RNN-T. It is NOT wired into
 * scoring — GOP needs per-frame phoneme posteriors, which CTC provides
 * natively and RNN-T does not (see docs/decoder-choice.md).
 *
 * Inputs are the three RNN-T graphs' outputs at one utterance:
 *  - encoder: [T × D] acoustic frames,
 *  - prediction network: embedding of previously emitted tokens
 *    (here already applied — a [U + 1 × D] state matrix),
 *  - joint: [T × U + 1 × V] logits (V includes the blank symbol).
 */

export interface RnntInputs {
  /** Number of encoder frames T. */
  T: number;
  /** Number of prediction states U + 1 (U tokens emitted so far). */
  U: number;
  /** [T × (U + 1) × V] joint logits; V includes blank at index 0. */
  joint: Float32Array;
  /** Vocabulary (joint index → symbol); index 0 is the RNN-T blank. */
  vocab: string[];
  /** Max tokens to emit (safety bound for the stateful loop). */
  maxTokens?: number;
}

export const RNNT_BLANK = 0;

/**
 * Greedy RNN-T decode: for each encoder frame t, we hold a decode state
 * (tokens emitted so far) and advance only while the joint network emits a
 * non-blank token for the CURRENT frame and the CURRENT prefix state.
 * Blank advances t. This loop-carried coupling of (t, prefix) state is the
 * extra complexity the CTC path does not have.
 */
export function greedyRnntDecode(inputs: RnntInputs): { tokens: string[]; steps: number } {
  const { T, U, joint, vocab } = inputs;
  if (T <= 0 || U <= 0 || joint.length < T * U * vocab.length) return { tokens: [], steps: 0 };

  const maxTokens = inputs.maxTokens ?? 2 * T;
  const tokens: string[] = [];
  let steps = 0;
  let u = 0; // number of emitted tokens — indexes the prediction state
  for (let t = 0; t < T && tokens.length < maxTokens; t++) {
    let emitted = true;
    while (emitted && tokens.length < maxTokens) {
      emitted = false;
      steps++;
      // argmax over vocab for (t, u)
      const base = (t * U + u) * vocab.length;
      let best = 0;
      for (let v = 1; v < vocab.length; v++) {
        if (joint[base + v]! > joint[base + best]!) best = v;
      }
      if (best !== RNNT_BLANK) {
        tokens.push(vocab[best]!);
        u++;
        emitted = true; // stay on frame t — state changed, re-evaluate
      }
      // blank: fall through, advance t
    }
  }
  return { tokens, steps };
}

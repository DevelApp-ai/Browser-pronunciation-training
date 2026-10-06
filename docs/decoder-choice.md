# Decoder choice: Conformer-CTC vs Conformer-Transducer in the browser

**Status: spike write-up (issue #15, TDS Open Question 5).** The analysis
framework and recommendation below are based on the current literature and
our architecture; the measurement cells marked ⏳ fill in when the Nepali /
Newari finetunes (#13, #14) produce exportable Conformer checkpoints.

## The question

If a Conformer encoder is preferred for Nepali/Newari, is the extra JS
decoding work for RNN-T worth it versus a Conformer-CTC that exports cleanly?

## Structural comparison

| Aspect | Conformer-CTC | Conformer-RNN-T |
| --- | --- | --- |
| Export to ONNX | Single encoder graph; Optimum/ESPnet-ONNX export paths are well-trodden | Two extra graphs (prediction network + joint network); stateful decode loop must be modelled in the graph or orchestrated from JS |
| Operator coverage (WASM/WebGPU) | Conv + attention ops — same set as our wav2vec2-CTC baseline, already proven in transformers.js | Adds LSTM/GRU cells or extra matmuls per emitted token; loop-carried state is awkward for the ORT Web graph runner |
| JS decode | `greedyCtcDecode` — already implemented in `src/scoring/gop.ts` (~30 lines, stateless) | Stateful greedy decode — now prototyped in `src/scoring/rnnt_decode.ts` (`greedyRnntDecode`, tested): the minimal loop already carries (frame, prefix) state, needs explicit T/U dimensions plus a `maxTokens` safety bound, and re-evaluates the joint after every emission; per-token state stays awkward to keep in sync with the worker protocol |
| Posteriors for GOP | Per-frame phoneme posteriors **fall out of the forward pass** — exactly what alignment-free GOP needs | Transducer posteriors are conditioned on the decode prefix; per-frame phoneme probability is NOT directly available without extra work |
| Streaming | No (offline CTC only) | Yes (native) — but our app is exercise-scoped, not streaming |
| Size | Encoder only | Encoder + prediction + joint (~+10–20%) |

Key point for *this* app: **GOP scoring needs per-frame posteriors.** CTC
gives them for free; RNN-T does not. Adapting RNN-T output to per-frame
phoneme posteriors (e.g. via frame-level pseudo-posteriors) is a research
exercise with its own error profile.

## Export friction (from the ESPnet-ONNX experience)

ESPnet-ONNX demonstrates both CTC and transducer Conformers convert to ONNX,
but the transducer path requires exporting the decoder as a separate
stateful component and reimplementing the decode loop in the host language —
exactly the "extra JS decoding work" the open question asks about. CTC
conversion keeps the whole model behind one `session.run()`.

## Recommendation

**Default stays wav2vec2-CTC** (the M1 architecture):

1. GOP needs per-frame posteriors — CTC's native output; RNN-T's is not.
2. Our CTC decode is already implemented and tested.
3. Nepali has enough data to finetune wav2vec2-CTC (#13); Newari is
   zero-shot + tiny CTC via Nepali transfer (#14). Neither requires a
   transducer.

**Revisit RNN-T only if** (a) streaming scoring becomes a product
requirement, or (b) CTC PER on Nepali/Newari stalls above the gate with a
documented transducer advantage on the same data. The measurement table
below is the decision record for that day:

| Model | PER (Nepali) | PER (Newari) | p50 WASM | p50 WebGPU | Size q8 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| wav2vec2-lv-60-espeak-cv-ft (baseline) | ⏳ | ⏳ | ⏳ | ⏳ | ~360 MB→q8 | reference |
| Conformer-CTC (small) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Conformer-RNN-T (small) | ⏳ | n/a (posterior problem) | ⏳ | ⏳ | ⏳ | ⏳ |

Answer to TDS Open Question 5 (provisional, to be confirmed with ⏳ data):
**not worth it — stay with CTC.**

# Backend benchmarks — WebGPU vs WASM per model

This document answers **TDS Open Question 1**: *which exact backend wins for each finetuned model?*
It pins the default backend per model. The harness lives in `src/benchmark/harness.ts`;
re-run it whenever a new model artifact lands (M2 pipeline) and paste the table here.

## Running the harness

Open the app with `?benchmark=1` (e.g. `http://localhost:5173/?benchmark=1`). The page:

1. generates a fixed set of deterministic synthetic utterances (16 kHz PCM);
2. runs the model on both backends (WebGPU is skipped automatically when unavailable);
3. prints a markdown table — paste it under **Results** below and update **Pinned defaults**.

The same page run also reports the detected device tier. Repeat across device tiers
(discrete GPU, integrated, mobile) and paste one row set per tier.

## Method

- Fixed test utterances (16 kHz PCM, deterministic synthetic set), warm-up excluded, ≥5 runs per utterance.
- p50/p95 latency measured in the phoneme worker across both backends.
- Device tiers: `high` (discrete GPU / ≥8 cores), `integrated`, `mobile`, `no-webgpu`.
- Known risk: wav2vec2 conv stacks can be *slower* on WebGPU ([ONNX Runtime #21618](https://github.com/microsoft/onnxruntime/issues/21618)).

## Results

| Model | Device tier | Backend | p50 (ms) | p95 (ms) | n | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| espeak-phoneme (wav2vec2-lv-60-espeak-cv-ft-ONNX, q8) | — | — | — | — | — | *measurements pending first run on real hardware* |

## Pinned defaults

| Model | Default backend |
| --- | --- |
| espeak-phoneme | `wasm` (placeholder until first measurements) |


## WebNN as a third backend (issue #20, TDS Open Question 6)

Spike status as of 2026-10:

- **Spec status:** the W3C Web Machine Learning WG advanced the WebNN API to
  Candidate Recommendation Draft in August 2026.
- **Browser availability:** WebNN is **flag-gated, not enabled by default** in
  any shipping browser. In Chromium (Chrome/Edge) it is available behind the
  "Enables WebNN API" flag on Windows, Linux, macOS, Android and ChromeOS;
  ONNX Runtime Web's WebNN EP requires Windows 11 24H2+ and a flag on the
  ONNX Runtime side as well. Firefox and Safari have no shipping WebNN path.
- **Operator coverage:** the ORT Web WebNN EP supports a subset of opsets;
  wav2vec2-style conv stacks have not been validated on it by us.

### Decision

**Keep flagged, revisit later.** Rationale:

1. Flag-gated APIs are unusable for a consumer product — the learner cannot
   be asked to flip `chrome://flags`.
2. Our latency-critical path (phoneme GOP) already meets the ~1 s budget
   on WASM (pending #28's Edge/Firefox measurements) and has WebGPU as the
   acceleration tier.
3. WebNN's value proposition (NPU execution) matters most for larger models
   (Whisper-class); our scoring model is small.

### Revisit trigger

Revisit when WebNN ships **enabled-by-default in one major browser on
Windows or Android** AND the #7 harness shows WebGPU slower than WASM on a
device class that matters (the ONNX Runtime #21618 conv risk). At that
point: prototype the WebNN EP behind our own `?backend=webnn` flag, extend
`LatencyStats` with a `"webnn"` backend, and record the table below.

| Model | Device tier | Backend | p50 (ms) | p95 (ms) | n | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| espeak-phoneme | — | webnn | — | — | — | not measurable until browsers ship WebNN unflagged |

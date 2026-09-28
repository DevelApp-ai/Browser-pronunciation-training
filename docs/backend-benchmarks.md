# Backend benchmarks — WebGPU vs WASM per model

This document answers **TDS Open Question 1**: *which exact backend wins for each finetuned model?*
It pins the default backend per model. The harness lives in `src/benchmark/harness.ts`;
re-run it whenever a new model artifact lands (M2 pipeline) and paste the table here.

## Method

- Fixed test utterances (16 kHz PCM), warm-up excluded, ≥5 runs per utterance.
- p50/p95 latency measured in the phoneme worker across both backends.
- Device tiers: `high` (discrete GPU / ≥8 cores), `integrated`, `mobile`, `no-webgpu`.
- Known risk: wav2vec2 conv stacks can be *slower* on WebGPU ([ONNX Runtime #21618](https://github.com/microsoft/onnxruntime/issues/21618)).

## Results

| Model | Device tier | Backend | p50 (ms) | p95 (ms) | Notes |
| --- | --- | --- | --- | --- | --- |
| espeak-phoneme (wav2vec2-lv-60-espeak-cv-ft-ONNX, q8) | — | — | — | — | *measurements pending first run on real hardware* |

## Pinned defaults

| Model | Default backend |
| --- | --- |
| espeak-phoneme | `wasm` (placeholder until first measurements) |

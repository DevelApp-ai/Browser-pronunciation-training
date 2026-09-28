# Browser Pronunciation Training

In-browser pronunciation training: capture audio on-device, extract phoneme
posteriors from a browser-ready CTC model (transformers.js), score with an
alignment-free GOP, and give per-phoneme feedback — no server round-trips.

Reference design: [docs/TDS.md](docs/TDS.md). Milestones are tracked as
GitHub issues (M1 epic: #1).

## Structure

- `src/audio/` — getUserMedia → 16 kHz Float32 PCM capture, VAD trim, chunking (issue #4)
- `src/model/` — Web Worker wrapping the phoneme model; exposes raw logits (issue #5)
- `src/scoring/` — alignment-free, substitution-aware GOP scorer (issue #6)
- `src/packs/` — phoneme-pack JSON schema + English pack (issue #8)
- `src/benchmark/` — WebGPU vs WASM latency harness (issue #7)
- `src/ui/` — phoneme-ribbon feedback UI, i18n strings (issue #9)
- `test/` — synthetic-posterior scorer suite + PCM/pack unit tests

## Development

```bash
npm install
npm test     # vitest
npm run dev  # vite dev server
```

# Browser Pronunciation Training

In-browser pronunciation training: capture audio on-device, extract phoneme
posteriors from a browser-ready CTC model (transformers.js), score with an
alignment-free GOP, and give per-phoneme feedback — no server round-trips.

Reference design: [docs/TDS.md](docs/TDS.md). Milestones are tracked as
GitHub issues (M1 epic: #1, complete; M2: #2; M3: #3).

## Live demo (GitHub Pages)

> Requires the repo setting **Settings → Pages → Build and deployment →
> Source: GitHub Actions**. The deploy workflow
> (`.github/workflows/deploy-pages.yml`) then publishes every push to `main`.

Once enabled: **https://develapp-ai.github.io/Browser-pronunciation-training/**

URL parameters:

- *(none)* — the trainer app (record → per-phoneme feedback)
- `?lang=da` / `?lang=ne` — UI + phoneme-pack language
- `?benchmark=1` — WebGPU vs WASM latency harness; prints a markdown table
  for [docs/backend-benchmarks.md](docs/backend-benchmarks.md)
- `?selftest=1` — browser self-test page (see below)

## Browser self-test & log upload

To test the app in a specific browser (e.g. Edge and Firefox):

1. Open `https://develapp-ai.github.io/Browser-pronunciation-training/?selftest=1`
   in the browser (over HTTPS — getUserMedia and the Hub download need it).
2. Click **Run self-test (environment + model)** — checks getUserMedia,
   module workers, 16 kHz AudioContext, MediaRecorder, WebGPU,
   speechSynthesis, then loads the model and runs one synthetic utterance.
3. Click **Run microphone test** (grants mic permission; records 1.5 s).
4. Click **Download log (JSON)** and/or **Download log (TXT)** and upload the
   file(s) from the chat for analysis. One log per browser please.

## Learner progress (spaced repetition)

The app remembers each learner's phoneme difficulty and drills the sounds
they find hard, Anki-style — entirely on-device (no account, no server):

- Every (language, phoneme) pair is an SM-2 flashcard; the GOP score of each
  attempt is the "answer
 quality". Failed phonemes come back within hours,
  mastered ones grow intervals exponentially.
- Cards persist in IndexedDB (`bpt-progress` DB), with a localStorage
  fallback where IndexedDB is unavailable (e.g. private-mode Safari).
- On return, the app picks the exercise with the most due/weak phonemes; a
  **Next** button re-picks, and a status line shows what's due and the
  weakest sounds.
- `?lang=` isolates progress per language.

## Structure

- `src/audio/` — getUserMedia → 16 kHz Float32 PCM capture, VAD trim, chunking (issue #4)
- `src/model/` — Web Worker wrapping the phoneme model; exposes raw logits (issue #5)
- `src/scoring/` — alignment-free, substitution-aware GOP scorer (issue #6)
- `src/packs/` — phoneme-pack JSON schema + English pack (issue #8)
- `src/benchmark/` — WebGPU vs WASM latency harness (issue #7)
- `src/progress/` — SM-2 spaced repetition + IndexedDB/localStorage persistence
- `src/ui/` — phoneme-ribbon feedback UI, i18n strings (issue #9)
- `src/selftest/` — in-browser self-test with downloadable log (Pages rollout)
- `test/` — synthetic-posterior scorer suite + PCM/pack/harness/selftest/SRS unit tests

## Development

```bash
npm install
npm test     # vitest
npm run dev  # vite dev server
```

<!-- ci-smoke-test -->

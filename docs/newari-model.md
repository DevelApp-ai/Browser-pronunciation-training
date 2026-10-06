# Newari model decision record — issue #14 (TDS Open Questions 2 & 3)

Status: scaffold. OQ2/OQ3 cells fill in when the zero-shot evaluation and
size sweep run on real Nwāchā Munā data. The pipeline wiring (config,
finetune stage, browser revision pinning via `src/model/config.ts`) is in
place so the answer drops in without code changes.

## OQ2: zero-shot vs finetuned at launch

*Does the multilingual espeak phoneme model recognise Newari phones well
enough for usable GOP, or must a finetuned model be the launch artifact?*

| Step | Method | Result |
| --- | --- | --- |
| Zero-shot PER | multilingual espeak model on Nwāchā Munā test split | ⏳ |
| Confusion inspection | per-phoneme error breakdown, aspirated/tap contrasts | ⏳ |
| Decision gate | zero-shot PER usable for scripted read-aloud GOP → ship zero-shot at launch | ⏳ |

Zero-shot evaluation recipe: build the manifest with
`python pipeline/build_manifest.py --lang new --source <nwacha-muna.csv|jsonl> --format ...`
({audio, phonemes}) → greedy decode with the baseline model →
`pipeline/validate_model.py` error-rate functions (PER).

## OQ3: size vs first-load

*What parameter count keeps a finetuned Newari wav2vec2-CTC small enough
(q8/q4) for reasonable first-load while staying accurate?*

| Candidate | Params | q8 MB | q4 MB | PER (Nwāchā Munā) | p50 WASM | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| lv-60 (95M) proximal transfer | 95M | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| base (9.6k h, smaller) | ~94M | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| distilled tiny | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

Sweep procedure: for each candidate × {q8, q4} run
`finetune_model.py new-phoneme` (base = the **Nepali** checkpoint — proximal
transfer), then `quantize_model.py`, and record accuracy vs download MB in
the table above. The config entry pins the launch choice's revision.

## Bundle decision

⏳ pending OQ2 — zero-shot (baseline multilingual) or `new-phoneme` finetune.
Either way the browser consumes it through the language → revision map
(`src/model/config.ts`), no code changes.

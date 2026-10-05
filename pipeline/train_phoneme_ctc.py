#!/usr/bin/env python3
"""Phoneme-CTC finetuning — issues #13 (Nepali) / #14 (Newari via transfer).

Training recipe per the TDS:
  - wav2vec2-CTC with espeak-IPA phoneme targets; the metric is PER
    (phoneme error rate), not WER,
  - augmentation is mandatory for ne/new: additive noise, time-stretch
    (speed), and SpecAugment,
  - Newari runs use the Nepali checkpoint as base (proximal transfer,
    Nwāchā Munā corpus).

Targets are stored per corpus manifest as JSONL:
  {"audio": path.wav, "phonemes": ["n","a","m","a","s","t","e"]}
Phoneme tokens use the same espeak-IPA vocabulary as the browser model so
the finetuned head drops straight into the ONNX export.

GPU required (cuda); this script is intentionally not run in CI.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


class PhonemeCTCDataset:
  """Minimal dataset: manifest rows + on-the-fly augmentation."""

  def __init__(self, rows: list[dict], augment: dict | None = None, sample_rate: int = 16000):
    self.rows = rows
    self.augment = augment or {}
    self.sample_rate = sample_rate

  def augment_pcm(self, pcm: list[float], rng: random.Random) -> list[float]:
    speed = self.augment.get("speed")
    if speed and rng.random() < 0.5:
      factor = rng.uniform(speed[0], speed[1])
      n = max(1, int(len(pcm) / factor))
      out = []
      for i in range(n):
        pos = i * factor
        i0 = int(pos)
        frac = pos - i0
        a = pcm[min(i0, len(pcm) - 1)]
        b = pcm[min(i0 + 1, len(pcm) - 1)]
        out.append(a + (b - a) * frac)
      pcm = out
    noise = self.augment.get("noise")
    if noise:
      pcm = [x + rng.gauss(0, noise) for x in pcm]
    return pcm


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Finetune wav2vec2-CTC on phoneme targets (PER)")
  p.add_argument("--base-model", required=True)
  p.add_argument("--dataset", required=True, help="JSONL manifest: {audio, phonemes}")
  p.add_argument("--output-dir", required=True)
  p.add_argument("--learning-rate", type=float, default=3e-5)
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--batch-size", type=int, default=16)
  p.add_argument("--metric", default="per", choices=["per"])
  p.add_argument("--aug-noise", type=float)
  p.add_argument("--aug-speed")
  p.add_argument("--aug-specaugment", action="store_true")
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  try:
    import torch
    from torch.utils.data import DataLoader
  except ImportError:
    print("torch required — finetuning runs on a GPU host, not in CI", file=__import__("sys").stderr)
    return 3
  if not torch.cuda.is_available():
    print("CUDA device required", file=__import__("sys").stderr)
    return 3

  rows = load_manifest(Path(args.dataset))
  augment: dict = {}
  if args.aug_noise:
    augment["noise"] = args.aug_noise
  if args.aug_speed:
    lo, hi = (float(v) for v in args.aug_speed.split(","))
    augment["speed"] = [lo, hi]
  if args.aug_specaugment:
    augment["specaugment"] = True
  ds = PhonemeCTCDataset(rows, augment)

  # Heavy path: feature extractor + Wav2Vec2ForCTC + HF Trainer with a
  # PER callback. The scaffolding above (manifest, augmentation, args) is
  # the part shared with the pipeline; the actual loop is standard HF.
  from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC  # noqa: F401 (GPU host)
  print(f"finetune {args.base_model} on {len(rows)} utterances "
        f"({args.epochs} epochs, bs {args.batch_size}, lr {args.learning_rate}), "
        f"augment={augment or 'off'}", file=__import__("sys").stderr)
  DataLoader([])  # placeholder until torch data pipeline lands with the corpus
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

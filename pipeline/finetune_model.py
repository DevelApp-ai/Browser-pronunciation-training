#!/usr/bin/env python3
"""Finetune stage — issues #13/#14 (per-language phoneme CTC finetunes).

Precedes export in the pipeline:  finetune → export → quantize → validate →
publish. The command builder is pure (unit-testable, like the other
stages); running it needs torch + transformers + datasets and GPU time.

The training recipe follows the TDS:
  - wav2vec2-CTC phoneme finetune with espeak-IPA targets (PER, not WER),
  - mandatory augmentation for ne/new (noise / speed / SpecAugment),
  - Newari finetunes use proximal transfer from the Nepali checkpoint
    (base_model = ne-phoneme artifact), answering OQ3 with a size sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from export_model import get_artifact, load_config

FINETUNE_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "build"

DEFAULT_AUGMENT = {"noise": 0.005, "speed": [0.9, 1.1], "specaugment": True}


def finetune_cmd(name: str, artifact: dict[str, Any]) -> list[str]:
  """The exact training-script invocation for this artifact (unit-testable)."""
  train = artifact.get("finetune", {})
  if not train:
    raise KeyError(f"artifact '{name}' has no finetune config")
  script = train.get("script", "pipeline/train_phoneme_ctc.py")
  cmd = [
    "python", str(script),
    "--base-model", str(train["base_model"]),
    "--dataset", str(train["dataset"]),
    "--output-dir", str(FINETUNE_DIR / name / "ckpt"),
    "--learning-rate", str(train.get("learning_rate", 3e-5)),
    "--epochs", str(train.get("epochs", 10)),
    "--batch-size", str(train.get("batch_size", 16)),
    "--metric", "per",
  ]
  aug = train.get("augmentation") or {}
  if aug.get("noise"):
    cmd += ["--aug-noise", str(aug["noise"])]
  if aug.get("speed"):
    cmd += ["--aug-speed", f"{aug['speed'][0]},{aug['speed'][1]}"]
  if aug.get("specaugment"):
    cmd += ["--aug-specaugment"]
  return cmd


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Build (and optionally run) the finetune command for an artifact")
  parser.add_argument("name", help="artifact name from pipeline/config.yaml")
  parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
  args = parser.parse_args(argv)

  artifact = get_artifact(load_config(), args.name)
  cmd = finetune_cmd(args.name, artifact)
  if args.dry_run:
    print(" ".join(cmd))
    return 0
  try:
    import subprocess
  except ImportError:  # pragma: no cover
    print("subprocess unavailable", file=sys.stderr)
    return 1
  print(f"Finetuning {args.name}: {' '.join(cmd)}", file=sys.stderr)
  result = subprocess.run(cmd, check=False)
  if result.returncode != 0:
    print("finetune failed — torch/datasets/GPU available?", file=sys.stderr)
    return result.returncode
  print(f"checkpoint written to {FINETUNE_DIR / args.name / 'ckpt'}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

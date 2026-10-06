#!/usr/bin/env python3
"""Corpus manifest builder — issues #13/#14 (and #11/#12).

Converts raw speech corpora into the JSONL manifests the finetune and
validation stages consume:
  {"audio": "path.wav", "phonemes": ["n", "a", "m", "a", ...], "text": "..."}

Sources supported:
  - Mozilla Common Voice (validated.tsv: client_id, path, sentence, ...)
  - Fleurs (transcribed JSONL with {audio, text} rows)
  - Nwāchā Munā / custom CSV (path,text) pairs

Phonemization uses espeak-ng (`espeak-ng --ipa`) so targets use the SAME
IPA vocabulary as the browser model's phoneme set; when espeak-ng is
unavailable, rows pass through with empty phonemes and are flagged for the
per-language G2P fallback (the pack maps) instead — the builder stays
runnable anywhere, including CI.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ESPEAK_VOICES = {"en": "en-us", "da": "da", "ne": "ne", "new": "ne"}


def phonemize_espeak(text: str, voice: str) -> list[str] | None:
  """IPA phoneme list via espeak-ng, or None when unavailable."""
  if shutil.which("espeak-ng") is None:
    return None
  try:
    out = subprocess.run(
      ["espeak-ng", "-q", "--ipa", "-v", voice, "--sep=_", text],
      capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
  except (subprocess.SubprocessError, OSError):
    return None
  # strip stress/length markers and split on the separator
  tokens = []
  for t in out.split("_"):
    t = t.replace("ˈ", "").replace("ˌ", "").replace("ː", "").strip()
    if t:
      tokens.append(t)
  return tokens or None


def read_common_voice(tsv: Path) -> list[dict]:
  rows = []
  with tsv.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
      if r.get("path") and r.get("sentence"):
        rows.append({"audio": r["path"], "text": r["sentence"].strip()})
  return rows


def read_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        r = json.loads(line)
        if r.get("audio") and r.get("text"):
          rows.append({"audio": r["audio"], "text": r["text"].strip()})
  return rows


def read_csv_pairs(path: Path) -> list[dict]:
  rows = []
  with path.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
      if r.get("audio") and r.get("text"):
        rows.append({"audio": r["audio"], "text": r["text"].strip()})
  return rows


def build_manifest(
  rows: list[dict],
  lang: str,
  *,
  seed: int = 13,
  heldout_fraction: float = 0.05,
  max_rows: int | None = None,
) -> tuple[list[dict], list[dict]]:
  """Phonemize + deterministic split → (train, heldout) manifest rows."""
  rng = random.Random(seed)
  rows = list(rows)
  rng.shuffle(rows)
  if max_rows:
    rows = rows[:max_rows]
  train: list[dict] = []
  heldout: list[dict] = []
  n_hold = round(len(rows) * heldout_fraction)
  for i, r in enumerate(rows):
    phonemes = phonemize_espeak(r["text"], ESPEAK_VOICES.get(lang, lang)) or []
    entry = {"audio": r["audio"], "text": r["text"], "phonemes": phonemes}
    (heldout if i < n_hold else train).append(entry)
  return train, heldout


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Build finetune/validation JSONL manifests from raw corpora")
  p.add_argument("--lang", required=True, choices=sorted(ESPEAK_VOICES))
  p.add_argument("--source", required=True, help="validated.tsv / corpus.jsonl / corpus.csv")
  p.add_argument("--format", required=True, choices=["commonvoice", "jsonl", "csv"])
  p.add_argument("--out-dir", default="data")
  p.add_argument("--heldout-fraction", type=float, default=0.05)
  p.add_argument("--max-rows", type=int)
  args = p.parse_args(argv)

  src = Path(args.source)
  readers = {"commonvoice": read_common_voice, "jsonl": read_jsonl, "csv": read_csv_pairs}
  rows = readers[args.format](src)
  if not rows:
    print(f"no usable rows in {src}", file=sys.stderr)
    return 2
  train, heldout = build_manifest(
    rows, args.lang, heldout_fraction=args.heldout_fraction, max_rows=args.max_rows
  )
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  dest_dir = out / args.lang
  dest_dir.mkdir(parents=True, exist_ok=True)
  for split, rows_out in (("train", train), ("heldout", heldout)):
    dest = dest_dir / f"{src.stem}-{split}.jsonl"
    with dest.open("w", encoding="utf-8") as f:
      for r in rows_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows_out)} rows → {dest}", file=sys.stderr)
  if shutil.which("espeak-ng") is None:
    print("note: espeak-ng not found — phoneme targets are empty; "
          "install espeak-ng or use the per-language pack G2P", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

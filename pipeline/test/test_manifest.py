import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from build_manifest import build_manifest, phonemize_espeak, read_common_voice, read_jsonl


def test_phonemize_none_without_espeak(monkeypatch):
  import build_manifest as bm
  monkeypatch.setattr(bm.shutil, "which", lambda _: None)
  assert phonemize_espeak("hello", "en-us") is None


def test_phonemize_parses_separator(monkeypatch):
  import build_manifest as bm
  monkeypatch.setattr(bm.shutil, "which", lambda _: "/usr/bin/espeak-ng")
  monkeypatch.setattr(
    bm.subprocess, "run",
    lambda *a, **k: type("R", (), {"stdout": "h_ə_l_ˈoʊ_"})(),
  )
  assert phonemize_espeak("hello", "en-us") == ["h", "ə", "l", "oʊ"]


def test_build_manifest_deterministic_split_and_rows():
  rows = [{"audio": f"{i}.wav", "text": f"sent {i}"} for i in range(100)]
  train, heldout = build_manifest(rows, "ne", seed=13, heldout_fraction=0.05)
  assert len(heldout) == 5
  assert len(train) == 95
  train2, heldout2 = build_manifest(rows, "ne", seed=13, heldout_fraction=0.05)
  assert train == train2 and heldout == heldout2
  # no espeak in CI → empty phoneme targets, rows preserved
  assert all(r["phonemes"] == [] for r in train + heldout)
  assert all({"audio", "text", "phonemes"} == set(r) for r in train)


def test_max_rows_bounds_the_manifest():
  rows = [{"audio": f"{i}.wav", "text": str(i)} for i in range(50)]
  train, heldout = build_manifest(rows, "da", max_rows=10, heldout_fraction=0.1)
  assert len(train) + len(heldout) == 10


def test_readers(tmp_path):
  tsv = tmp_path / "validated.tsv"
  tsv.write_text("path\tsentence\na.mp3\nb.wav\tHej verden\n", encoding="utf-8")
  assert read_common_voice(tsv) == [{"audio": "b.wav", "text": "Hej verden"}]
  jl = tmp_path / "c.jsonl"
  jl.write_text(json.dumps({"audio": "x.wav", "text": "नमस्ते"}) + "\n\n", encoding="utf-8")
  assert read_jsonl(jl) == [{"audio": "x.wav", "text": "नमस्ते"}]

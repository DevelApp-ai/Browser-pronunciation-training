#!/usr/bin/env python3
"""
Validation gate — issue #10.
CER/PER regression check before publish. Pure-Python edit-distance metric,
so the gate logic itself is unit-testable without torch/onnxruntime; the
transcription comparison against the dataset is what needs the runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_model import artifact_dir, get_artifact, load_config


def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over token lists (phonemes or characters)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rate(hyp: list[str], ref: list[str]) -> float:
    """PER/CER in [0, 1]; 0.0 for empty reference."""
    if not ref:
        return 0.0
    return levenshtein(hyp, ref) / len(ref)


def corpus_error_rate(pairs: list[tuple[list[str], list[str]]]) -> float:
    """Aggregate (micro) error rate over (hypothesis, reference) pairs."""
    total_err = 0
    total_len = 0
    for hyp, ref in pairs:
        total_err += levenshtein(hyp, ref)
        total_len += len(ref)
    return 0.0 if total_len == 0 else total_err / total_len


def gate_passes(artifact: dict, new_rate: float, baseline_rate: float) -> bool:
    """True when the new artifact does not regress beyond the configured gate."""
    max_reg = artifact.get("validation", {}).get("max_regression")
    if max_reg is None:
        return True  # no gate configured
    return new_rate <= baseline_rate + float(max_reg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CER/PER validation gate")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    parser.add_argument("--results", required=True,
                        help="JSON file: [{hyp: [...], ref: [...]}, ...]")
    parser.add_argument("--baseline", type=float, required=True,
                        help="baseline error rate this artifact must not regress against")
    args = parser.parse_args(argv)

    artifact = get_artifact(load_config(), args.name)
    pairs = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rate = corpus_error_rate([(p["hyp"], p["ref"]) for p in pairs])
    ok = gate_passes(artifact, rate, args.baseline)
    print(json.dumps({
        "artifact": args.name,
        "error_rate": round(rate, 4),
        "baseline": args.baseline,
        "gate": artifact.get("validation", {}).get("max_regression"),
        "pass": ok,
    }))
    if not ok:
        print("validation gate FAILED — refusing to publish", file=sys.stderr)
        return 1
    # record the rate for the publish stage to consume
    (artifact_dir(args.name) / "validation.json").write_text(
        json.dumps({"error_rate": rate, "baseline": args.baseline, "pass": True}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

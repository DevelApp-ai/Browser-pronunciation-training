#!/usr/bin/env python3
"""
Rater-correlation analysis — issue #19.

Consumes the eval CSV exported by the app (`?eval=1`, src/eval/collect.ts)
after a human has filled the `rater_segmental` / `rater_prosody` /
`rater_overall` columns, and reports:

  - Pearson + Spearman correlation, GOP vs human ratings (per dimension)
  - per-phoneme F1 for mispronunciation detection (model flag vs rater flag)
  - GOP threshold sweep: which mispronunciation threshold best matches the
    raters — the per-language threshold tuning, derived from data not guesses

Pure Python (no pandas/numpy) so it runs anywhere, including CI.

Usage:
  python eval/analyze_correlation.py bpt-eval-en.csv
  python eval/analyze_correlation.py bpt-eval-new.csv --threshold-sweep 0.3,0.4,0.5,0.6,0.7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _clean(v: str | None) -> float | None:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson(_rank(xs), _rank(ys))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def mispronounced_from_row(row: dict[str, str], threshold: float) -> int:
    """Count phonemes below threshold from the phoneme detail column."""
    detail = row.get("phonemes", "")
    n = 0
    for token in detail.split():
        if ":" not in token:
            continue
        try:
            score = float(token.rsplit(":", 1)[1])
        except ValueError:
            continue
        if score < threshold:
            n += 1
    return n


def analyze(rows: list[dict[str, str]]) -> dict:
    pairs = {
        "segmental": ([], []),
        "prosody": ([], []),
        "overall": ([], []),
    }
    for row in rows:
        gop = _clean(row.get("segmental"))
        rater = _clean(row.get("rater_segmental"))
        if gop is not None and rater is not None:
            pairs["segmental"][0].append(gop)
            pairs["segmental"][1].append(rater)
        gop = _clean(row.get("phonemeMean"))
        rater = _clean(row.get("rater_prosody"))
        if gop is not None and rater is not None:
            pairs["prosody"][0].append(gop * 100)  # normalise 0-1 → 0-100
            pairs["prosody"][1].append(rater)
        gop = _clean(row.get("overall"))
        rater = _clean(row.get("rater_overall"))
        if gop is not None and rater is not None:
            pairs["overall"][0].append(gop)
            pairs["overall"][1].append(rater)

    out: dict = {"n_rated": len(pairs["overall"][0])}
    for dim, (xs, ys) in pairs.items():
        out[dim] = {
            "n": len(xs),
            "pearson": round(pearson(xs, ys), 3) if xs else None,
            "spearman": round(spearman(xs, ys), 3) if xs else None,
        }
    return out


def threshold_sweep(rows: list[dict[str, str]], thresholds: list[float], rater_flag_column: str = "rater_flag") -> list[dict]:
    """
    For each threshold: predict mispronounced-phoneme counts per row and
    compare against rater flag counts. Rows need `rater_flag` = number of
    phonemes the rater judged mispronounced (per-row judgment).
    """
    results = []
    for t in thresholds:
        tp = fp = fn = 0
        for row in rows:
            rater_n = _clean(row.get(rater_flag_column))
            if rater_n is None:
                continue
            pred_n = mispronounced_from_row(row, t)
            # row-level approximation: the count comparison acts as a
            # confusion proxy; exact per-phoneme IDs come from the JSON log
            tp += min(pred_n, int(rater_n))
            fp += max(0, pred_n - int(rater_n))
            fn += max(0, int(rater_n) - pred_n)
        prec, rec, f1 = prf(tp, fp, fn)
        results.append({"threshold": t, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GOP vs human-rater correlation analysis")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--threshold-sweep", default="0.4,0.5,0.6,0.7",
                        help="comma-separated thresholds to tune against rater flags")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    rated = [r for r in rows if _clean(r.get("rater_overall")) is not None]
    if not rated:
        print("no rated rows — fill rater_* columns first", file=sys.stderr)
        return 1

    report = {"file": str(args.csv), "correlations": analyze(rows)}
    thresholds = [float(x) for x in args.threshold_sweep.split(",") if x.strip()]
    sweep = threshold_sweep(rows, thresholds)
    if sweep:
        report["threshold_sweep"] = sweep
        best = max(sweep, key=lambda s: s["f1"])
        report["recommended_threshold"] = best["threshold"]

    print(json.dumps(report, indent=2))
    # "good enough to ship" guidance: |r| >= 0.6 on overall is the bar we publish
    r_overall = report["correlations"]["overall"]["pearson"]
    if r_overall is not None and r_overall >= 0.6:
        print("meets the ship bar (pearson r >= 0.6 on overall)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

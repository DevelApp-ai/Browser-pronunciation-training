import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

import pytest  # noqa: E402

from analyze_correlation import (  # noqa: E402
    analyze,
    mispronounced_from_row,
    pearson,
    prf,
    read_rows,
    spearman,
    threshold_sweep,
)


def test_pearson_perfect_and_anti():
    xs = [1, 2, 3, 4, 5]
    assert pearson(xs, [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert pearson(xs, [10, 8, 6, 4, 2]) == pytest.approx(-1.0)
    assert math.isnan(pearson([1], [1]))


def test_spearman_handles_ties_and_monotonic():
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert spearman([1, 1, 2], [3, 2, 1]) == pytest.approx(-0.866, abs=1e-3)


def test_prf():
    assert prf(3, 1, 1) == (pytest.approx(0.75), pytest.approx(0.75), pytest.approx(0.75))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_mispronounced_from_row():
    row = {"phonemes": "θ:0.90 r:0.40 iː:0.55 l:0.99"}
    assert mispronounced_from_row(row, 0.6) == 2
    assert mispronounced_from_row(row, 0.5) == 1


def test_analyze_correlations(tmp_path):
    csv_text = (
        "timestamp,language,exercise,overall,segmental,intonation,stress,fluency,"
        "phonemeMean,mispronounced,latencyMs,phonemes,rater_segmental,rater_prosody,rater_overall
"
        "t1,en,three,72,72,55,80,60,0.65,2,950,θ:0.90 r:0.40,70,50,71
"
        "t2,en,three,90,90,60,85,70,0.88,0,800,θ:0.91 r:0.85,88,65,92
"
        "t3,en,three,50,50,40,70,55,0.35,3,1100,θ:0.51 r:0.20,45,40,48
"
    )
    p = tmp_path / "eval.csv"
    p.write_text(csv_text, encoding="utf-8")
    rows = read_rows(p)
    out = analyze(rows)
    assert out["n_rated"] == 3
    assert out["overall"]["pearson"] == pytest.approx(1.0, abs=0.01)  # near-perfect mock ratings
    assert out["segmental"]["n"] == 3


def test_threshold_sweep_prefers_matching_threshold():
    rows = [
        {"phonemes": "a:0.30 b:0.90", "rater_flag": "1"},
        {"phonemes": "a:0.95 b:0.98", "rater_flag": "0"},
    ]
    sweep = threshold_sweep(rows, [0.4, 0.5, 0.6, 0.7])
    best = max(sweep, key=lambda s: s["f1"])
    assert best["threshold"] in (0.4, 0.5)  # flags live at ~0.3 → low thresholds win
    assert sweep[0]["f1"] >= 0.99

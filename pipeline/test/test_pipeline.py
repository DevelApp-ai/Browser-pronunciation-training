import json
import subprocess
import sys
from pathlib import Path

# the pipeline modules live one level up from the tests
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # CI installs pyyaml; these tests require it

from export_model import get_artifact, load_config, optimum_export_cmd
from validate_model import corpus_error_rate, error_rate, gate_passes, levenshtein
from publish_model import build_language_map


def test_config_loads_and_has_espeak_artifact():
    cfg = load_config()
    art = get_artifact(cfg, "espeak-phoneme")
    assert art["task"] == "automatic-speech-recognition"
    assert "q8" in art["quantizations"]


def test_export_command_shape():
    art = get_artifact(load_config(), "espeak-phoneme")
    cmd = optimum_export_cmd("espeak-phoneme", art)
    assert cmd[:6] == ["optimum-cli", "export", "--task", "automatic-speech-recognition", "--framework", "pt"]
    assert "fp32" in cmd[-1]


def test_levenshtein_and_rates():
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert error_rate(["θ", "ɪ", "ŋ"], ["θ", "ɪ", "ŋ"]) == 0.0
    # one substitution of three → 1/3
    assert abs(error_rate(["s", "ɪ", "ŋ"], ["θ", "ɪ", "ŋ"]) - 1 / 3) < 1e-9
    assert error_rate(["x"], []) == 0.0


def test_corpus_rate_micro_aggregation():
    pairs = [
        (["a", "b", "c"], ["a", "b", "c"]),   # 0 errors / 3
        (["d"], ["e"]),                       # 1 error / 1
    ]
    assert abs(corpus_error_rate(pairs) - 1 / 4) < 1e-9


def test_gate_logic():
    art = {"validation": {"max_regression": 0.005}}
    assert gate_passes(art, 0.10, 0.099)     # +0.001 ≤ 0.005
    assert not gate_passes(art, 0.11, 0.099)  # +0.011 > 0.005
    assert gate_passes({"validation": {}}, 0.9, 0.1)  # no gate configured


def test_language_map_covers_all_four_languages():
    cfg = load_config()
    lang_map = build_language_map(cfg)
    for lang in ("en", "da", "ne", "new"):
        assert lang in lang_map, f"{lang} missing from the language map"
        assert lang_map[lang]["revision"] == "v1-espeak"


def test_cli_dry_runs():
    pipeline_dir = Path(__file__).resolve().parent.parent
    for script in ("export_model.py", "quantize_model.py"):
        out = subprocess.run(
            [sys.executable, str(pipeline_dir / script),
             "espeak-phoneme", "--dry-run"] + (["--dtype", "q8"] if "quantize" in script else []),
            capture_output=True, text=True, cwd=pipeline_dir,
        )
        assert out.returncode == 0, out.stderr
        assert "optimum" in out.stdout.lower() or "quantize" in out.stdout.lower()

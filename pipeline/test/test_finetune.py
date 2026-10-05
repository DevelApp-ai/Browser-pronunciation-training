import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from export_model import load_config
from finetune_model import finetune_cmd


def test_ne_finetune_command_shape():
    art = load_config()["artifacts"]["ne-phoneme"]
    cmd = finetune_cmd("ne-phoneme", art)
    assert cmd[0:2] == ["python", "pipeline/train_phoneme_ctc.py"]
    flags = {cmd[i]: cmd[i + 1] for i in range(2, len(cmd) - 1, 2)}
    assert flags["--base-model"] == "facebook/wav2vec2-lv-60-espeak-cv-ft"
    assert flags["--metric"] == "per"
    # TDS: augmentation mandatory for Nepali
    assert "--aug-noise" in flags
    assert "--aug-specaugment" in cmd
    assert flags["--aug-speed"] == "0.9,1.1"


def test_new_finetune_uses_nepali_checkpoint():
    art = load_config()["artifacts"]["new-phoneme"]
    cmd = finetune_cmd("new-phoneme", art)
    flags = {cmd[i]: cmd[i + 1] for i in range(2, len(cmd) - 1, 2)}
    # proximal transfer from Nepali (OQ2/OQ3 recipe)
    assert flags["--base-model"] == "pipeline/build/ne-phoneme/ckpt"
    assert flags["--dataset"] == "data/new/nwacha-muna.jsonl"


def test_artifact_without_finetune_raises():
    art = load_config()["artifacts"]["espeak-phoneme"]
    with pytest.raises(KeyError):
        finetune_cmd("espeak-phoneme", art)


def test_train_script_dry_args_parse():
    from train_phoneme_ctc import build_arg_parser
    args = build_arg_parser().parse_args([
        "--base-model", "m", "--dataset", "d.jsonl", "--output-dir", "o",
        "--aug-noise", "0.005", "--aug-speed", "0.9,1.1", "--aug-specaugment",
    ])
    assert args.aug_specaugment is True

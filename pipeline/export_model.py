#!/usr/bin/env python3
"""
Model-prep pipeline — issue #10.
Script/CI wrapper around Optimum: `optimum-cli export --task ... --framework pt`
for wav2vec2-CTC, Whisper, and tiny classifier models.

Stages (one command each, composable in CI):
  export_model.py    PyTorch/HF Hub → ONNX (fp32)
  quantize_model.py  fp32 → fp16 / q8 / q4 (selectable per artifact)
  validate_model.py  CER/PER regression gate before publish
  publish_model.py   versioned release: self-host under public/models/ or Hub

Heavy deps (torch, optimum, onnxruntime) are imported lazily so the config /
CLI surface stays testable without them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # PyYAML is optional for the pure functions; CI installs it
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "pipeline" / "config.yaml"
BUILD_DIR = REPO_ROOT / "pipeline" / "build"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the pipeline config; supports JSON as a yaml-less fallback."""
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        return json.loads(text)
    raise RuntimeError("PyYAML is required to read the YAML config (pip install pyyaml)")


def get_artifact(config: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = config.get("artifacts") or {}
    if name not in artifacts:
        available = ", ".join(sorted(artifacts)) or "(none)"
        raise KeyError(f"unknown artifact '{name}'. configured: {available}")
    return artifacts[name]


def artifact_dir(name: str) -> Path:
    out = BUILD_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def optimum_export_cmd(name: str, artifact: dict[str, Any]) -> list[str]:
    """The exact optimum-cli invocation for this artifact (unit-testable)."""
    task = artifact["task"]
    model = artifact["source_model"]
    out = artifact_dir(name) / "fp32"
    return [
        "optimum-cli", "export", "--task", task, "--framework", "pt",
        str(model), str(out),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a model to ONNX via Optimum")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args(argv)

    artifact = get_artifact(load_config(), args.name)
    cmd = optimum_export_cmd(args.name, artifact)

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    try:
        import subprocess
    except ImportError:  # pragma: no cover
        print("subprocess unavailable", file=sys.stderr)
        return 1
    print(f"Exporting {args.name}: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("export failed — is optimum-cli installed?", file=sys.stderr)
        return result.returncode
    print(f"ONNX export written to {artifact_dir(args.name) / 'fp32'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

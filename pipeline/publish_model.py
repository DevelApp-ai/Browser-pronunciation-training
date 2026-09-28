#!/usr/bin/env python3
"""
Publish stage — issue #10.
Versioned artifacts: each language model is tagged; the browser config maps
language → model revision. Targets:
  - local: self-host alongside the app under public/models/<name>/<revision>/
    (offline PWA, issue #18, picks these up with zero Hub round-trips)
  - hub:  create a versioned release on the Hugging Face Hub
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from export_model import REPO_ROOT, artifact_dir, get_artifact, load_config

MODELS_DIR = REPO_ROOT / "public" / "models"
# browser-side language → model revision map (imported by the runtime config)
LANG_MAP_PATH = REPO_ROOT / "public" / "models" / "languages.json"


def build_language_map(config: dict) -> dict:
    """language → { artifact, revision } for every configured artifact."""
    lang_map: dict[str, dict] = {}
    for name, artifact in (config.get("artifacts") or {}).items():
        revision = artifact.get("revision")
        if not revision:
            continue
        for lang in artifact.get("languages", []):
            lang_map[lang] = {"artifact": name, "revision": revision}
    return lang_map


def publish_local(name: str, artifact: dict) -> Path:
    revision = artifact["revision"]
    dest_root = MODELS_DIR / name / revision
    dest_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for dtype in artifact.get("quantizations", []):
        src = artifact_dir(name) / dtype / "model.onnx"
        if not src.exists():
            print(f"warning: {src} missing — did export/quantize run?", file=sys.stderr)
            continue
        dest = dest_root / dtype
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / "model.onnx")
        copied.append(dtype)
    (dest_root / "manifest.json").write_text(json.dumps({
        "artifact": name,
        "revision": revision,
        "quantizations": copied,
        "task": artifact.get("task"),
    }, indent=2), encoding="utf-8")
    print(f"published {name}@{revision} ({', '.join(copied) or 'no weights!'}) → {dest_root}", file=sys.stderr)
    return dest_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a validated model artifact")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    args = parser.parse_args(argv)

    config = load_config()
    artifact = get_artifact(config, args.name)

    # hard gate: refuse to publish without a passed validation record
    record = artifact_dir(args.name) / "validation.json"
    if not record.exists():
        print("refusing to publish: no validation record — run validate_model.py first", file=sys.stderr)
        return 2
    if not json.loads(record.read_text(encoding="utf-8"))["pass"]:
        print("refusing to publish: last validation failed", file=sys.stderr)
        return 2

    target = artifact.get("publish", {}).get("target", "local")
    if target == "local":
        publish_local(args.name, artifact)
    elif target == "hub":
        print("hub publishing requires huggingface_hub — run in CI with credentials", file=sys.stderr)
        return 3
    else:
        print(f"unknown publish target '{target}'", file=sys.stderr)
        return 2

    # refresh the language → revision map the browser config consumes
    LANG_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    LANG_MAP_PATH.write_text(json.dumps(build_language_map(config), indent=2), encoding="utf-8")
    print(f"language map updated → {LANG_MAP_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

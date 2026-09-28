#!/usr/bin/env python3
"""
Quantization stage — issue #10.
fp32 → fp16 / q8 (int8, ~75% size cut) / q4, selectable per artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_model import artifact_dir, get_artifact, load_config

# AvoriaZ/MRT-style integer-quant via onnxruntime.quantization in the real run;
# sizes are informational for the browser download UX.
DTYPE_DIRS = {"fp16": "fp16", "q8": "q8", "q4": "q4"}


def quantize_cmd(name: str, dtype: str, artifact: dict[str, Any]) -> list[str]:
    """The quantization invocation for (artifact, dtype). Unit-testable."""
    if dtype not in DTYPE_DIRS:
        raise ValueError(f"unsupported dtype '{dtype}' (expected one of {sorted(DTYPE_DIRS)})")
    if dtype not in artifact.get("quantizations", []):
        raise ValueError(f"dtype '{dtype}' not enabled for artifact '{name}' "
                         f"(enabled: {artifact.get('quantizations')})")
    src = artifact_dir(name) / "fp32" / "model.onnx"
    out = artifact_dir(name) / DTYPE_DIRS[dtype] / "model.onnx"
    # onnxruntime quantization entry point (preprocess + per-channel weights)
    return [
        sys.executable, "-m", "pipeline.quantize_run",
        "--input", str(src), "--output", str(out), "--mode", dtype,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize an exported ONNX artifact")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    parser.add_argument("--dtype", choices=sorted(DTYPE_DIRS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    artifact = get_artifact(load_config(), args.name)
    try:
        cmd = quantize_cmd(args.name, args.dtype, artifact)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    import subprocess
    print(f"Quantizing {args.name} → {args.dtype}: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        size = (artifact_dir(args.name) / DTYPE_DIRS[args.dtype] / "model.onnx").stat().st_size
        print(json.dumps({"artifact": args.name, "dtype": args.dtype,
                          "bytes": size, "mb": round(size / 1e6, 1)}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

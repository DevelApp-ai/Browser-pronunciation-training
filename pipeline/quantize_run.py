#!/usr/bin/env python3
"""
Quantization runner — issue #10.
The real onnxruntime/onnxconverter-common stage invoked by quantize_model.py:

    python -m pipeline.quantize_run --input <fp32.onnx> --output <out.onnx> --mode q8

Heavy deps (onnx, onnxruntime, onnxconverter_common) are imported lazily so
the recipe/CLI surface stays unit-testable without them, matching the rest of
the pipeline. q4 is intentionally unsupported with a clear pointer (ORT dynamic
quantization has no 4-bit path; browser-side q4 artifacts come from
MatMulNBits-style tooling, tracked separately).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODES = ("fp16", "q8", "q4")


def plan(mode: str) -> dict:
    """Recipe descriptor for a quantization mode (pure, unit-testable)."""
    if mode not in MODES:
        raise ValueError(f"unsupported mode '{mode}' (expected one of {MODES})")
    if mode == "fp16":
        return {
            "mode": "fp16",
            "tool": "onnxconverter_common.float16.convert_float_to_float16",
            "note": "fp32 weights → fp16; no calibration data needed",
        }
    if mode == "q8":
        return {
            "mode": "q8",
            "tool": "onnxruntime.quantization.quantize_dynamic",
            "weight_type": "QInt8",
            "per_channel": True,
            "note": "dynamic int8 weights (~75% size cut vs fp32), no calibration set",
        }
    return {
        "mode": "q4",
        "tool": None,
        "error": "4-bit quantization is not provided by onnxruntime dynamic quantization",
    }


def check_paths(input_path: Path, output_path: Path) -> None:
    """Validate in/out paths before heavy imports (pure)."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input model not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def quantize_fp16(input_path: Path, output_path: Path) -> None:
    import onnx  # lazy: heavy dep
    from onnxconverter_common import float16

    model = onnx.load(str(input_path))
    onnx.save(float16.convert_float_to_float16(model), str(output_path))


def quantize_q8(input_path: Path, output_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # lazy: heavy dep

    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize an ONNX graph (real onnxruntime runner)")
    parser.add_argument("--input", required=True, help="source fp32 ONNX file")
    parser.add_argument("--output", required=True, help="destination ONNX file")
    parser.add_argument("--mode", choices=MODES, required=True, help="quantization mode")
    parser.add_argument("--dry-run", action="store_true", help="print the recipe without running it")
    args = parser.parse_args(argv)

    recipe = plan(args.mode)
    if recipe.get("error"):
        print(f"error: {recipe['error']} — use fp16 or q8, or produce the q4 artifact "
              "with MatMulNBits tooling and commit it to the Hub repo instead", file=sys.stderr)
        return 2

    print(json.dumps(recipe), file=sys.stderr)
    if args.dry_run:
        return 0

    try:
        check_paths(Path(args.input), Path(args.output))
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"quantizing {args.input} → {args.output} ({args.mode})", file=sys.stderr)
    try:
        if args.mode == "fp16":
            quantize_fp16(Path(args.input), Path(args.output))
        else:
            quantize_q8(Path(args.input), Path(args.output))
    except ImportError as err:
        print(f"missing quantization dependency: {err} — install onnx, onnxruntime, "
              "onnxconverter_common (see .github/workflows/model-prep.yml)", file=sys.stderr)
        return 3
    print(f"done: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
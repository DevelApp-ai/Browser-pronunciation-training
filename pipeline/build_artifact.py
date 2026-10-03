#!/usr/bin/env python3
"""
One-command model build — issue #10 acceptance:
"a new finetuned model drops into the browser by running one pipeline command".

Chains the stages for one artifact from pipeline/config.yaml:

    python pipeline/build_artifact.py <name> [--dtypes fp16,q8] [--skip-validate]

Stages:
  1. export   (export_model.py)   PyTorch/HF Hub → ONNX fp32
  2. quantize (quantize_model.py) fp32 → each requested dtype
  3. validate(validate_model.py)  CER/PER regression gate (--results required
                                  unless --skip-validate for dry runs)
  4. verify  (verify_runtime.py) WASM inference + WebGPU op coverage
                                  (--verify; writes runtime_report.json for #7)
  5. publish  (publish_model.py)  versioned local release + languages.json

Each stage runs as a subprocess and must succeed for the next to start —
the same discipline a CI job would apply.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent


def stage(name: str, cmd: list[str]) -> int:
    print(f"\n=== {name}: {' '.join(cmd)} ===", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"stage '{name}' failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export → quantize → validate → publish in one command")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    parser.add_argument("--dtypes", default="q8", help="comma-separated quantizations to build (default: q8)")
    parser.add_argument("--baseline", type=float, help="baseline error rate for the validation gate")
    parser.add_argument("--results", help="JSON file with [{hyp: [...], ref: [...]}, ...] for validation")
    parser.add_argument("--skip-validate", action="store_true", help="dry-run mode: skip the validation gate")
    parser.add_argument("--verify", action="store_true",
                        help="after quantize: verify WASM inference + WebGPU op coverage (writes runtime_report.json)")
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="seconds of zero-audio for the verification run (default: 1)")
    parser.add_argument("--verify-dtype", default="q8", help="which quantized graph to verify (default: q8)")
    args = parser.parse_args(argv)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    py = sys.executable

    if (rc := stage("export", [py, str(PIPELINE / "export_model.py"), args.name])) != 0:
        return rc
    for dtype in dtypes:
        if (rc := stage(f"quantize:{dtype}", [py, str(PIPELINE / "quantize_model.py"), args.name, "--dtype", dtype])) != 0:
            return rc

    if not args.skip_validate:
        if not args.results or args.baseline is None:
            print("validation requires --results and --baseline (or pass --skip-validate for a dry run)",
                  file=sys.stderr)
            return 2
        if (rc := stage("validate", [py, str(PIPELINE / "validate_model.py"), args.name,
                                     "--results", args.results, "--baseline", str(args.baseline)])) != 0:
            return rc
    else:
        print("skipping validation gate (--skip-validate) — DO NOT publish unvalidated artifacts to prod",
              file=sys.stderr)

    if args.verify:
        verify_cmd = [py, str(PIPELINE / "verify_runtime.py"), args.name,
                      "--dtype", args.verify_dtype, "--seconds", str(args.seconds)]
        if shutil.which("node"):
            verify_cmd.append("--run-wasm")
        else:
            print("node not found — WebGPU static coverage only (no WASM inference)", file=sys.stderr)
        if (rc := stage("verify", verify_cmd)) != 0:
            return rc

    return stage("publish", [py, str(PIPELINE / "publish_model.py"), args.name])


if __name__ == "__main__":
    raise SystemExit(main())
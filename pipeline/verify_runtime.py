#!/usr/bin/env python3
"""
Runtime verification stage — issue #10.
"Verify every exported graph runs on BOTH WASM and WebGPU (operator coverage
differs) — feed results to issue #7's harness."

Two checks per artifact:
  WASM   — real inference: pipeline/verify_web.mjs runs the graph with
          onnxruntime-web on the wasm EP (Node), feeding zeros shaped from the
          graph's own input signature, and records latency.
  WebGPU — static operator coverage: the graph's op inventory is diffed
          against the curated WebGPU kernel list below. The authoritative
          check is #7's in-browser harness (CI has no GPU); this report tells
          it what to look for.

Writes pipeline/build/<name>/runtime_report.json for #7's harness to consume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from export_model import REPO_ROOT, artifact_dir, get_artifact, load_config

# onnxruntime-web WebGPU kernel coverage (opset 17-ish, curated from the ORT
# WebGPU kernel registry; extend as ORT gains ops). Best-effort by design —
# the in-browser harness (#7) is the source of truth.
WEBGPU_SUPPORTED_OPS = {
    "Add", "And", "ArgMax", "AveragePool", "Cast", "Concat", "Constant",
    "ConstantOfShape", "Conv", "ConvTranspose", "Cos", "Div", "Dropout",
    "Einsum", "Equal", "Erf", "Exp", "Expand", "Floor", "Gather", "GatherElements",
    "Gelu", "Gemm", "GlobalAveragePool", "Greater", "GreaterOrEqual", "Identity",
    "InstanceNormalization", "LayerNormalization", "LeakyRelu", "Less",
    "LessOrEqual", "Log", "LogSoftmax", "MatMul", "Max", "MaxPool", "Mean",
    "Min", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "Range", "Reciprocal",
    "ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "Relu", "Reshape",
    "Resize", "Round", "Shape", "Sigmoid", "Sin", "Slice", "Softmax",
    "Softplus", "Softsign", "Split", "Sqrt", "Squeeze", "Sub", "Sum", "Tan",
    "Tanh", "Tile", "TopK", "Transpose", "Unsqueeze", "Where", "Xor",
}


def collect_ops(onnx_path: Path) -> tuple[list[str], int]:
    """Sorted op inventory + opset of the graph (needs the `onnx` package)."""
    import onnx  # lazy: heavy dep

    model = onnx.load(str(onnx_path))
    ops = sorted({node.op_type for node in model.graph.node})
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    return ops, opsets.get("ai.onnx", 0)


def webgpu_coverage(ops: list[str]) -> dict:
    """Static WebGPU operator-coverage diff (pure, unit-testable)."""
    supported = WEBGPU_SUPPORTED_OPS
    missing = [op for op in ops if op not in supported]
    return {
        "covered": len(ops) - len(missing),
        "total": len(ops),
        "missing_ops": missing,
        "ok": not missing,
        "note": "static best-effort coverage; in-browser harness (#7) is authoritative",
    }


def wasm_run_cmd(name: str, dtype: str, model_path: Path, seconds: float = 1.0) -> list[str]:
    """The Node invocation that really runs the graph on the wasm EP."""
    return [
        "node", str(REPO_ROOT / "pipeline" / "verify_web.mjs"),
        "--model", str(model_path),
        "--ep", "wasm",
        "--seconds", str(seconds),
    ]


def write_report(name: str, artifact: dict, report: dict) -> Path:
    out = artifact_dir(name) / "runtime_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an exported graph runs on WASM and covers WebGPU ops")
    parser.add_argument("name", help="artifact name from pipeline/config.yaml")
    parser.add_argument("--dtype", default="q8",
                        help="which quantized graph to verify (default: q8)")
    parser.add_argument("--model", help="verify an explicit ONNX file instead of the artifact build dir")
    parser.add_argument("--run-wasm", action="store_true",
                        help="run real WASM inference via Node + onnxruntime-web")
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="seconds of zero-audio to feed the graph (default: 1)")
    args = parser.parse_args(argv)

    artifact = get_artifact(load_config(), args.name)
    model_path = (Path(args.model) if args.model
                  else artifact_dir(args.name) / args.dtype / "model.onnx")
    if not model_path.is_file():
        print(f"error: {model_path} not found — run export_model.py + quantize_model.py first, "
              f"or pass --model", file=sys.stderr)
        return 2

    try:
        ops, opset = collect_ops(model_path)
    except ImportError:
        print("error: the `onnx` package is required for op inventory "
              "(pip install onnx)", file=sys.stderr)
        return 3

    report: dict = {
        "artifact": args.name,
        "revision": artifact.get("revision"),
        "dtype": args.dtype,
        "model": str(model_path),
        "opset": opset,
        "ops": ops,
        "wasm": {"attempted": False},
        "webgpu": webgpu_coverage(ops),
    }

    if args.run_wasm:
        cmd = wasm_run_cmd(args.name, args.dtype, model_path, args.seconds)
        print(f"WASM run: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        try:
            report["wasm"] = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            report["wasm"] = {"attempted": True, "ok": False,
                              "error": (result.stderr or result.stdout)[-500:]}
    else:
        report["wasm"] = {"attempted": False,
                          "note": "pass --run-wasm for real inference (needs node + onnxruntime-web)"}

    out = write_report(args.name, artifact, report)
    print(json.dumps({k: report[k] for k in ("artifact", "revision", "dtype", "opset")}))
    print(f"report → {out}", file=sys.stderr)
    if not report["webgpu"]["ok"]:
        print(f"WebGPU coverage gaps: {', '.join(report['webgpu']['missing_ops'])}", file=sys.stderr)
    return 0 if report["webgpu"]["ok"] and report["wasm"].get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
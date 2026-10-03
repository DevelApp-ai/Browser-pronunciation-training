import sys
from pathlib import Path

# the pipeline modules live one level up from the tests
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # CI installs pytest

import quantize_run
import verify_runtime
from publish_model import hub_files_to_upload
from quantize_model import quantize_cmd


def test_quantize_run_recipes():
    fp16 = quantize_run.plan("fp16")
    assert fp16["tool"].startswith("onnxconverter_common")
    q8 = quantize_run.plan("q8")
    assert q8["tool"] == "onnxruntime.quantization.quantize_dynamic"
    assert q8["weight_type"] == "QInt8"
    assert q8["per_channel"] is True
    # q4 is deliberately unsupported with a clear pointer
    q4 = quantize_run.plan("q4")
    assert q4["error"] and q4["tool"] is None
    with pytest.raises(ValueError):
        quantize_run.plan("q3")


def test_quantize_cmd_invokes_the_real_runner():
    cmd = quantize_cmd("espeak-phoneme", "q8",
                       {"quantizations": ["fp16", "q8"]})
    assert cmd[:3] == [sys.executable, "-m", "pipeline.quantize_run"]
    flags = dict(zip(cmd[3::2], cmd[4::2]))
    assert flags["--mode"] == "q8"
    assert flags["--input"].endswith("fp32/model.onnx")
    assert flags["--output"].endswith("q8/model.onnx")


def test_webgpu_coverage_diff():
    wav2vec2_ops = ["Add", "Concat", "Conv", "Erf", "Gather", "LayerNormalization",
                    "MatMul", "ReduceMean", "Reshape", "Softmax", "Squeeze", "Transpose"]
    cov = verify_runtime.webgpu_coverage(wav2vec2_ops)
    assert cov["ok"] and cov["total"] == len(wav2vec2_ops) and cov["missing_ops"] == []

    with_exotic = wav2vec2_ops + ["ATen", "RandomUniformLike"]
    cov2 = verify_runtime.webgpu_coverage(with_exotic)
    assert not cov2["ok"]
    assert cov2["missing_ops"] == ["ATen", "RandomUniformLike"]
    assert cov2["covered"] == len(wav2vec2_ops)


def test_wasm_run_cmd_shape():
    cmd = verify_runtime.wasm_run_cmd("espeak-phoneme", "q8", Path("/tmp/model.onnx"), 1.0)
    assert cmd[0] == "node"
    assert cmd[1].endswith("verify_web.mjs")
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert flags["--model"] == "/tmp/model.onnx"
    assert flags["--ep"] == "wasm"


def test_hub_upload_plan_only_includes_built_graphs(tmp_path, monkeypatch):
    # point the build dir at a tmp dir so the test does not touch pipeline/build
    monkeypatch.setattr("export_model.BUILD_DIR", tmp_path)
    artifact = {"revision": "v1-en", "quantizations": ["q8", "fp16"]}
    q8 = tmp_path / "en-phoneme" / "q8" / "model.onnx"
    q8.parent.mkdir(parents=True)
    q8.write_bytes(b"weights")
    pairs = hub_files_to_upload("en-phoneme", artifact)
    assert [name for _, name in pairs] == ["onnx/model_q8.onnx"]
    assert pairs[0][0] == q8
    # nothing built → nothing to upload
    assert hub_files_to_upload("missing", artifact) == []


def test_collect_ops_on_real_graph():
    onnx = pytest.importorskip("onnx")  # heavy dep: skip when not installed
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"], name="add"),
         helper.make_node("MatMul", ["z", "w"], ["out"], name="mm")],
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2]),
         helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.onnx"
        onnx.save(model, str(p))
        ops, opset = verify_runtime.collect_ops(p)
        assert ops == ["Add", "MatMul"] and opset == 17
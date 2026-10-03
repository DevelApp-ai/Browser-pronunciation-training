// WASM runtime verification — issue #10.
// Runs a real inference of the ONNX graph with onnxruntime-web on the wasm
// execution provider (Node), feeding zeros shaped from the graph's own input
// signature. Prints a single JSON line on stdout; everything else goes to
// stderr. Used by pipeline/verify_runtime.py and reusable by #7's harness.
//
//   node pipeline/verify_web.mjs --model <file.onnx> --ep wasm [--seconds 1]
//
// With --ep webgpu this script reports not-supported outside a browser
// (no GPU in Node/CI); the in-browser harness (#7) performs that run.
import { parseArgs } from "node:util";

function log(msg) { process.stderr.write(`[verify_web] ${msg}\n`); }

async function main() {
  const { values } = parseArgs({
    options: {
      model: { type: "string" },
      ep: { type: "string", default: "wasm" },
      seconds: { type: "string", default: "1" },
    },
  });
  if (!values.model) throw new Error("--model is required");
  const seconds = Math.max(0.05, Number(values.seconds) || 1);

  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1; // deterministic, CI-friendly

  if (values.ep === "webgpu" && !globalThis.navigator?.gpu) {
    console.log(JSON.stringify({
      ep: "webgpu", attempted: false, ok: false,
      error: "no WebGPU adapter in this environment — run the in-browser harness (issue #7)",
    }));
    return;
  }

  const buf = await (await import("node:fs/promises")).readFile(values.model);
  const t0 = Date.now();
  const session = await ort.InferenceSession.create(buf, {
    executionProviders: [values.ep],
  });
  const loadMs = Date.now() - t0;

  // Feed zeros shaped from the graph's declared input signature; dynamic
  // dimensions (0/-1/'X') become 1000 samples so any audio-shaped model runs.
  const feeds = {};
  const inputs = session.inputNames || [];
  const meta = session.inputMetadata || {};
  for (const name of inputs) {
    const dims = meta[name]?.dims ?? [1, -1];
    const shape = dims.map((d) => (typeof d === "number" && d > 0 ? d : 1000));
    feeds[name] = new ort.Tensor("float32", new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }

  const t1 = Date.now();
  const results = await session.run(feeds);
  const inferMs = Date.now() - t1;

  const output = Object.values(results)[0];
  console.log(JSON.stringify({
    ep: values.ep,
    attempted: true,
    ok: true,
    loadMs,
    inferMs,
    inputNames: inputs,
    outputShape: output?.dims ?? null,
    outputSample: Array.from(output?.data ?? []).slice(0, 5),
  }));
  log(`ok: load ${loadMs} ms, inference ${inferMs} ms on ${values.ep}`);
}

main().catch((err) => {
  console.log(JSON.stringify({ ep: "unknown", attempted: true, ok: false, error: String(err) }));
  log(`FAILED: ${err}`);
});
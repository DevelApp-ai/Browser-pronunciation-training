/**
 * Phoneme-model Web Worker — issue #5.
 *
 * Loads `onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX` via transformers.js
 * and returns the RAW per-frame logits (espeak IPA set) for GOP scoring, so the
 * UI thread stays responsive. A dedicated Worker makes `wasm.proxy` redundant.
 */
import { pipeline, env } from "@huggingface/transformers";

// Model files: prefer self-hosted /public/models when present (issue #18),
// fall back to the Hub on first load.
env.allowLocalModels = true;
env.backends.onnx.wasm.proxy = false;

const MODEL_ID = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX";

let phonemePipeline: Awaited<ReturnType<typeof pipeline>> | null = null;
let phonemeSet: string[] | null = null;

async function ensureModel(device: "wasm" | "webgpu") {
  if (!phonemePipeline) {
    phonemePipeline = await pipeline("audio-classification", MODEL_ID, {
      dtype: "q8",
      device,
    });
  }
  return phonemePipeline;
}

export interface PhonemeLogitsRequest {
  type: "phoneme-logits";
  pcm: Float32Array; // 16 kHz mono
  device?: "wasm" | "webgpu";
}

export interface PhonemeLogitsResponse {
  type: "phoneme-logits-result" | "error";
  logits?: Float32Array;
  /** Vocabulary order for the logit rows (espeak IPA set + CTC blank). */
  phonemeSet?: string[];
  latencyMs?: number;
  error?: string;
}

self.onmessage = async (event: MessageEvent<PhonemeLogitsRequest>) => {
  const { type, pcm, device = "wasm" } = event.data;
  if (type !== "phoneme-logits") return;
  const t0 = performance.now();
  try {
    const pipe = await ensureModel(device);
    // Raw logits path: run the underlying model and read the CTC output
    // *before* any argmax/collapse, exposing [time × phonemes] to the scorer.
    const output = await (pipe as unknown as {
      (pcm: Float32Array, opts: Record<string, unknown>): Promise<{
        logits: { data: Float32Array; dims: number[] };
        id2label?: Record<number, string>;
      }>;
    })(pcm, { return_timestamps: false, raw_logits: true });

    const dims = output.logits.dims; // [batch=1, time, vocab]
    const vocab = dims[dims.length - 1]!;
    const logits = new Float32Array(output.logits.data); // copy out of the WASM heap
    if (!phonemeSet) {
      phonemeSet =
        output.id2label && Object.keys(output.id2label).length === vocab
          ? Object.values(output.id2label)
          : Array.from({ length: vocab }, (_, i) => (output.id2label?.[i] ?? i.toString()));
    }
    const response: PhonemeLogitsResponse = {
      type: "phoneme-logits-result",
      logits,
      phonemeSet,
      latencyMs: performance.now() - t0,
    };
    (self as unknown as Worker).postMessage(response, [logits.buffer]);
  } catch (err) {
    (self as unknown as Worker).postMessage({
      type: "error",
      error: err instanceof Error ? err.message : String(err),
    } satisfies PhonemeLogitsResponse);
  }
};

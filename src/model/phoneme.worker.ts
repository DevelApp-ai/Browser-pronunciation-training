/**
 * Phoneme-model Web Worker — issue #5.
 *
 * Loads `onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX` via transformers.js
 * and returns the RAW per-frame logits (espeak IPA set + CTC blank) for GOP
 * scoring, so the UI thread stays responsive. A dedicated Worker makes
 * `wasm.proxy` redundant (tested off — see env.backends.onnx.wasm.proxy).
 *
 * Fixes over the first draft:
 * - responses echo `seq` so the main-thread promise map can resolve the
 *   right request (previously the promise never resolved at all);
 * - pipeline type is `automatic-speech-recognition` with `output_logits: true`
 *   (this is a CTC ASR model — `audio-classification` has no logits path);
 * - the pipeline is cached PER DEVICE, so a WebGPU benchmark run (#7) cannot
 *   silently reuse an instance loaded for WASM;
 * - the greedy transcript is returned for the English sanity check.
 */
import { pipeline, env } from "@huggingface/transformers";

// Model files: prefer self-hosted /public/models when present (issue #18),
// fall back to the Hub on first load.
env.allowLocalModels = true;
// Issue #28: transformers.js defaults localModelPath to "/models/", which is
// wrong under the GitHub Pages base path. Vite injects BASE_URL at build time.
env.localModelPath = import.meta.env.BASE_URL + "models/";
const onnxWasm = (env.backends as { onnx?: { wasm?: { proxy?: boolean } } }).onnx?.wasm;
if (onnxWasm) onnxWasm.proxy = false;

const MODEL_ID = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX";

type Device = "wasm" | "webgpu";

/**
 * Minimal structural type for the ASR pipeline — the library's own
 * `ReturnType<typeof pipeline>` union is too complex for tsc to represent.
 */
type AsrPipeline = (
  pcm: Float32Array,
  opts: Record<string, unknown>,
) => Promise<{ text: string; logits?: { data: Float32Array; dims: number[] } }>;

/** One pipeline instance per device — a WASM warm-up must not pin WebGPU out. */
const pipelines = new Map<Device, Promise<AsrPipeline>>();

function ensureModel(device: Device): Promise<AsrPipeline> {
  let p = pipelines.get(device);
  if (!p) {
    p = pipeline("automatic-speech-recognition", MODEL_ID, { dtype: "q8", device }) as unknown as Promise<AsrPipeline>;
    // Do not cache a rejected promise.
    p = p.catch((err) => {
      pipelines.delete(device);
      throw err;
    });
    pipelines.set(device, p);
  }
  return p;
}

/** Vocabulary order for the logit rows (espeak IPA set + CTC blank). */
function vocabFromPipeline(pipe: unknown, vocabSize: number): string[] | null {
  const anyPipe = pipe as {
    model?: { config?: { id2label?: Record<string, string> } };
    tokenizer?: { get_vocab?: () => Record<string, number> };
  };
  const id2label = anyPipe.model?.config?.id2label;
  if (id2label && Object.keys(id2label).length === vocabSize) {
    return Array.from({ length: vocabSize }, (_, i) => id2label[String(i)] ?? i.toString());
  }
  const vocab = anyPipe.tokenizer?.get_vocab?.();
  if (vocab) {
    const arr = Array.from({ length: vocabSize }, () => "");
    let filled = 0;
    for (const [token, idx] of Object.entries(vocab)) {
      if (idx >= 0 && idx < vocabSize && !arr[idx]) {
        arr[idx] = token;
        filled++;
      }
    }
    if (filled === vocabSize) return arr;
  }
  return null;
}

export interface PhonemeLogitsRequest {
  type: "phoneme-logits";
  pcm: Float32Array; // 16 kHz mono
  device?: Device;
  /** Correlation id set by the main thread (src/model/runtime.ts). */
  seq?: number;
}

export interface PhonemeLogitsResponse {
  type: "phoneme-logits-result" | "error";
  logits?: Float32Array;
  /** Vocabulary order for the logit rows (espeak IPA set + CTC blank). */
  phonemeSet?: string[];
  /** Greedy transcript of the utterance (English sanity check). */
  transcript?: string;
  latencyMs?: number;
  error?: string;
  seq?: number;
}

self.onmessage = async (event: MessageEvent<PhonemeLogitsRequest>) => {
  const { type, pcm, device = "wasm", seq } = event.data;
  if (type !== "phoneme-logits") return;
  const t0 = performance.now();
  try {
    const pipe = await ensureModel(device);
    // Raw logits path: `output_logits: true` skips the argmax/CTC-collapse
    // inside the pipeline and exposes the [time × phonemes] tensor to the scorer.
    const output = await pipe(pcm, { output_logits: true, return_timestamps: false });

    if (!output.logits) {
      throw new Error(
        "Model output had no raw logits — the installed @huggingface/transformers must support output_logits for CTC ASR.",
      );
    }
    const dims = output.logits.dims; // [batch=1, time, vocab]
    const vocab = dims[dims.length - 1]!;
    const logits = new Float32Array(output.logits.data); // copy out of the WASM/WebGPU heap
    if (!phonemeSetCache) {
      phonemeSetCache = vocabFromPipeline(pipe, vocab) ?? Array.from({ length: vocab }, (_, i) => i.toString());
    }
    const response: PhonemeLogitsResponse = {
      type: "phoneme-logits-result",
      logits,
      phonemeSet: phonemeSetCache,
      transcript: output.text,
      latencyMs: performance.now() - t0,
      seq, // echo so the main thread resolves the right request
    };
    (self as unknown as Worker).postMessage(response, [logits.buffer]);
  } catch (err) {
    (self as unknown as Worker).postMessage({
      type: "error",
      error: err instanceof Error ? err.message : String(err),
      seq, // echo here too — the rejection must reach the right caller
    } satisfies PhonemeLogitsResponse);
  }
};

let phonemeSetCache: string[] | null = null;
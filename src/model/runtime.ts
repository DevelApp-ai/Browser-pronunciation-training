/**
 * Main-thread wrapper around the phoneme worker — issue #5.
 * startCapture() (#4) → PCM → worker logits → scoreUtterance() (#6).
 */
import type { PhonemeLogitsRequest, PhonemeLogitsResponse } from "./phoneme.worker.ts";
import type { LogitFrames } from "../scoring/types.ts";

export interface PhonemeModel {
  /**
   * Given 16 kHz mono PCM, return the [time × phonemes] raw logit tensor
   * plus inference latency.
   */
  logitFrames(pcm: Float32Array, device?: "wasm" | "webgpu"): Promise<LogitFrames & { latencyMs: number }>;
  terminate(): void;
}

export function loadPhonemeModel(): PhonemeModel {
  const worker = new Worker(new URL("./phoneme.worker.ts", import.meta.url), { type: "module" });
  let seq = 0;
  const pending = new Map<number, { resolve: (v: LogitFrames & { latencyMs: number }) => void; reject: (e: Error) => void }>();

  worker.onmessage = (event: MessageEvent<PhonemeLogitsResponse & { seq?: number }>) => {
    const data = event.data;
    const entry = pending.get(data.seq ?? 0);
    if (!entry) return;
    pending.delete(data.seq ?? 0);
    if (data.type === "error" || !data.logits || !data.phonemeSet) {
      entry.reject(new Error(data.error ?? "phoneme worker failed"));
    } else {
      entry.resolve({
        logits: data.logits,
        phonemeSet: data.phonemeSet,
        latencyMs: data.latencyMs ?? 0,
      });
    }
  };
  worker.onerror = (e) => {
    const err = new Error(`phoneme worker error: ${e.message}`);
    for (const entry of pending.values()) entry.reject(err);
    pending.clear();
  };

  return {
    logitFrames(pcm: Float32Array, device: "wasm" | "webgpu" = "wasm") {
      seq += 1;
      const id = seq;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        const msg: PhonemeLogitsRequest & { seq: number } = { type: "phoneme-logits", pcm, device, seq: id };
        // Copy the PCM — transferring would detach the caller's buffer.
        worker.postMessage(msg);
      });
    },
    terminate() {
      worker.terminate();
    },
  };
}

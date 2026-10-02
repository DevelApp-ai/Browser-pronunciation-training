/**
 * Main-thread wrapper around the phoneme worker — issue #5.
 * startCapture() (#4) → PCM → worker logits → scoreUtterance() (#6).
 */
import type { PhonemeLogitsRequest, PhonemeLogitsResponse } from "./phoneme.worker.ts";
import type { LogitFrames } from "../scoring/types.ts";

export interface PhonemeModel {
  /**
   * Given 16 kHz mono PCM, return the [time × phonemes] raw logit tensor
   * plus inference latency and the greedy transcript (sanity check).
   */
  logitFrames(pcm: Float32Array, device?: "wasm" | "webgpu"): Promise<
    LogitFrames & { latencyMs: number; transcript?: string }
  >;
  terminate(): void;
}

export function loadPhonemeModel(): PhonemeModel {
  const worker = new Worker(new URL("./phoneme.worker.ts", import.meta.url), { type: "module" });
  let seq = 0;
  const pending = new Map<
    number,
    { resolve: (v: LogitFrames & { latencyMs: number; transcript?: string }) => void; reject: (e: Error) => void }
  >();

  worker.onmessage = (event: MessageEvent<PhonemeLogitsResponse>) => {
    const data = event.data;
    const id = data.seq;
    if (id === undefined) return; // unsolicited message — no matching request
    const entry = pending.get(id);
    if (!entry) return;
    pending.delete(id);
    if (data.type === "error" || !data.logits || !data.phonemeSet) {
      entry.reject(new Error(data.error ?? "phoneme worker failed"));
    } else {
      entry.resolve({
        logits: data.logits,
        phonemeSet: data.phonemeSet,
        latencyMs: data.latencyMs ?? 0,
        transcript: data.transcript,
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
        const msg: PhonemeLogitsRequest = { type: "phoneme-logits", pcm, device, seq: id };
        // Copy the PCM — transferring would detach the caller's buffer.
        worker.postMessage(msg);
      });
    },
    terminate() {
      worker.terminate();
    },
  };
}

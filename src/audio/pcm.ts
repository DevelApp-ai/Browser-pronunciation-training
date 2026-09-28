/**
 * Pure PCM helpers (no browser APIs) — issue #4.
 *
 * The scoring models (wav2vec2-CTC, Whisper) require 16 kHz mono Float32 PCM.
 * These helpers are unit-testable in Node.
 */

/** Target sample rate required by the phoneme models. */
export const TARGET_SAMPLE_RATE = 16_000;

/** Maximum utterance length the scorer accepts (seconds). */
export const MAX_CHUNK_SECONDS = 30;

/**
 * Linear resample to `TARGET_SAMPLE_RATE` (16 kHz).
 * `OfflineAudioContext` handles this in the browser; this is the
 * deterministic fallback/reference implementation and test vehicle.
 */
export function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  if (inputRate <= 0) throw new Error(`invalid input sample rate: ${inputRate}`);
  const ratio = TARGET_SAMPLE_RATE / inputRate;
  const outLen = Math.max(1, Math.floor(input.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i / ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(input.length - 1, i0 + 1);
    const frac = srcPos - i0;
    out[i] = input[i0]! * (1 - frac) + input[i1]! * frac;
  }
  return out;
}

/** Root-mean-square of a frame. */
export function frameRms(buf: Float32Array, start: number, len: number): number {
  let sum = 0;
  const end = Math.min(buf.length, start + len);
  for (let i = start; i < end; i++) sum += buf[i]! * buf[i]!;
  const n = Math.max(1, end - start);
  return Math.sqrt(sum / n);
}

export interface TrimOptions {
  /** RMS threshold below which a frame counts as silence (default 0.01). */
  threshold?: number;
  /** Frame size in samples (default 10 ms at 16 kHz). */
  frameSize?: number;
  /** Padding in ms kept around the trimmed region (default 100 ms). */
  padMs?: number;
}

/**
 * Trim leading/trailing silence (RMS gate VAD), issue #4.
 * Returns the original buffer if no speech is detected.
 */
export function trimSilence(pcm: Float32Array, sampleRate = TARGET_SAMPLE_RATE, opts: TrimOptions = {}): Float32Array {
  const threshold = opts.threshold ?? 0.01;
  const frameSize = opts.frameSize ?? Math.floor(sampleRate * 0.01);
  const pad = Math.floor(((opts.padMs ?? 100) / 1000) * sampleRate);

  let first = -1;
  let last = -1;
  for (let i = 0; i + frameSize <= pcm.length; i += frameSize) {
    if (frameRms(pcm, i, frameSize) > threshold) {
      if (first < 0) first = i;
      last = i + frameSize;
    }
  }
  if (first < 0) return pcm; // no speech detected — keep everything
  const from = Math.max(0, first - pad);
  const to = Math.min(pcm.length, last + pad);
  return pcm.slice(from, to);
}

/**
 * Split long recordings into chunks of at most MAX_CHUNK_SECONDS,
 * aligned on frame boundaries so phonemes are not cut mid-frame.
 */
export function chunkPcm(pcm: Float32Array, sampleRate = TARGET_SAMPLE_RATE, maxSeconds = MAX_CHUNK_SECONDS): Float32Array[] {
  const maxLen = Math.floor(maxSeconds * sampleRate);
  if (pcm.length <= maxLen) return [pcm];
  const chunks: Float32Array[] = [];
  for (let i = 0; i < pcm.length; i += maxLen) {
    chunks.push(pcm.subarray(i, Math.min(pcm.length, i + maxLen)));
  }
  return chunks;
}

/** Mix multi-channel input down to mono by channel averaging. */
export function toMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) return new Float32Array(0);
  if (channels.length === 1) return channels[0]!;
  const len = channels[0]!.length;
  const out = new Float32Array(len);
  for (const ch of channels) {
    for (let i = 0; i < len; i++) out[i]! += ch[i]! / channels.length;
  }
  return out;
}

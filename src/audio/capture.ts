/**
 * Audio capture pipeline — issue #4.
 *
 * getUserMedia → Web Audio graph (AnalyserNode for live waveform + RMS gate VAD)
 * → decode/resample to 16 kHz mono Float32 PCM → silence trim + chunking.
 *
 * Exported surface: startCapture() → { pcm, blob, vizStream }
 */
import { chunkPcm, TARGET_SAMPLE_RATE, toMono, trimSilence } from "./pcm.ts";

export interface CaptureConstraints {
  /** AGC stays off by default — dynamics matter for scoring. */
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  channelCount: number;
}

export const DEFAULT_CONSTRAINTS: CaptureConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: false,
  channelCount: 1,
};

export type CaptureErrorCode = "permission-denied" | "no-device" | "unsupported" | "unknown";

export class CaptureError extends Error {
  constructor(
    public code: CaptureErrorCode,
    message: string,
    public cause?: unknown,
  ) {
    super(message);
    this.name = "CaptureError";
  }
}

export interface DeviceInfo {
  deviceId: string;
  label: string;
}

export interface CaptureResult {
  /** 16 kHz mono Float32 PCM, silence-trimmed; `chunks[0]` when ≤30 s. */
  pcm: Float32Array;
  /** Chunks of ≤30 s for long utterances (`pcm` is `chunks[0]`). */
  chunks: Float32Array[];
  /** Original recording as an encoded Blob for playback of the attempt. */
  blob: Blob;
}

export interface CaptureHandle {
  /** Live analyser for waveform visualisation while recording. */
  readonly analyser: AnalyserNode;
  readonly stream: MediaStream;
  /** Stop recording and resolve the 16 kHz PCM result. */
  stop(): Promise<CaptureResult>;
}

export interface CaptureOptions {
  deviceId?: string;
  constraints?: Partial<CaptureConstraints>;
}

function normalizeError(err: unknown): CaptureError {
  const name = (err as { name?: string })?.name ?? "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return new CaptureError("permission-denied", "Microphone permission was denied.", err);
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return new CaptureError("no-device", "No matching microphone was found.", err);
  if (typeof navigator !== "undefined" && !navigator.mediaDevices?.getUserMedia)
    return new CaptureError("unsupported", "This browser does not support getUserMedia.", err);
  return new CaptureError("unknown", "Audio capture failed.", err);
}

/** List available input devices (labels need prior permission). */
export async function listInputDevices(): Promise<DeviceInfo[]> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === "audioinput")
      .map((d) => ({ deviceId: d.deviceId, label: d.label || "Microphone" }));
  } catch (err) {
    throw normalizeError(err);
  }
}

/** Create an AudioContext pinned to 16 kHz (the model input rate). */
function createCaptureContext(): AudioContext {
  try {
    return new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  } catch {
    // Safari may reject non-default rates — fall back and resample later.
    return new AudioContext();
  }
}

/**
 * Start capturing. `stop()` resolves with the CaptureResult.
 * AGC is disabled by default (dynamics matter for scoring); EC/NS are on.
 */
export function startCapture(options: CaptureOptions = {}): CaptureHandle {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    throw new CaptureError("unsupported", "This browser does not support getUserMedia.");
  }

  const collected: Float32Array[] = [];
  let stream: MediaStream;
  let ctx: AudioContext;
  let analyser: AnalyserNode;
  let source: MediaStreamAudioSourceNode;
  let processor: ScriptProcessorNode;
  let recorder: MediaRecorder | null = null;
  const encChunks: Blob[] = [];
  let ready: Promise<void>;
  let stopped = false;

  ready = (async () => {
    try {
      const c = { ...DEFAULT_CONSTRAINTS, ...options.constraints };
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: options.deviceId ? { exact: options.deviceId } : undefined,
          echoCancellation: c.echoCancellation,
          noiseSuppression: c.noiseSuppression,
          autoGainControl: c.autoGainControl,
          channelCount: c.channelCount,
        },
      });
    } catch (err) {
      throw normalizeError(err);
    }

    ctx = createCaptureContext();
    source = ctx.createMediaStreamSource(stream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    // PCM tap. ScriptProcessorNode is deprecated but universally supported;
    // an AudioWorklet migration is tracked separately.
    const bufferSize = Math.floor(ctx.sampleRate * 0.1);
    processor = ctx.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = (e) => {
      collected.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    const silent = ctx.createGain();
    silent.gain.value = 0; // keep the tap inaudible
    source.connect(processor);
    processor.connect(silent);
    processor.connect(ctx.destination);

    try {
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) encChunks.push(e.data);
      };
      recorder.start(250);
    } catch {
      recorder = null; // encoded playback is best-effort
    }
  })();

  function flushPcm(): Float32Array {
    const total = collected.reduce((n, c) => n + c.length, 0);
    const out = new Float32Array(total);
    let off = 0;
    for (const c of collected) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  return {
    get analyser(): AnalyserNode {
      return analyser;
    },
    get stream(): MediaStream {
      return stream;
    },
    async stop(): Promise<CaptureResult> {
      await ready;
      if (stopped) throw new CaptureError("unknown", "stop() was already called.");
      stopped = true;

      const rawPcm = flushPcm();
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }
      stream.getTracks().forEach((t) => t.stop());
      processor.disconnect();
      source.disconnect();

      // If the context could not be pinned to 16 kHz (Safari), resample.
      const pcm16k =
        ctx.sampleRate === TARGET_SAMPLE_RATE ? rawPcm : await resampleOffline(rawPcm, ctx.sampleRate);
      await ctx.close();

      const mono = toMono([pcm16k]);
      const trimmed = trimSilence(mono);
      const chunks = chunkPcm(trimmed);
      const blob = new Blob(encChunks, { type: recorder?.mimeType ?? "audio/webm" });
      return { pcm: chunks[0]!, chunks, blob };
    },
  };
}

/** Deterministic resampling via OfflineAudioContext. */
async function resampleOffline(pcm: Float32Array, fromRate: number): Promise<Float32Array> {
  const outLen = Math.max(1, Math.ceil((pcm.length * TARGET_SAMPLE_RATE) / fromRate));
  const offline = new OfflineAudioContext(1, outLen, TARGET_SAMPLE_RATE);
  const buf = offline.createBuffer(1, pcm.length, fromRate);
  // TS 5.7: copyToChannel needs a concrete ArrayBuffer-backed array
  buf.copyToChannel(new Float32Array(pcm), 0);
  const src = offline.createBufferSource();
  src.buffer = buf;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/**
 * Model download tracking — issue #18 (offline PWA).
 *
 * The first run downloads the q8 ONNX weights (~50 MB) from the Hugging Face
 * Hub; transformers.js then keeps them in Cache Storage ("transformers-cache"),
 * so every later session scores fully offline. This module turns the
 * library's per-file progress callbacks into one aggregate progress state
 * for the UI and reports whether the model is already cached on this device.
 */

/** Progress event for one file, as emitted by transformers.js. */
export interface FileProgress {
  file: string;
  loaded: number;
  total: number;
}

export interface AggregateProgress {
  files: number;
  loadedBytes: number;
  totalBytes: number;
  percent: number;
  complete: boolean;
}

/** Aggregates per-file download progress across the model's files. */
export class DownloadTracker {
  private readonly perFile = new Map<string, FileProgress>();

  update(event: FileProgress): AggregateProgress {
    // Some events arrive without a known size (total <= 0) — ignore those.
    if (event.total > 0) {
      this.perFile.set(event.file, {
        file: event.file,
        loaded: Math.min(event.loaded, event.total),
        total: event.total,
      });
    }
    return this.aggregate();
  }

  aggregate(): AggregateProgress {
    let loadedBytes = 0;
    let totalBytes = 0;
    for (const p of this.perFile.values()) {
      loadedBytes += p.loaded;
      totalBytes += p.total;
    }
    return {
      files: this.perFile.size,
      loadedBytes,
      totalBytes,
      percent: totalBytes > 0 ? (loadedBytes / totalBytes) * 100 : 0,
      complete: totalBytes > 0 && loadedBytes >= totalBytes,
    };
  }
}

/** Human-readable size, e.g. "52 MB" (rounded). */
export function formatMB(bytes: number): string {
  return Math.max(0, Math.round(bytes / (1024 * 1024))) + " MB";
}

const MODEL_REPO = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX";

/**
 * Whether the model weights are already in the browser's Cache Storage,
 * i.e. scoring will work offline. Returns false outside the browser (Node
 * test environment) or when Cache Storage is unavailable.
 */
export async function modelFilesCached(): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some((req) => req.url.includes(MODEL_REPO));
  } catch {
    return false;
  }
}

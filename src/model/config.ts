/**
 * Language → model revision config — issue #10 (versioned artifacts).
 *
 * `pipeline/publish_model.py` regenerates `public/models/languages.json`
 * after every publish; this module resolves the artifact + revision the
 * browser should load for a language, falling back to the multilingual
 * espeak baseline when the map is absent (pre-M2) or lacks the language.
 *
 * A pinned artifact with a local manifest (published with target `local`)
 * is served from `public/models/<artifact>/<revision>/` — transformers.js
 * picks it up via env.localModelPath with zero Hub round-trips, so the
 * PWA (issue #18) precaches it with the rest of the shell.
 */

export interface LanguageModelEntry {
  artifact: string;
  revision: string;
}

export interface ModelConfig {
  /** Hub repo id or local artifact directory name. */
  modelId: string;
  /** True when the artifact is self-hosted under public/models. */
  local: boolean;
  entry?: LanguageModelEntry;
}

export const BASELINE_MODEL_ID = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX";

interface FetchedConfig {
  languages: Record<string, LanguageModelEntry>;
  localArtifacts: string[];
}

const cache = new Map<string, Promise<FetchedConfig>>();

async function fetchConfig(baseUrl = import.meta.env.BASE_URL + "models"): Promise<FetchedConfig> {
  const languages: Record<string, LanguageModelEntry> = {};
  try {
    const res = await fetch(`${baseUrl}/languages.json`);
    if (res.ok) {
      const map = (await res.json()) as Record<string, LanguageModelEntry>;
      for (const [lang, entry] of Object.entries(map)) {
        if (entry && typeof entry.artifact === "string" && typeof entry.revision === "string") {
          languages[lang] = entry;
        }
      }
    }
  } catch {
    /* offline / not published yet — the baseline fallback applies */
  }
  // A local publish creates public/models/<artifact>/<revision>/manifest.json;
  // only artifacts actually present locally can be loaded without the Hub.
  const localArtifacts: string[] = [];
  await Promise.all(
    Object.values(languages).map(async (entry) => {
      try {
        const res = await fetch(`${baseUrl}/${entry.artifact}/${entry.revision}/manifest.json`, { method: "HEAD" });
        if (res.ok) localArtifacts.push(entry.artifact);
      } catch {
        /* Hub-only artifact */
      }
    }),
  );
  return { languages, localArtifacts };
}

export function modelConfigForLang(lang: string, baseUrl?: string): Promise<ModelConfig> {
  const key = baseUrl ?? import.meta.env.BASE_URL + "models";
  let cached = cache.get(key);
  if (!cached) {
    cached = fetchConfig(key);
    cache.set(key, cached);
  }
  return cached.then(({ languages, localArtifacts }) => {
    const entry = languages[lang];
    if (!entry) {
      return { modelId: BASELINE_MODEL_ID, local: false };
    }
    if (localArtifacts.includes(entry.artifact)) {
      return { modelId: `${entry.artifact}/${entry.revision}`, local: true, entry };
    }
    return { modelId: entry.artifact, local: false, entry };
  });
}

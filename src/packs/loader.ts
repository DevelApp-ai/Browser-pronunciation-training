/**
 * Pack loader — issue #8.
 * Packs are plain JSON fetched at runtime: adding a language requires no rebuild.
 * In dev, Vite serves source files, so the default base covers `src/packs`;
 * for production builds copy the pack JSONs to a served `/packs` directory.
 */
import type { PhonemePack } from "./schema.ts";

const cache = new Map<string, PhonemePack>();

export async function loadPack(language: string, baseUrl = "/src/packs"): Promise<PhonemePack> {
  const cached = cache.get(language);
  if (cached) return cached;
  const res = await fetch(`${baseUrl}/${language}.json`);
  if (!res.ok) throw new Error(`No phoneme pack for language "${language}" (${res.status})`);
  const pack = (await res.json()) as PhonemePack;
  // minimal validation so a linguist's typo fails loudly
  if (!pack.language || !Array.isArray(pack.ipaInventory) || typeof pack.g2p !== "object" || typeof pack.learnerErrors !== "object") {
    throw new Error(`Phoneme pack for "${language}" does not match the schema.`);
  }
  cache.set(language, pack);
  return pack;
}

/** Available language codes known to the app (pack files may or may not exist yet). */
export const SUPPORTED_LANGUAGES = ["en", "da", "ne", "new"] as const;

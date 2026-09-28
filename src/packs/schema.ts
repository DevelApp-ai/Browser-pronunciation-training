/**
 * Phoneme-pack schema — issue #8.
 * Packs are plain JSON, loadable per language without rebuilds, and
 * linguist-editable. Adding a language = adding a JSON file.
 */

export interface LearnerError {
  /** The phoneme the learner produced instead. */
  diagnosed: string;
  /** Contrastive tip shown in the UI. */
  tip: string;
}

export interface ProsodyRules {
  /** Whether the language is stress-timed / syllable-timed (display hint). */
  rhythmType?: "stress-timed" | "syllable-timed" | "mora-timed";
  /** Primary stress marker used by the G2P output. */
  stressMarker?: string;
  /** Free-form per-language prosody notes for M3 (issue #16). */
  notes?: string;
}

export interface PhonemePack {
  /** BCP-47-ish language code (`en`, `en-GB`, `da`, `ne`, `new`). */
  language: string;
  /** IPA phonemes this language scores. */
  ipaInventory: string[];
  /** Grapheme → phoneme map (CMUdict-style for English; language-specific otherwise). */
  g2p: Record<string, string[]>;
  /** Common learner errors per phoneme, used to restrict candidate sets. */
  learnerErrors: Record<string, LearnerError[]>;
  prosodyRules: ProsodyRules;
}

/** Candidate set for a target phoneme: the phoneme itself + its confusables. */
export function candidateSet(pack: PhonemePack, phoneme: string): string[] {
  const errors = (pack.learnerErrors[phoneme] ?? []).map((e) => e.diagnosed);
  return Array.from(new Set([phoneme, ...errors]));
}

/** Tip lookup for a target/substitution pair. */
export function tipFor(pack: PhonemePack, target: string, said: string): string | undefined {
  return pack.learnerErrors[target]?.find((e) => e.diagnosed === said)?.tip;
}

/**
 * Convert text to a target phoneme sequence using the pack's G2P map.
 * Unknown words fall back to per-character lookup, then are skipped.
 */
export function graphemesToPhonemes(pack: PhonemePack, text: string): string[] {
  const out: string[] = [];
  const words = text.toLowerCase().match(/[a-z\u00c0-\u024f\u0900-\u097F]+/g) ?? [];
  for (const word of words) {
    if (pack.g2p[word]) {
      out.push(...pack.g2p[word]);
      continue;
    }
    // longest-match character fallback
    let i = 0;
    while (i < word.length) {
      let matched = false;
      for (let len = Math.min(3, word.length - i); len > 0; len--) {
        const seq = pack.g2p[word.slice(i, i + len)];
        if (seq) {
          out.push(...seq);
          i += len;
          matched = true;
          break;
        }
      }
      if (!matched) i += 1; // skip unmodelled character silently
    }
  }
  return out;
}

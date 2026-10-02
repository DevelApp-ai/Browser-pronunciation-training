/**
 * Progress session — ties SRS + storage into the app loop.
 *
 * On load: fetch the learner's cards, pick the best next exercise
 * (spaced-repetition weighted), show what's due.
 * After each scored attempt: update every target phoneme's card.
 * On unload: mark a session visit.
 */
import type { PhonemeScore } from "../scoring/types.ts";
import type { SrsCard } from "./srs.ts";
import { exercisePriority, newCard, review, weakestPhonemes } from "./srs.ts";
import type { ProgressStore } from "./store.ts";

export interface DueSummary {
  dueCount: number;
  totalCards: number;
  weakest: SrsCard[];
}

/** Pick the exercise with the highest spaced-repetition priority. */
export function pickExercise(
  exercises: string[],
  exercisePhonemes: (exercise: string) => string[],
  cards: Map<string, SrsCard>,
  now = Date.now(),
): string {
  let best = exercises[0] ?? "";
  let bestPriority = -1;
  for (const ex of exercises) {
    const p = exercisePriority(exercisePhonemes(ex), cards, now);
    if (p > bestPriority) {
      bestPriority = p;
      best = ex;
    }
  }
  return best;
}

/** Apply one scored attempt to the learner's cards (mutates store). */
export async function recordAttempt(
  store: ProgressStore,
  lang: string,
  phonemes: PhonemeScore[],
  now = Date.now(),
): Promise<SrsCard[]> {
  const updated: SrsCard[] = [];
  for (const p of phonemes) {
    const existing = await store.getCard(lang, p.phoneme);
    const card = existing ?? newCard(lang, p.phoneme, now);
    const next = review(card, p.score, now);
    await store.putCard(next);
    updated.push(next);
  }
  return updated;
}

/** Due/weak summary for the UI. */
export async function loadSummary(store: ProgressStore, lang: string, now = Date.now()): Promise<DueSummary> {
  const cards = await store.getAllCards(lang);
  return {
    dueCount: cards.filter((c) => c.dueAt <= now).length,
    totalCards: cards.length,
    weakest: weakestPhonemes(cards),
  };
}

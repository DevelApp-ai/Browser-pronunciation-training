/**
 * Spaced repetition for phonemes — SM-2 (the algorithm behind Anki).
 *
 * Each (language, phoneme) is a flashcard whose "answer quality" is the
 * learner's GOP score on that phoneme in the exercises where it appears.
 * Poor scores shorten the interval (review soon), good ones grow it
 * exponentially — so returning learners are drilled on exactly the
 * phonemes they find difficult.
 *
 * Pure functions: storage lives in store.ts.
 */

/** One SRS card per (language, phoneme). */
export interface SrsCard {
  /** Language code, e.g. "en" — cards are per-language. */
  lang: string;
  /** Target phoneme (IPA). */
  phoneme: string;
  /** SM-2 easiness factor (2.5 default, ≥1.3). */
  ease: number;
  /** Current review interval in days (0 = new/unseen). */
  intervalDays: number;
  /** Successful reviews in a row. */
  reps: number;
  /** Times the phoneme lapsed below the pass mark. */
  lapses: number;
  /** Epoch ms when the phoneme is due again. */
  dueAt: number;
  /** Last seen, epoch ms (for display). */
  lastSeenAt: number;
  /** Rolling mean GOP 0–1 (display + weakest-phoneme ranking). */
  meanScore: number;
  /** Times practised. */
  attempts: number;
}

/** A brand-new card for a phoneme the learner has never practised. */
export function newCard(lang: string, phoneme: string, now = Date.now()): SrsCard {
  return {
    lang,
    phoneme,
    ease: 2.5,
    intervalDays: 0,
    reps: 0,
    lapses: 0,
    dueAt: now,
    lastSeenAt: now,
    meanScore: 0,
    attempts: 0,
  };
}

/** GOP score (0–1) → SM-2 answer quality 0–5. */
export function gopToQuality(score: number): number {
  if (score >= 0.9) return 5; // perfect
  if (score >= 0.8) return 4; // correct, slight hesitation
  if (score >= 0.6) return 3; // correct, with effort
  if (score >= 0.4) return 2; // incorrect but recognised
  return 1; // incorrect
}

export const DAY_MS = 86_400_000;

/**
 * SM-2 update after one attempt.
 * q < 3 (below the 0.6 GOP pass mark) resets the interval and lapses.
 */
export function review(card: SrsCard, gopScore: number, now = Date.now()): SrsCard {
  const q = gopToQuality(gopScore);
  const attempts = card.attempts + 1;
  // exponential moving average keeps the display honest without storing history
  const meanScore = attempts === 1 ? gopScore : card.meanScore * 0.7 + gopScore * 0.3;
  const base = { ...card, attempts, meanScore, lastSeenAt: now };

  if (q < 3) {
    // Anki-style "again": a lapsed phoneme is due *now*, so the learner
    // re-drills it within the same session instead of hours later.
    return {
      ...base,
      ease: Math.max(1.3, base.ease - 0.2),
      reps: 0,
      lapses: base.lapses + 1,
      intervalDays: 0,
      dueAt: now,
    };
  }

  const reps = card.reps + 1;
  let intervalDays: number;
  if (reps === 1) intervalDays = 1;
  else if (reps === 2) intervalDays = 6;
  else intervalDays = Math.round(card.intervalDays * card.ease);

  const ease = card.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02));
  return {
    ...base,
    ease: Math.max(1.3, ease),
    reps,
    intervalDays,
    dueAt: now + intervalDays * DAY_MS,
  };
}

/** True when the card needs practising again (or is new). */
export function isDue(card: SrsCard, now = Date.now()): boolean {
  return card.dueAt <= now;
}

/** Days until due (negative = overdue by that many days). */
export function daysUntilDue(card: SrsCard, now = Date.now()): number {
  return (card.dueAt - now) / DAY_MS;
}

/**
 * Exercise priority for spaced review.
 *
 * An exercise is worth practising when it contains phonemes that are
 * due, weak (low mean GOP), or frequently lapsed. Combines:
 *   - due pressure: overdue/new phonemes pull strongly
 *   - weakness: mean GOP below the pass mark
 *   - lapsed history
 * Returns a score >= 0; higher = better candidate for the next exercise.
 */
export function exercisePriority(
  exercisePhonemes: string[],
  cards: Map<string, SrsCard>,
  now = Date.now(),
): number {
  let priority = 0;
  for (const p of exercisePhonemes) {
    const card = cards.get(p);
    if (!card) {
      priority += 2; // unseen phoneme — worth a first exposure
      continue;
    }
    if (isDue(card, now)) {
      // the more overdue, the stronger the pull (capped)
      priority += 1 + Math.min(3, Math.abs(daysUntilDue(card, now)));
    }
    if (card.meanScore < 0.6) priority += 2 * (1 - card.meanScore); // weakness weight
    priority += Math.min(1.5, 0.3 * card.lapses);
  }
  return priority;
}

/** The learner's weakest phonemes, worst mean GOP first (for the UI). */
export function weakestPhonemes(cards: SrsCard[], limit = 5): SrsCard[] {
  return cards
    .filter((c) => c.attempts > 0)
    .sort((a, b) => a.meanScore - b.meanScore)
    .slice(0, limit);
}

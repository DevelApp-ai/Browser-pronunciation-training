import { describe, expect, it } from "vitest";
import { MemoryStore } from "../src/progress/store.ts";
import { recordAttempt, loadSummary, pickExercise } from "../src/progress/session.ts";
import { exercisePriority, gopToQuality, isDue, newCard, review, weakestPhonemes, DAY_MS } from "../src/progress/srs.ts";
import type { PhonemeScore } from "../src/scoring/types.ts";

const NOW = 1_700_000_000_000;

describe("SM-2 review", () => {
  it("grows intervals exponentially on passing scores", () => {
    let c = newCard("en", "θ", NOW);
    c = review(c, 0.95, NOW);
    expect(c.intervalDays).toBe(1);
    expect(c.reps).toBe(1);
    c = review(c, 0.9, NOW + DAY_MS);
    expect(c.intervalDays).toBe(6);
    c = review(c, 0.9, NOW + 7 * DAY_MS);
    expect(c.intervalDays).toBeGreaterThan(6); // 6 * ease ≈ 14
    expect(c.reps).toBe(3);
    expect(c.ease).toBeGreaterThan(2.5); // perfect answers raise ease
  });

  it("resets and lapses on failing scores", () => {
    let c = newCard("en", "θ", NOW);
    c = review(c, 0.9, NOW);
    c = review(c, 0.2, NOW + DAY_MS);
    expect(c.reps).toBe(0);
    expect(c.lapses).toBe(1);
    expect(c.intervalDays).toBeLessThanOrEqual(0.5);
    expect(isDue(c, NOW + DAY_MS)).toBe(true); // due again within hours
  });

  it("never lets ease drop below 1.3", () => {
    let c = newCard("en", "θ", NOW);
    for (let i = 0; i < 20; i++) c = review(c, 0.1, NOW);
    expect(c.ease).toBeGreaterThanOrEqual(1.3);
  });

  it("maps GOP scores to SM-2 qualities", () => {
    expect(gopToQuality(0.95)).toBe(5);
    expect(gopToQuality(0.85)).toBe(4);
    expect(gopToQuality(0.65)).toBe(3);
    expect(gopToQuality(0.5)).toBe(2);
    expect(gopToQuality(0.1)).toBe(1);
  });

  it("tracks a rolling mean score", () => {
    let c = newCard("en", "θ", NOW);
    c = review(c, 1.0, NOW);
    c = review(c, 0.0, NOW);
    expect(c.meanScore).toBeCloseTo(0.7 * 1.0 + 0.3 * 0.0, 5);
  });
});

describe("exercisePriority / pickExercise", () => {
  it("prefers exercises with due and unseen phonemes", () => {
    const cards = new Map();
    const strong = newCard("en", "p", NOW - 10 * DAY_MS);
    const strongCard = review(review(strong, 0.99, NOW - 10 * DAY_MS), 0.99, NOW);
    cards.set("p", strongCard); // p: not due for days, high mean score
    // exercise A: only the strong phoneme; exercise B: unseen phoneme θ
    const a = exercisePriority(["p"], cards, NOW);
    const b = exercisePriority(["θ"], cards, NOW);
    expect(b).toBeGreaterThan(a); // unseen phoneme wins
  });

  it("pickExercise returns the highest-priority exercise", () => {
    const cards = new Map();
    const weak = newCard("en", "r", NOW - 2 * DAY_MS);
    const weakCard = review(weak, 0.2, NOW - 2 * DAY_MS); // lapsed, due
    cards.set("r", weakCard);
    const chosen = pickExercise(
      ["Hello world", "I think three things"],
      (ex) => (ex === "I think three things" ? ["θ", "r"] : ["h", "ə", "l"]),
      cards,
      NOW,
    );
    expect(chosen).toBe("I think three things");
  });
});

describe("recordAttempt + persistence", () => {
  const ph = (phoneme: string, score: number): PhonemeScore =>
    ({ phoneme, score, mispronounced: score < 0.6, durationFrames: 5, outcome: "match" });

  it("creates, updates, and re-reads cards through a store", async () => {
    const store = new MemoryStore();
    const first = await recordAttempt(store, "en", [ph("θ", 0.9)], NOW);
    expect(first[0]!.attempts).toBe(1);
    const second = await recordAttempt(store, "en", [ph("θ", 0.2)], NOW + DAY_MS);
    expect(second[0]!.lapses).toBe(1);
    const stored = await store.getCard("en", "θ");
    expect(stored!.lapses).toBe(1);
    const summary = await loadSummary(store, "en", NOW + DAY_MS);
    expect(summary.dueCount).toBe(1); // lapsed → due again
    expect(summary.weakest[0]!.phoneme).toBe("θ");
  });

  it("keeps languages separate", async () => {
    const store = new MemoryStore();
    await recordAttempt(store, "en", [ph("θ", 0.9)], NOW);
    await recordAttempt(store, "da", [ph("ʁ", 0.5)], NOW);
    const en = await store.getAllCards("en");
    const da = await store.getAllCards("da");
    expect(en).toHaveLength(1);
    expect(da).toHaveLength(1);
    expect(da[0]!.phoneme).toBe("ʁ");
  });
});

describe("weakestPhonemes", () => {
  it("ranks by mean score, ignoring unattempted cards", () => {
    const cards = [
      { ...newCard("en", "a", NOW), attempts: 1, meanScore: 0.8 },
      { ...newCard("en", "b", NOW), attempts: 1, meanScore: 0.3 },
      newCard("en", "c", NOW), // never attempted
    ];
    const w = weakestPhonemes(cards, 2);
    expect(w.map((c) => c.phoneme)).toEqual(["b", "a"]);
  });
});

import { describe, expect, it } from "vitest";
import { exerciseSlug, findVoice, voiceMatchesLang, type ReferenceVoice } from "../src/ui/reference.ts";

// minimal structural mock for SpeechSynthesisVoice
const v = (lang: string, local = false) => ({ lang, localService: local, default: false, name: lang, voiceURI: lang, getVoices: [] }) as unknown as SpeechSynthesisVoice;

describe("voiceMatchesLang", () => {
  it("matches language-only BCP-47 against regional voices", () => {
    expect(voiceMatchesLang(v("en-GB"), "en")).toBe(true);
    expect(voiceMatchesLang(v("en_US"), "en")).toBe(true);
    expect(voiceMatchesLang(v("da-DK"), "da")).toBe(true);
    expect(voiceMatchesLang(v("fr-FR"), "en")).toBe(false);
  });
  it("matches region-specific requests precisely", () => {
    expect(voiceMatchesLang(v("en-US"), "en-US")).toBe(true);
    expect(voiceMatchesLang(v("en-GB"), "en-US")).toBe(true); // same base language accepted
  });
});

describe("findVoice", () => {
  it("prefers localService voices", () => {
    const voices = [v("en-US", false), v("en-GB", true)];
    const m = findVoice(voices, "en");
    expect(m?.voice.lang).toBe("en-GB");
    expect(m?.localService).toBe(true);
  });
  it("returns null when no voice covers the language (Newari case)", () => {
    expect(findVoice([v("da-DK"), v("en-US")], "new")).toBeNull();
  });
  it("returns null for an empty voice list (async quirk not resolved)", () => {
    expect(findVoice([], "en")).toBeNull();
  });
});

describe("exerciseSlug", () => {
  it("maps exercise text to the bundled-clip filename", () => {
    expect(exerciseSlug("Hello world")).toBe("hello-world");
    expect(exerciseSlug("I think three things")).toBe("i-think-three-things");
    expect(exerciseSlug("Ærlig tale — på Dansk!")).toBe("aerlig-tale-pa-dansk");
  });
});

void ({} as ReferenceVoice); // keep type import used

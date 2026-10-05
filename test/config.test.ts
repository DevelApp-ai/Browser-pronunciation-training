import { describe, expect, it } from "vitest";
import { modelConfigForLang, BASELINE_MODEL_ID } from "../src/model/config.ts";

function stubFetch(map: Record<string, number> = {}) {
  const calls: string[] = [];
  const stub = async (url: string, init?: RequestInit) => {
    calls.push(url);
    if (url.endsWith("/languages.json") && map[url] !== 404) {
      return new Response(
        JSON.stringify({
          en: { artifact: "en-phoneme", revision: "v1-en" },
          da: { artifact: "da-phoneme", revision: "v1-da" },
        }),
        { status: 200 },
      );
    }
    const ok = map[url] !== 404 && (init?.method === "HEAD" || url.includes("manifest.json"));
    return new Response(null, { status: ok ? 200 : 404 });
  };
  (globalThis as Record<string, unknown>).fetch = stub;
  return calls;
}

describe("modelConfigForLang (issue #10 revision map)", () => {
  it("falls back to the multilingual baseline when the map is missing", async () => {
    stubFetch({ "b1/languages.json": 404 });
    const cfg = await modelConfigForLang("en", "b1");
    expect(cfg.modelId).toBe(BASELINE_MODEL_ID);
    expect(cfg.local).toBe(false);
  });

  it("resolves a locally published artifact with its revision", async () => {
    stubFetch();
    const cfg = await modelConfigForLang("en", "b2");
    expect(cfg.modelId).toBe("en-phoneme/v1-en");
    expect(cfg.local).toBe(true);
    expect(cfg.entry?.revision).toBe("v1-en");
  });

  it("falls back to the baseline for a language not in the map", async () => {
    stubFetch();
    const cfg = await modelConfigForLang("xx", "b3");
    expect(cfg.modelId).toBe(BASELINE_MODEL_ID);
  });
});

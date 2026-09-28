import { describe, expect, it } from "vitest";
import { chunkPcm, frameRms, resampleTo16k, toMono, trimSilence, TARGET_SAMPLE_RATE } from "../src/audio/pcm.ts";

describe("resampleTo16k", () => {
  it("keeps the buffer untouched at the target rate", () => {
    const pcm = new Float32Array([1, 2, 3]);
    expect(resampleTo16k(pcm, TARGET_SAMPLE_RATE)).toBe(pcm);
  });

  it("downsamples 48k → 16k by 3×", () => {
    const pcm = new Float32Array(4800).fill(0.5);
    const out = resampleTo16k(pcm, 48000);
    expect(out.length).toBe(1600);
  });
});

describe("frameRms", () => {
  it("returns 0 for silence", () => {
    expect(frameRms(new Float32Array(100), 0, 100)).toBe(0);
  });
  it("returns ~0.707 for a full-scale square wave", () => {
    const pcm = new Float32Array(1000).map((_, i) => (i % 2 ? 1 : -1));
    expect(frameRms(pcm, 0, 1000)).toBeCloseTo(1, 2);
  });
});

describe("trimSilence", () => {
  it("trims leading and trailing silence but keeps speech", () => {
    const sr = TARGET_SAMPLE_RATE;
    const silence = new Float32Array(sr * 0.5); // 0.5 s silence
    const speech = new Float32Array(sr).map((_, i) => (i % 2 ? 0.5 : -0.5));
    const pcm = new Float32Array([...silence, ...speech, ...silence]);
    const trimmed = trimSilence(pcm, sr);
    expect(trimmed.length).toBeGreaterThan(sr * 0.8);
    expect(trimmed.length).toBeLessThan(pcm.length - sr * 0.5);
  });

  it("returns the original when no speech is detected", () => {
    const pcm = new Float32Array(1600);
    expect(trimSilence(pcm)).toBe(pcm);
  });
});

describe("chunkPcm", () => {
  it("splits long recordings on ≤30 s boundaries", () => {
    const sr = TARGET_SAMPLE_RATE;
    const pcm = new Float32Array(sr * 65); // 65 s
    const chunks = chunkPcm(pcm, sr);
    expect(chunks.length).toBe(3);
    expect(chunks[0]!.length).toBe(sr * 30);
    expect(chunks.reduce((n, c) => n + c.length, 0)).toBe(pcm.length);
  });
  it("returns a single chunk for short recordings", () => {
    expect(chunkPcm(new Float32Array(100)).length).toBe(1);
  });
});

describe("toMono", () => {
  it("averages stereo channels", () => {
    const left = new Float32Array([1, 2]);
    const right = new Float32Array([3, 4]);
    expect(Array.from(toMono([left, right]))).toEqual([2, 3]);
  });
});

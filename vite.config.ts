import { defineConfig } from "vite";

// Project-site base path — GitHub Pages serves this repo at
// https://<owner>.github.io/Browser-pronunciation-training/
const BASE = "/Browser-pronunciation-training/";

export default defineConfig({
  base: BASE,
  build: { target: "es2022" },
  worker: { format: "es" as const },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});

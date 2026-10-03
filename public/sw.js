/* Service worker — issue #18 (offline PWA).
 * Strategy:
 *  - App shell (index.html + hashed assets): cache-first, refreshed in background.
 *  - Phoneme packs + languages.json: cache-first (small, versioned with the app).
 *  - Model weights are NOT cached here — transformers.js manages its own
 *    Cache Storage for ONNX files (and HF downloads are cross-origin).
 * Lazy per-language loading: packs are fetched only when a language is used.
 */
const VERSION = "bpt-v2";
const SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./packs/en.json",
  "./packs/da.json",
  "./packs/ne.json",
  "./packs/new.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  // Same-origin only: Hub/CDN downloads are handled by transformers.js caches.
  if (url.origin !== self.location.origin) return;
  // Skip non-GET precacheable APIs / dev sockets.
  if (url.pathname.startsWith("/@") || url.pathname.includes("/node_modules/")) return;

  event.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const cached = await cache.match(req, { ignoreSearch: true });
    if (cached) {
      // Stale-while-revalidate so deploys land without losing offline.
      event.waitUntil(fetch(req).then((res) => res.ok && cache.put(req, res.clone())).catch(() => {}));
      return cached;
    }
    const res = await fetch(req);
    if (res.ok && (req.destination !== "" || url.pathname.endsWith(".html") || url.pathname === self.location.pathname)) {
      cache.put(req, res.clone());
    } else if (res.ok) {
      cache.put(req, res.clone());
    }
    return res;
  })());
});

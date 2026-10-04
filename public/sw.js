/* Service worker — issue #18 (offline PWA), v3.
 * Strategy:
 *  - Navigations / HTML: NETWORK-FIRST. A deploy replaces the hashed assets
 *    on GitHub Pages, so a cached index.html from an older deploy can point
 *    at assets that no longer exist (v2 served it cache-first and left
 *    returning users with a blank page). HTML is tiny; always prefer the
 *    network and only fall back to the cached shell when offline.
 *  - Hashed assets + packs + manifest: cache-first (content-addressed by
 *    URL, refreshed in the background). Model weights are NOT cached here —
 *    transformers.js manages its own Cache Storage for ONNX files.
 * Lazy per-language loading: packs are fetched only when a language is used.
 */
const VERSION = "bpt-v3";
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

  // HTML documents (page navigations): network-first so deploys land.
  const isDocument = req.mode === "navigate" || req.destination === "document" || url.pathname.endsWith(".html");

  event.respondWith((async () => {
    const cache = await caches.open(VERSION);
    if (isDocument) {
      try {
        const res = await fetch(req);
        if (res.ok) cache.put(req, res.clone());
        return res;
      } catch {
        const shell = (await cache.match(req, { ignoreSearch: true })) || (await cache.match("./index.html"));
        if (shell) return shell;
        return new Response("Offline and no cached app shell.", { status: 503, headers: { "Content-Type": "text/plain" } });
      }
    }
    const cached = await cache.match(req, { ignoreSearch: true });
    if (cached) {
      // Stale-while-revalidate so updated packs/manifests land without losing offline.
      event.waitUntil(fetch(req).then((res) => res.ok && cache.put(req, res.clone())).catch(() => {}));
      return cached;
    }
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  })());
});

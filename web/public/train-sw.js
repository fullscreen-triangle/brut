// BRUT Train — minimal service worker.
//
// Its only job is to make the training instance installable ("Add to Home
// Screen" → opens standalone, like an app) and to let the shell launch when
// the network is flaky at the gym. It caches the shell on install and serves
// cached responses when offline, but never gets between the app and the live
// camera/microphone streams (those are not fetches).

const CACHE = 'brut-train-v1';
const SHELL = ['/train.html', '/train.webmanifest', '/train-icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  // Network-first for navigations so a live deploy is picked up; fall back to
  // the cached shell offline. Cache-first for other static GETs.
  if (req.mode === 'navigate') {
    event.respondWith(fetch(req).catch(() => caches.match('/train.html')));
    return;
  }
  event.respondWith(
    caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
      return res;
    }).catch(() => hit))
  );
});

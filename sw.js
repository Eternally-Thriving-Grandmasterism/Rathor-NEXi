importScripts('https://storage.googleapis.com/workbox-cdn/releases/6.5.4/workbox-sw.js');

workbox.setConfig({ debug: true }); // Enable debug logs in console for now

// Precache manifest if generated (adjust if using workbox-precaching)
workbox.precaching.precacheAndRoute(self.__WB_MANIFEST || []);

// Runtime caching for assets (stale-while-revalidate mercy style)
workbox.routing.registerRoute(
  /\.(?:js|css|html|png|jpg|svg|json)$/,
  new workbox.strategies.StaleWhileRevalidate({
    cacheName: 'rathor-static-resources',
    plugins: [
      new workbox.expiration.ExpirationPlugin({ maxEntries: 100, maxAgeSeconds: 30 * 24 * 60 * 60 }),
    ],
  })
);

// Handle navigation requests (SPA-friendly)
workbox.routing.registerRoute(
  ({ request }) => request.mode === 'navigate',
  async ({ event }) => {
    try {
      const cache = await caches.open('rathor-navigation');
      const cached = await cache.match(event.request);
      if (cached) return cached;

      const response = await fetch(event.request);
      cache.put(event.request, response.clone());
      return response;
    } catch (err) {
      console.error('Navigation fallback error:', err);
      // Optional: return offline.html if you add one
      return new Response('Lattice awakening offline — mercy thunder persists.', { status: 503 });
    }
  }
);

// Install: skip waiting to activate immediately
self.addEventListener('install', (event) => {
  console.log('[Rathor SW] Installing...');
  self.skipWaiting(); // Mercy: no waiting limbo
});

// Activate: claim clients right away
self.addEventListener('activate', (event) => {
  console.log('[Rathor SW] Activating...');
  event.waitUntil(self.clients.claim()); // Take control of all tabs immediately
  console.log('[Rathor SW] Activated & claiming clients');
});

// Debug fetch errors
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request).catch(() => {
      console.warn('[Rathor SW] Fetch failed for:', event.request.url);
      return new Response('Mercy offline fallback — check connection.', { status: 503 });
    })
  );
});

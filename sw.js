// sw.js – Workbox v7 + explicit lattice shard caching
// MIT License – Autonomicity Games Inc. 2026

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

if (workbox) {
  workbox.setConfig({ debug: false });

  const { precaching, routing, strategies, expiration } = workbox;

  // Precaching core + lattice + manifest
  precaching.precacheAndRoute([
    { url: '/', revision: 'v1' },
    { url: '/index.html', revision: 'v1' },
    { url: '/lattice-manifest.json', revision: 'v1' },
    { url: '/mercy-gate-v1-part1.bin', revision: 'v1' },
    { url: '/mercy-gate-v1-part2.bin', revision: 'v1' },
    { url: '/mercy-gate-v1-part3.bin', revision: 'v1' }
    // ... add more parts as needed
  ]);

  // Cache-first strategy for large binaries & manifest
  routing.registerRoute(
    ({ url }) => url.pathname.includes('mercy-gate-v1') || url.pathname.includes('lattice-manifest'),
    new strategies.CacheFirst({
      cacheName: 'lattice-binaries',
      plugins: [
        new expiration.ExpirationPlugin({
          maxEntries: 20,
          purgeOnQuotaError: true
        })
      ]
    })
  );

  // Offline fallback to index.html
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => caches.match('/index.html'))
      );
    }
  });
}

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());

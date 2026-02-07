// sw.js — Rathor-NEXi Mercy-gated Offline Service Worker
// Workbox v7 — stable, offline-first, PWA-ready

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

workbox.setConfig({ debug: false });

// Precache core assets (updated on build/deploy)
workbox.precaching.precacheAndRoute(self.__WB_MANIFEST || [
  { url: '/index.html', revision: null },
  { url: '/chat.html', revision: null },
  { url: '/css/main.css', revision: null },
  { url: '/js/common.js', revision: null },
  { url: '/js/chat.js', revision: null },
  { url: '/locales/languages.json', revision: null },
  { url: '/icons/thunder-192.png', revision: null },
  { url: '/icons/thunder-512.png', revision: null }
]);

// Navigation fallback to offline.html
workbox.routing.setCatchHandler(({ event }) => {
  switch (event.request.destination) {
    case 'document':
      return caches.match('/offline.html');
    default:
      return Response.error();
  }
});

// Cache-first for static assets
workbox.routing.registerRoute(
  /\.(?:png|jpg|jpeg|svg|gif|ico|woff2?|ttf|css|js)$/,
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-static',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 60 * 24 * 60 * 60 // 60 days
      })
    ]
  })
);

// Cache Transformers.js models (once downloaded)
workbox.routing.registerRoute(
  ({ url }) => url.href.includes('@xenova/transformers'),
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-models',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
      })
    ]
  })
);

// Network-first for future dynamic API calls
workbox.routing.registerRoute(
  /\/api\//,
  new workbox.strategies.NetworkFirst({
    cacheName: 'rathor-dynamic',
    networkTimeoutSeconds: 3
  })
);

// Clean old caches on activate
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key.startsWith('rathor-') && !key.includes('v1'))
          .map(key => caches.delete(key))
      );
    })
  );
});

// Claim clients immediately
self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});

console.log('[Rathor SW] Mercy thunder online — eternal lattice protected offline ⚡️');

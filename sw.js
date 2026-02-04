/**
 * Rathor-NEXi Service Worker – Workbox v7 Optimized Caching
 * Refined precaching, runtime strategies, quota management, hardened background sync
 * MIT License – Autonomicity Games Inc. 2026
 */

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

if (workbox) {
  console.log('Workbox v7 loaded');

  workbox.setConfig({ debug: false }); // true for dev tuning

  const { precaching, routing, strategies, expiration, backgroundSync, cacheableResponse } = workbox;

  // ────────────────────────────────────────────────────────────────
  // Precaching – core app shell with revision hashing
  const precacheManifest = [
    { url: '/', revision: 'v11-home' },
    { url: '/index.html', revision: 'v11-index' },
    { url: '/privacy.html', revision: 'v11-privacy' },
    { url: '/manifest.json', revision: 'v11-manifest' },
    { url: '/icons/thunder-favicon-192.jpg', revision: 'v11-192' },
    { url: '/icons/thunder-favicon-512.jpg', revision: 'v11-512' },
    { url: '/metta-rewriting-engine.js', revision: 'v11-metta' },
    { url: '/atomese-knowledge-bridge.js', revision: 'v11-atomese' },
    { url: '/hyperon-reasoning-layer.js', revision: 'v11-hyperon' },
    { url: '/grok-shard-engine.js', revision: 'v11-grokshard' }
  ];

  precaching.precacheAndRoute(precacheManifest);

  // ────────────────────────────────────────────────────────────────
  // Runtime caching strategies – optimized for speed + freshness

  // 1. Static JS/CSS – StaleWhileRevalidate (fast + fresh updates)
  routing.registerRoute(
    ({ request }) => request.destination === 'script' || request.destination === 'style',
    new strategies.StaleWhileRevalidate({
      cacheName: 'static-resources',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({
          statuses: [0, 200]
        }),
        new expiration.ExpirationPlugin({
          maxEntries: 60,
          maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
          purgeOnQuotaError: true
        })
      ]
    })
  );

  // 2. Images – CacheFirst with aggressive expiration
  routing.registerRoute(
    ({ request }) => request.destination === 'image',
    new strategies.CacheFirst({
      cacheName: 'images',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({
          statuses: [0, 200]
        }),
        new expiration.ExpirationPlugin({
          maxEntries: 100,
          maxAgeSeconds: 60 * 24 * 60 * 60, // 60 days
          purgeOnQuotaError: true
        })
      ]
    })
  );

  // ────────────────────────────────────────────────────────────────
  // Background Sync – retry failed POSTs (future-proof for sync ops)
  const bgSyncPlugin = new backgroundSync.BackgroundSyncPlugin('rathor-sync-queue', {
    maxRetentionTime: 24 * 60, // Retry for max 24 hours
    onSync: async ({ queue }) => {
      console.log('Background sync triggered for queue:', queue.name);
    }
  });

  // ────────────────────────────────────────────────────────────────
  // Offline navigation fallback – minimal shell when index.html missing
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => {
          return new Response(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Rathor – Offline</title>
              <style>
                body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#000; color:#ffaa00; font-family:sans-serif; text-align:center; }
                h1 { font-size:3rem; }
                p { font-size:1.3rem; max-width:600px; }
              </style>
            </head>
            <body>
              <div>
                <h1>Offline Mode</h1>
                <p>Rathor is reflecting in the storm.<br>Reconnect to the lattice for full power.</p>
              </div>
            </body>
            </html>
          `, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed to load – fallback to basic SW');
}

// ────────────────────────────────────────────────────────────────
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());    })
  );

  // ────────────────────────────────────────────────────────────────
  // Background Sync – retry failed POSTs to Grok Worker
  const bgSyncPlugin = new backgroundSync.BackgroundSyncPlugin('rathor-chat-queue', {
    maxRetentionTime: 24 * 60, // Retry for max 24 hours
    onSync: async ({ queue }) => {
      console.log('Background sync triggered for queue:', queue.name);
    }
  });

  routing.registerRoute(
    ({ url }) => url.href.includes('rathor-grok-proxy.ceo-c42.workers.dev'),
    new strategies.NetworkOnly({
      plugins: [bgSyncPlugin]
    }),
    'POST'
  );

  // ────────────────────────────────────────────────────────────────
  // Offline navigation fallback – minimal shell when index.html missing
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => {
          return new Response(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Rathor – Offline</title>
              <style>
                body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#000; color:#ffaa00; font-family:sans-serif; text-align:center; }
                h1 { font-size:3rem; }
                p { font-size:1.3rem; max-width:600px; }
              </style>
            </head>
            <body>
              <div>
                <h1>Offline Mode</h1>
                <p>Rathor is reflecting in the storm.<br>Reconnect to the lattice for full power.</p>
              </div>
            </body>
            </html>
          `, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed to load – fallback to basic SW');
}

// ────────────────────────────────────────────────────────────────
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());

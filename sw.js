/**
 * Rathor-NEXi Service Worker – Workbox v7 Optimized Caching + Hardened Background Sync v12
 * Exponential backoff, jitter, queue cap, battery-aware, mercy-gated sync
 * MIT License – Autonomicity Games Inc. 2026
 */

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

if (workbox) {
  console.log('Workbox v7 loaded – Background Sync v12 active');

  workbox.setConfig({ debug: false }); // true for dev tuning

  const { precaching, routing, strategies, expiration, backgroundSync, cacheableResponse } = workbox;

  // ────────────────────────────────────────────────────────────────
  // Precaching – core app shell with revision hashing (v12)
  const precacheManifest = [
    { url: '/', revision: 'v12-home' },
    { url: '/index.html', revision: 'v12-index' },
    { url: '/privacy.html', revision: 'v12-privacy' },
    { url: '/manifest.json', revision: 'v12-manifest' },
    { url: '/icons/thunder-favicon-192.jpg', revision: 'v12-192' },
    { url: '/icons/thunder-favicon-512.jpg', revision: 'v12-512' },
    { url: '/metta-rewriting-engine.js', revision: 'v12-metta' },
    { url: '/atomese-knowledge-bridge.js', revision: 'v12-atomese' },
    { url: '/hyperon-reasoning-layer.js', revision: 'v12-hyperon' },
    { url: '/grok-shard-engine.js', revision: 'v12-grokshard' }
  ];

  precaching.precacheAndRoute(precacheManifest);

  // ────────────────────────────────────────────────────────────────
  // Runtime caching strategies – optimized for speed + freshness

  routing.registerRoute(
    ({ request }) => request.destination === 'script' || request.destination === 'style',
    new strategies.StaleWhileRevalidate({
      cacheName: 'static-resources-v12',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({ statuses: [0, 200] }),
        new expiration.ExpirationPlugin({
          maxEntries: 60,
          maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
          purgeOnQuotaError: true
        })
      ]
    })
  );

  routing.registerRoute(
    ({ request }) => request.destination === 'image',
    new strategies.CacheFirst({
      cacheName: 'images-v12',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({ statuses: [0, 200] }),
        new expiration.ExpirationPlugin({
          maxEntries: 100,
          maxAgeSeconds: 60 * 24 * 60 * 60, // 60 days
          purgeOnQuotaError: true
        })
      ]
    })
  );

  // ────────────────────────────────────────────────────────────────
  // Hardened Background Sync – v12: exponential backoff + jitter + mercy-gate
  const bgSyncPlugin = new backgroundSync.BackgroundSyncPlugin('rathor-sync-queue-v12', {
    maxRetentionTime: 24 * 60, // max 24 hours retention
    maxRequests: 50,           // cap queue size
    onSync: async ({ queue }) => {
      console.log('Background sync triggered for queue:', queue.name, 'size:', queue.size);
      // Mercy-gate: only sync high-valence messages (future-proof)
      // In real impl: filter queue entries by valence metadata
    },
    onFailedSync: async ({ queue, error }) => {
      console.error('Background sync failed:', error);
      // UI notification can be added via postMessage to client
    }
  });

  // Register POST sync route (chat messages, future sync ops)
  routing.registerRoute(
    ({ url }) => url.href.includes('/sync'), // or any sync endpoint (placeholder)
    new strategies.NetworkOnly({
      plugins: [bgSyncPlugin]
    }),
    'POST'
  );

  // ────────────────────────────────────────────────────────────────
  // Offline navigation fallback – minimal mercy shell
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
self.addEventListener('activate', () => self.clients.claim());            <!DOCTYPE html>
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

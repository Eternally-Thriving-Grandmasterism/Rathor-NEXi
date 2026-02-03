/**
 * Rathor-NEXi Service Worker – v8
 * Sovereign offline caching + background sync for chat queue
 * MIT License – Autonomicity Games Inc. 2026
 */

const CACHE_VERSION = 'rathor-nexi-cache-v8';
const CACHE_NAME = `static-${CACHE_VERSION}`;

const CORE_ASSETS = [
  '/',
  '/index.html',
  '/privacy.html',
  '/thanks.html',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/metta-rewriting-engine.js',
  '/atomese-knowledge-bridge.js',
  '/hyperon-reasoning-layer.js',
  // Add any other critical JS/CSS here when externalized
];

// ────────────────────────────────────────────────────────────────
// Install – cache core assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(CORE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// ────────────────────────────────────────────────────────────────
// Activate – clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys
          .filter(key => key.startsWith('static-') && key !== CACHE_NAME)
          .map(key => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// ────────────────────────────────────────────────────────────────
// Fetch – Cache-First for static, Network-First for dynamic
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Skip non-GET requests and non-same-origin (e.g. Worker calls)
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) {
    event.respondWith(fetch(event.request));
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(cached => {
        // Cache hit → return cached
        if (cached) return cached;

        // Network fetch → cache successful responses
        return fetch(event.request).then(networkResponse => {
          if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
            return networkResponse;
          }

          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME)
            .then(cache => cache.put(event.request, responseToCache));

          return networkResponse;
        }).catch(() => {
          // Offline fallback – serve index.html for app shell
          return caches.match('/index.html');
        });
      })
  );
});

// ────────────────────────────────────────────────────────────────
// Background Sync – retry queued chat messages
self.addEventListener('sync', event => {
  if (event.tag === 'sync-chat-messages') {
    event.waitUntil(syncQueuedMessages());
  }
});

async function syncQueuedMessages() {
  const db = await openDB();
  const tx = db.transaction('queuedMessages', 'readwrite');
  const store = tx.objectStore('queuedMessages');
  const messages = await store.getAll();

  for (const msg of messages) {
    try {
      const response = await fetch(msg.url || 'https://rathor-grok-proxy.ceo-c42.workers.dev', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg.payload)
      });

      if (response.ok) {
        await store.delete(msg.id);
        // Notify client UI (optional – postMessage)
        const clients = await self.clients.matchAll();
        clients.forEach(client => client.postMessage({ type: 'sync-success', id: msg.id }));
      }
    } catch (err) {
      // Keep queued – retry next sync event
      console.log('Sync retry failed, keeping queued:', err);
    }
  }
}

// IndexedDB helper for queue (used by sync)
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('rathorChatDB', 3);
    req.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('messages')) {
        db.createObjectStore('messages', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('queuedMessages')) {
        db.createObjectStore('queuedMessages', { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

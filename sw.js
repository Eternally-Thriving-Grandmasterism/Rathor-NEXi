/**
 * Rathor-NEXi Service Worker – v9 Optimized
 * Stale-While-Revalidate + Cache-First hybrid, hardened background sync, minimal offline fallback
 * MIT License – Autonomicity Games Inc. 2026
 */

const CACHE_VERSION = 'rathor-nexi-v9';
const CACHE_NAME = `static-${CACHE_VERSION}`;
const RUNTIME_CACHE = 'runtime-cache';

const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/privacy.html',
  '/thanks.html',
  '/manifest.json',
  '/icons/thunder-favicon-192.jpg',
  '/icons/thunder-favicon-512.jpg',
  '/metta-rewriting-engine.js',
  '/atomese-knowledge-bridge.js',
  '/hyperon-reasoning-layer.js'
  // Add more critical JS/CSS if externalized later
];

// ────────────────────────────────────────────────────────────────
// Install – precache core assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// ────────────────────────────────────────────────────────────────
// Activate – clean old caches + claim clients
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys
          .filter(key => key.startsWith('rathor-nexi-') && key !== CACHE_NAME && key !== RUNTIME_CACHE)
          .map(key => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// ────────────────────────────────────────────────────────────────
// Fetch – Stale-While-Revalidate for precached + Cache-First for runtime
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Bypass non-GET, cross-origin, or non-app requests
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Precached assets → Cache-First
  if (PRECACHE_ASSETS.includes(url.pathname) || url.pathname === '/') {
    event.respondWith(
      caches.match(event.request)
        .then(cached => cached || fetchAndCache(event.request))
        .catch(() => caches.match('/index.html')) // offline fallback
    );
    return;
  }

  // Runtime assets (JS bridges, future dynamic) → Stale-While-Revalidate
  event.respondWith(
    caches.match(event.request)
      .then(cached => {
        const fetchPromise = fetch(event.request).then(networkResponse => {
          if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
            return networkResponse;
          }
          caches.open(RUNTIME_CACHE)
            .then(cache => cache.put(event.request, networkResponse.clone()));
          return networkResponse;
        }).catch(() => cached); // fallback to cache on network failure

        return cached || fetchPromise;
      })
  );
});

// Helper: fetch + cache + return response
async function fetchAndCache(request) {
  const response = await fetch(request);
  if (response && response.status === 200 && response.type === 'basic') {
    const responseToCache = response.clone();
    caches.open(CACHE_NAME).then(cache => cache.put(request, responseToCache));
  }
  return response;
}

// ────────────────────────────────────────────────────────────────
// Background Sync – retry queued messages with exponential backoff
self.addEventListener('sync', event => {
  if (event.tag === 'sync-chat-messages') {
    event.waitUntil(syncQueuedMessagesWithRetry());
  }
});

async function syncQueuedMessagesWithRetry() {
  const db = await openDB();
  const tx = db.transaction('queuedMessages', 'readwrite');
  const store = tx.objectStore('queuedMessages');
  const messages = await store.getAll();

  let backoff = 1000; // start 1s

  for (const msg of messages) {
    const attempts = msg.attempts || 0;
    if (attempts >= 5) {
      // Max retries reached – mark failed or delete
      await store.delete(msg.id);
      continue;
    }

    try {
      const response = await fetch(msg.url || 'https://rathor-grok-proxy.ceo-c42.workers.dev', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg.payload)
      });

      if (response.ok) {
        await store.delete(msg.id);
        notifyClients({ type: 'sync-success', id: msg.id });
      } else {
        msg.attempts = attempts + 1;
        await store.put(msg);
        await delay(backoff);
        backoff = Math.min(backoff * 2, 300000); // cap at 5 min
      }
    } catch (err) {
      msg.attempts = attempts + 1;
      await store.put(msg);
      await delay(backoff);
      backoff = Math.min(backoff * 2, 300000);
    }
  }
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function notifyClients(message) {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => client.postMessage(message));
  });
}

// IndexedDB helper for queue
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

// public/sw.js – Custom Service Worker with Background Sync v1.1
// Caches core assets + offline page + queues mutations for sync
// MIT License – Autonomicity Games Inc. 2026

const CACHE_NAME = 'rathor-nexi-cache-v1';
const OFFLINE_URL = '/offline.html';

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
  // Add critical chunks/models here after build
  // e.g. '/assets/entry/main-[hash].js',
  // '/models/gesture-transformer-qat-int8.onnx'
];

// Install event – precache essentials
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Pre-caching files');
      return cache.addAll(PRECACHE_URLS);
    }).then(() => self.skipWaiting())
  );
});

// Activate event – clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event – CacheFirst for static, NetworkFirst for navigation with offline fallback
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match(OFFLINE_URL);
      })
    );
    return;
  }

  // CacheFirst for known static assets
  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      if (cachedResponse) return cachedResponse;

      return fetch(event.request).then(networkResponse => {
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, responseToCache);
        });

        return networkResponse;
      });
    })
  );
});

// Background Sync – queue mutations when offline
self.addEventListener('sync', event => {
  if (event.tag === 'rathor-sync-mutations') {
    event.waitUntil(syncMutations());
  }
});

async function syncMutations() {
  const db = await openDB();
  const tx = db.transaction('pendingMutations', 'readwrite');
  const store = tx.objectStore('pendingMutations');
  const pending = await store.getAll();

  for (const mutation of pending) {
    try {
      const response = await fetch(mutation.url, {
        method: mutation.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mutation.payload)
      });

      if (response.ok) {
        await store.delete(mutation.id);
        console.log('[SW] Synced mutation:', mutation.id);
      } else {
        throw new Error('Sync failed');
      }
    } catch (err) {
      console.warn('[SW] Mutation sync failed:', mutation.id, err);
      // Re-queue for next sync
    }
  }

  await tx.done;
}

// IndexedDB for pending mutations
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('rathor-nexi-db', 1);

    request.onupgradeneeded = event => {
      const db = event.target.result;
      db.createObjectStore('pendingMutations', { keyPath: 'id', autoIncrement: true });
    };

    request.onsuccess = event => resolve(event.target.result);
    request.onerror = event => reject(event.target.error);
  });
}

// Listen for offline mutations from client
self.addEventListener('message', event => {
  if (event.data.type === 'QUEUE_MUTATION') {
    queueMutation(event.data.payload);
  }
});

async function queueMutation(payload) {
  const db = await openDB();
  const tx = db.transaction('pendingMutations', 'readwrite');
  await tx.objectStore('pendingMutations').add(payload);
  await tx.done;

  // Register sync if not already
  if ('sync' in self.registration) {
    try {
      await self.registration.sync.register('rathor-sync-mutations');
      console.log('[SW] Sync registered for queued mutation');
    } catch (err) {
      console.warn('[SW] Sync registration failed:', err);
    }
  }
}

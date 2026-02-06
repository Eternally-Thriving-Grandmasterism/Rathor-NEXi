// public/sw.js – Service Worker for offline sovereignty v1.1
const CACHE_NAME = 'rathor-mercy-cache-v1.1';
const urlsToCache = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/src/integrations/gesture-recognition/GestureOverlay.tsx',
  '/node_modules/@tensorflow-models/handpose/model/model.json',
  '/node_modules/@tensorflow/tfjs-backend-webgl/tf-backend-webgl.min.js',
  // add more critical assets as we build
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
      .catch(() => caches.match('/offline.html')) // serve OfflineMercySanctuary
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(name => name !== CACHE_NAME)
          .map(name => caches.delete(name))
      );
    })
  );
});

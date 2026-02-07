// workbox-config.js – Workbox build-time config for Rathor-NEXi
module.exports = {
  globDirectory: './',
  globPatterns: [
    'index.html',
    'manifest.json',
    'offline.html',
    'icons/thunder-*.png',
    'src/storage/*.js'
    // Add more patterns as monorepo grows
    // '**/*.{js,css,html,png,svg,woff2,ttf}'
  ],
  swDest: 'sw.js',
  skipWaiting: true,
  clientsClaim: true,
  cleanupOutdatedCaches: true,
  runtimeCaching: [
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|ico|woff2?|ttf)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'rathor-static',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 24 * 60 * 60 // 60 days
        }
      }
    },
    {
      urlPattern: /\/api\//,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'rathor-dynamic',
        networkTimeoutSeconds: 3
      }
    },
    {
      urlPattern: ({ url }) => url.href.includes('@xenova/transformers'),
      handler: 'CacheFirst',
      options: {
        cacheName: 'rathor-models',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
        }
      }
    }
  ],
  navigateFallback: '/offline.html',
  navigateFallbackDenylist: [/^\/api\//]
};

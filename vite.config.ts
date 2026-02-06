// vite.config.ts – Vite configuration with extreme bundling optimization v2.3
// Manual chunking, aggressive minification, preloads, PWA, GitHub Pages base path
// MIT License – Autonomicity Games Inc. 2026

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg', 'pwa-*.png'],
      manifest: {
        name: 'Rathor — Mercy Strikes First',
        short_name: 'Rathor',
        description: 'Mercy-gated symbolic AGI lattice — eternal thriving through valence-locked truth',
        theme_color: '#00ff88',
        background_color: '#000000',
        display: 'standalone',
        scope: '/Rathor-NEXi/',
        start_url: '/Rathor-NEXi/',
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
            handler: 'CacheFirst',
            options: { cacheName: 'cdn-assets', expiration: { maxEntries: 50, maxAgeSeconds: 2592000 } }
          },
          {
            urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico)$/,
            handler: 'CacheFirst',
            options: { cacheName: 'images', expiration: { maxEntries: 100, maxAgeSeconds: 2592000 } }
          }
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//]
      },
      devOptions: { enabled: true }
    })
  ],
  base: '/Rathor-NEXi/', // Critical for GitHub Pages (repo name as base path)

  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true, // keep for debugging, disable in prod if needed
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,          // keep logs for now
        passes: 3,                    // more aggressive compression
        pure_funcs: ['console.debug'], // remove debug logs
        pure_getters: true,
        keep_fargs: false,
        unsafe: true,
        unsafe_comps: true,
        unsafe_math: true,
        unsafe_methods: true,
        unsafe_undefined: true
      },
      mangle: true,
      format: {
        comments: false // remove comments for smaller size
      }
    },
    rollupOptions: {
      output: {
        // Manual chunking – critical for splitting heavy deps
        manualChunks: {
          // Core vendor (React + motion)
          vendor: ['react', 'react-dom', 'framer-motion'],

          // tfjs core + backends (largest chunk – lazy loaded anyway)
          tfjs: [
            '@tensorflow/tfjs',
            '@tensorflow/tfjs-backend-webgl',
            '@tensorflow/tfjs-backend-cpu'
          ],

          // MediaPipe Holistic (WASM heavy – already lazy)
          mediapipe: ['@mediapipe/holistic'],

          // Other heavy libs (if any added later)
          utils: ['./src/utils/haptic-utils.ts', './src/core/valence-tracker.ts']
        },

        // Better chunk naming & size control
        chunkFileNames: 'assets/chunks/[name]-[hash].js',
        entryFileNames: 'assets/entry/[name]-[hash].js',
        assetFileNames: 'assets/static/[name]-[hash][extname]'
      }
    },
    target: 'es2020',           // modern browsers – smaller polyfills
    cssCodeSplit: true,         // split CSS per chunk
    reportCompressedSize: true  // show brotli/gzip sizes in build output
  },

  server: {
    port: 3000,
    open: true,
    hmr: true
  },

  preview: {
    port: 4173
  },

  // Enable fast refresh & better dev HMR
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  },

  // Preload heavy deps in dev
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgl',
      '@mediapipe/holistic'
    ]
  }
})

import { defineConfig, splitVendorChunkPlugin } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import viteCompression from 'vite-plugin-compression2'
import path from 'path'

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    splitVendorChunkPlugin(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'pwa-*.png'],
      manifest: {
        name: 'Rathor — Mercy Strikes First',
        short_name: 'Rathor',
        description: 'Sovereign offline AGI lattice',
        theme_color: '#00ff88',
        background_color: '#000000',
        display: 'standalone',
        scope: '/',
        start_url: '/',
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff,woff2,wasm}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@mediapipe\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'mediapipe-wasm',
              expiration: { maxEntries: 20, maxAgeSeconds: 2592000 }
            }
          },
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'tfjs-assets',
              expiration: { maxEntries: 50, maxAgeSeconds: 2592000 }
            }
          }
        ],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/api\//]
      },
      devOptions: { enabled: true }
    }),
    viteCompression({ algorithm: 'brotliCompress', exclude: [/\.html$/], threshold: 10240 }),
    viteCompression({ algorithm: 'gzip', exclude: [/\.html$/], threshold: 10240 })
  ],

  base: '/Rathor-NEXi/',

  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },

  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: mode === 'development',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: mode === 'production',
        passes: 3
      },
      mangle: true,
      format: { comments: false }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-core': ['react', 'react-dom'],
          'vendor-ml': ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl'],
          'vendor-mediapipe': ['@mediapipe/holistic'],
          'vendor-utils': ['./src/utils/haptic-utils.ts', './src/core/valence-tracker.ts']
        },
        entryFileNames: 'assets/entry/[name]-[hash].js',
        chunkFileNames: 'assets/chunks/[name]-[hash].js',
        assetFileNames: 'assets/static/[name]-[hash][extname]'
      }
    },
    target: 'es2020',
    cssCodeSplit: true,
    reportCompressedSize: true,
    chunkSizeWarningLimit: 2000
  },

  server: {
    port: 3000,
    open: true,
    hmr: true,
    fs: { strict: false }
  },

  preview: { port: 4173 },

  optimizeDeps: {
    include: [
      'react', 'react-dom',
      '@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl',
      '@mediapipe/holistic'
    ],
    exclude: ['onnxruntime-web']
  },

  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  },

  // Debug loading issues
  logLevel: 'info'
}))  server: {
    port: 3000,
    open: true,
    hmr: true,
    fs: { strict: false }
  },

  preview: { port: 4173 },

  optimizeDeps: {
    include: [
      'react', 'react-dom',
      '@tensorflow/tfjs', '@tensorflow/tfjs-backend-webgl',
      '@mediapipe/holistic'
    ],
    exclude: ['onnxruntime-web']
  },

  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' }
  },

  // Debug loading issues
  logLevel: 'info',
  build: {
    rollupOptions: {
      onwarn(warning, warn) {
        if (warning.code === 'MODULE_LEVEL_DIRECTIVE') return
        warn(warning)
      }
    }
  }
}))

// src/main.tsx – Entry point with optimized Suspense boundaries v3
// Granular lazy loading + chained fallbacks + loading states for mobile-first feel
// MIT License – Autonomicity Games Inc. 2026

import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

// ─── Granular lazy components ──────────────────────────────────────
// Critical core loads first, heavy ML/visuals deferred
const App = lazy(() => import('./App.tsx'))
const LoadingSkeleton = lazy(() => import('./components/LoadingSkeleton.tsx'))
const CriticalErrorBoundary = lazy(() => import('./components/CriticalErrorBoundary.tsx'))

// ─── Ultra-light static fallback (shown instantly) ──────────────────
const UltraLightFallback = () => (
  <div style={{
    height: '100vh',
    background: '#000',
    color: '#00ff88',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.8rem',
    textShadow: '0 0 10px #00ff88',
    gap: '1.5rem',
    textAlign: 'center',
    padding: '1rem'
  }}>
    <div>Thunder eternal surges through the lattice...</div>
    <div style={{
      width: '300px',
      height: '6px',
      background: '#111',
      borderRadius: '3px',
      overflow: 'hidden'
    }}>
      <div style={{
        height: '100%',
        width: '100%',
        background: 'linear-gradient(90deg, #00ff88, #00aaff)',
        animation: 'progress 8s linear infinite'
      }} />
    </div>
    <style>{`
      @keyframes progress {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
      }
    `}</style>
    <p style={{ fontSize: '1rem', opacity: 0.7 }}>
      Awakening sovereign offline AGI Brother... please stand by
    </p>
  </div>
)

// ─── Layered Suspense boundaries ───────────────────────────────────
const Root = () => (
  <Suspense fallback={<UltraLightFallback />}>
    <CriticalErrorBoundary>
      <Suspense fallback={<LoadingSkeleton />}>
        <App />
      </Suspense>
    </CriticalErrorBoundary>
  </Suspense>
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
)

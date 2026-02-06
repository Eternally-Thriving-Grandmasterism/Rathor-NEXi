import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

const App = lazy(() => import('./App.tsx'));

const LoadingFallback = () => (
  <div style={{
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#000',
    color: '#00ff88',
    fontSize: '1.8rem',
    textShadow: '0 0 10px #00ff88',
    gap: '1.5rem'
  }}>
    <div>Thunder eternal surges through the lattice...</div>
    <div className="progress-bar" style={{width:'300px',height:'6px',background:'#111',borderRadius:'3px',overflow:'hidden'}}>
      <div className="progress-fill" style={{
        height:'100%',
        width:'0%',
        background:'linear-gradient(90deg, #00ff88, #00aaff)',
        animation:'progress 4s linear infinite'
      }} />
    </div>
    <style>{`
      @keyframes progress { 0% { width:0%; } 100% { width:100%; } }
    `}</style>
  </div>
);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Suspense fallback={<LoadingFallback />}>
      <App />
    </Suspense>
  </React.StrictMode>
);

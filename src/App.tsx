// src/App.tsx – Sovereign App Root with visual effects integration
import React, { useRef } from 'react';
import ValenceParticleField from '@/ui/visual-effects/ValenceParticleField';
import FloatingSummon from '@/ui/components/FloatingSummon';
import GestureOverlay from '@/integrations/gesture-recognition/GestureOverlay';
import SovereignDashboard from '@/ui/dashboard/SovereignDashboard';
import OfflineMercySanctuary from '@/ui/offline-fallback/OfflineMercySanctuary';

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isOffline = !navigator.onLine;

  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      {/* Global particle field background */}
      <ValenceParticleField />

      {/* Offline sanctuary fallback */}
      {isOffline && <OfflineMercySanctuary />}

      {/* Main content */}
      {!isOffline && (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="fixed inset-0 w-full h-full object-cover z-0"
          />
          <div className="relative z-10">
            <SovereignDashboard />
          </div>
          <GestureOverlay videoRef={videoRef} />
          <FloatingSummon />
        </>
      )}
    </div>
  );
};

export default App;

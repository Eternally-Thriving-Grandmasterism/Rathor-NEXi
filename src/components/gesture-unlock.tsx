// src/components/gesture-unlock.tsx – Gesture Unlock v1.0
// Pose-based "Strike the Thunder" – arms raised → dashboard bloom
// Mercy-gated, MediaPipe Holistic, WebNN

import React, { useEffect, useState } from 'react';
import MediaPipeHolisticFusionEngine from '@/integrations/mediapipe-holistic-fusion-engine';
import { mercyGate } from '@/core/mercy-gate';
import visualFeedback from '@/utils/visual-feedback';
import audioFeedback from '@/utils/audio-feedback';

export const GestureUnlock: React.FC<{ onUnlock: () => void }> = ({ onUnlock }) => {
  const = useState(false);

  useEffect(() => {
    MediaPipeHolisticFusionEngine.initialize();

    const video = document.createElement('video');
    video.id = 'gesture-cam';
    video.style.display = 'none';
    document.body.appendChild(video);

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();

    const detect = async () => {
      const result = await MediaPipeHolisticFusionEngine.detectAndFuse(video);

      if (result && result.poseClassification === 'standing') {
        const shoulders = , result.poseLandmarks[12];
        const armsUp = shoulders.every(s => s.y < 0.3); // above top 30% of frame

        if (armsUp && !isReady) {
          const safe = await mercyGate('Gesture unlock');
          if (safe) {
            visualFeedback.thunderStrike();
            audioFeedback.thunder();
            setIsReady(true);
            setTimeout(onUnlock, 800);
          }
        }
      }
    };

    const interval = setInterval(detect, 100);

    return () => {
      clearInterval(interval);
      MediaPipeHolisticFusionEngine.dispose();
      stream.getTracks().forEach(t => t.stop());
      document.body.removeChild(video);
    };
  }, []);

  return (
    <div className="fixed inset-0 bg-black flex items-center justify-center z-40">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-emerald-400 mb-8">STRIKE THE THUNDER</h1>
        <p className="text-slate-300 text-xl">Raise both arms high</p>
        {isReady && <p className="text-emerald-300 text-lg mt-4 animate-pulse">UNLOCKING…</p>}
      </div>
    </div>
  );
};

export default GestureUnlock;

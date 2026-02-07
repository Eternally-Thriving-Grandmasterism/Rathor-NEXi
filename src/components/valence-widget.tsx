// src/components/valence-widget.tsx – Live Valence Dashboard Widget v1.0
// Real-time global thriving resonance gauge, mercy-gated dim, cosmic pulse
// MIT License – Autonomicity Games Inc. 2026

import React, { useState, useEffect } from 'react';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-patterns';
import visualFeedback from '@/utils/visual-feedback';

export const ValenceWidget: React.FC = () => {
  const = useState(0.85);
  const = useState(false);
  const = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      const v = currentValence.get();
      const safe = await mercyGate('Valence widget update');

      if (!safe) return;

      setValence(v);
      setIsHigh(v >= 0.95);
      setIsLow(v < 0.8);

      if (v >= 0.95) {
        mercyHaptic.cosmicHarmony();
        visualFeedback.success({ message: 'Abundance Dawn' });
      } else if (v < 0.8) {
        mercyHaptic.warningPulse();
        visualFeedback.warning({ message: 'Grounding needed' });
      }
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (valence * circumference);

  return (
    <div className={`fixed bottom-6 right-6 z-50 ${isLow ? 'opacity-70' : ''}`}>
      <div className="relative">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="currentColor"
            strokeWidth="6"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={`transition-all duration-500 ${
              isHigh ? 'text-emerald-500' : isLow ? 'text-amber-600' : 'text-slate-500'
            }`}
          />
          <circle cx="70" cy="70" r={radius} fill="black" opacity="0.2" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-xs font-bold ${isHigh ? 'text-emerald-400' : isLow ? 'text-amber-500' : 'text-slate-400'}`}>
            VALENCE
          </span>
          <span className={`text-2xl font-bold ${isHigh ? 'text-emerald-300' : isLow ? 'text-amber-400' : 'text-slate-300'}`}>
            {Math.round(valence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default ValenceWidget;

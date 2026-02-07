// src/components/SyncQueueWidget.tsx – Sync Queue Visualization Widget v1.2
// Real-time queue viewer + estimated sync time + live countdown + haptic feedback on completion
// MIT License – Autonomicity Games Inc. 2026

import React, { useState, useEffect } from 'react';
import { currentValence } from '@/core/valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

interface PendingMutation {
  id?: number;
  type: string;
  url: string;
  method: string;
  payload: any;
  valence: number;
  timestamp: number;
  retryCount: number;
  nextAttempt: number;
  status: 'pending' | 'retrying' | 'synced' | 'dropped' | 'conflict';
}

const SyncQueueWidget: React.FC = () => {
  const [queue, setQueue] = useState<PendingMutation[]>([]);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [expanded, setExpanded] = useState(false);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const valence = currentValence.get();

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Listen for sync completion from SW
    navigator.serviceWorker.addEventListener('message', event => {
      if (event.data.type === 'SYNC_COMPLETED') {
        const { synced, conflicts, dropped, total, valence } = event.data.payload;
        mercyHaptic.playPattern(
          synced > 0 ? 'cosmicHarmony' : 'warningPulse',
          valence * (synced > 0 ? 1.0 : 0.7)
        );
        loadQueue(); // refresh UI
        console.log(`[SyncWidget] Sync completed: ${synced} synced, ${conflicts} conflicts, ${dropped} dropped`);
      }
    });

    const interval = setInterval(() => {
      loadQueue();
      updateETA();
    }, 3000);

    loadQueue();
    updateETA();

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(interval);
    };
  }, []);

  // ... rest of the component remains unchanged from previous version ...

  const updateETA = () => {
    if (queue.length === 0) {
      setEtaSeconds(null);
      return;
    }

    const basePerItemMs = isOnline ? 800 : 5000;
    const valenceFactor = 1 + (valence - 0.5) * 0.5;
    const retryFactor = queue.reduce((sum, item) => sum + (item.retryCount || 0), 0) / (queue.length || 1) + 1;

    const totalMs = queue.length * basePerItemMs * retryFactor / valenceFactor;
    setEtaSeconds(Math.round(totalMs / 1000));
  };

  // ... rest of render logic unchanged ...
};

export default SyncQueueWidget;

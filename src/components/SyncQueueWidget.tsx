// src/components/SyncQueueWidget.tsx – Sync Queue Visualization Widget v1.0
// Real-time display of pending mutations, valence priority, retry status, conflict resolution
// Mercy-gated warnings, offline/online awareness, eternal thriving transparency
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
  const valence = currentValence.get();

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    const interval = setInterval(loadQueue, 5000); // refresh every 5s
    loadQueue(); // initial load

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(interval);
    };
  }, []);

  const loadQueue = async () => {
    if (!('indexedDB' in window)) return;

    try {
      const db = await openDB();
      const tx = db.transaction('pendingMutations', 'readonly');
      const store = tx.objectStore('pendingMutations');
      const items = await store.getAll();

      // Sort by valence desc + timestamp asc (high valence first)
      items.sort((a, b) => {
        if (b.valence !== a.valence) return b.valence - a.valence;
        return a.timestamp - b.timestamp;
      });

      setQueue(items);
    } catch (err) {
      console.warn('[SyncQueueWidget] Failed to load queue:', err);
    }
  };

  const openDB = () => {
    return new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('rathor-nexi-db', 3);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getRetryCountdown = (nextAttempt: number) => {
    const seconds = Math.max(0, Math.ceil((nextAttempt - Date.now()) / 1000));
    return seconds > 0 ? `${seconds}s` : 'now';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'synced': return '#00ff88';
      case 'retrying': return '#ff8800';
      case 'dropped': case 'conflict': return '#ff4444';
      default: return '#aaaaaa';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      background: 'rgba(0,0,0,0.7)',
      border: '1px solid rgba(0,255,136,0.3)',
      borderRadius: '12px',
      padding: '12px 16px',
      color: '#00ff88',
      fontFamily: 'Courier New, monospace',
      fontSize: '0.9rem',
      maxWidth: '360px',
      zIndex: 1000,
      boxShadow: '0 0 20px rgba(0,255,136,0.2)',
      transition: 'all 0.3s',
      transform: expanded ? 'scale(1.05)' : 'scale(1)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: expanded ? '12px' : '0' }}>
        <div>
          Sync Queue {isOnline ? '🟢 Online' : '🔴 Offline'}
          <span style={{ marginLeft: '8px', opacity: 0.7 }}>
            ({queue.length} pending)
          </span>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            background: 'none',
            border: 'none',
            color: '#00ff88',
            fontSize: '1.2rem',
            cursor: 'pointer'
          }}
        >
          {expanded ? '−' : '+'}
        </button>
      </div>

      {expanded && (
        <div style={{ maxHeight: '300px', overflowY: 'auto', paddingRight: '8px' }}>
          {queue.length === 0 ? (
            <div style={{ opacity: 0.6, textAlign: 'center', padding: '12px 0' }}>
              Queue empty – lattice thriving smoothly ✨
            </div>
          ) : (
            queue.map(item => (
              <div
                key={item.id}
                style={{
                  padding: '8px',
                  marginBottom: '8px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '8px',
                  borderLeft: `4px solid ${getStatusColor(item.status)}`
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontWeight: 'bold' }}>{item.type}</span>
                  <span style={{ opacity: 0.7, fontSize: '0.8rem' }}>
                    {formatTime(item.timestamp)}
                  </span>
                </div>
                <div style={{ fontSize: '0.85rem', opacity: 0.85 }}>
                  Valence: {item.valence.toFixed(2)}
                  {item.retryCount > 0 && ` • Retry ${item.retryCount}`}
                  {item.nextAttempt > Date.now() && ` • Next: ${getRetryCountdown(item.nextAttempt)}`}
                </div>
                {item.status === 'conflict' && (
                  <div style={{ color: '#ff8800', fontSize: '0.8rem', marginTop: '4px' }}>
                    Conflict resolved (last-write-wins)
                  </div>
                )}
                {item.status === 'dropped' && (
                  <div style={{ color: '#ff4444', fontSize: '0.8rem', marginTop: '4px' }}>
                    Dropped – mercy queue cap
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default SyncQueueWidget;

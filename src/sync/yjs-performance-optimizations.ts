// src/sync/yjs-performance-optimizations.ts – Yjs Performance Optimizations v1
// Multiplanetary-ready, high-latency, large-state tuning, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb';
import { currentValence } from '@/core/valence-tracker';

export function createOptimizedYDoc(name: string): Y.Doc {
  const ydoc = new Y.Doc({
    gc: true,
    gcFilter: () => true,               // aggressive tombstone cleanup
  });

  // Offline persistence
  const persistence = new IndexeddbPersistence(name, ydoc);

  // Relay (only connect when online & high valence)
  let provider: WebsocketProvider | null = null;
  const connectRelay = () => {
    if (provider || !navigator.onLine || currentValence.get() < 0.9) return;
    provider = new WebsocketProvider(
      import.meta.env.VITE_SYNC_RELAY_URL || 'wss://relay.rathor.ai',
      name,
      ydoc,
      { connectTimeout: 3000, maxBackoffTime: 15000 }
    );
  };

  window.addEventListener('online', connectRelay);
  if (navigator.onLine && currentValence.get() >= 0.9) connectRelay();

  // Awareness throttling (presence/cursor)
  ydoc.awareness.setLocalStateField('user', { name: 'MercyUser' });
  const awarenessInterval = setInterval(() => {
    if (currentValence.get() > 0.95) {
      ydoc.awareness.setLocalStateField('cursor', { x: 0, y: 0 }); // example
    }
  }, 800); // throttle to 1.25 Hz

  // Cleanup on unload
  window.addEventListener('unload', () => {
    clearInterval(awarenessInterval);
    persistence.destroy();
    provider?.destroy();
  });

  return ydoc;
}

// Batch transaction helper (reduces update size)
export function transactBatch<T>(ydoc: Y.Doc, fn: (t: Y.Transaction) => T): T {
  return ydoc.transact(fn);
}

// High-latency queue + compression helper
export async function queueUpdate(ydoc: Y.Doc, update: Uint8Array) {
  // In real impl: persist to IndexedDB queue
  // On reconnect: ydoc.store.update(update)
  console.log(`[YjsPerf] Queued update size: ${update.byteLength} bytes`);
}

// Usage example
// const ydoc = createOptimizedYDoc('mercy-lattice-global');
// transactBatch(ydoc, t => { t.getMap('state').set('valence', currentValence.get()); });

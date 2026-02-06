// src/sync/multiplanetary-sync-engine.ts – Multiplanetary Sync Engine v2
// Optimized ElectricSQL sync: valence prioritization, delta compression, reconnection bloom
// MIT License – Autonomicity Games Inc. 2026

import { electric } from '@electric-sql/pglite';
import * as zstd from '@boku7/zstd'; // or use pako/brotli for browser
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_SYNC_PIVOT = 0.9;
const MAX_OFFLINE_QUEUE_SIZE = 1000;
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];

let db: any = null;
let syncQueue: Array<{ table: string; row: any; valence: number }> = [];
let reconnectAttempts = 0;

export class MultiplanetarySyncEngine {
  static async initialize(databaseUrl: string) {
    const actionName = 'Initialize optimized ElectricSQL sync engine';
    if (!await mercyGate(actionName)) return;

    try {
      db = await electric.connect(databaseUrl, {
        auth: { token: 'your-auth-token-here' }, // replace with real auth
        schema: {
          users: { primaryKey: 'id' },
          progress: { primaryKey: 'id' },
          probes: { primaryKey: 'id' },
          habitats: { primaryKey: 'id' },
          valence_logs: { primaryKey: 'id' }
        }
      });

      // Subscribe to high-priority shapes first (valence-aware)
      await db.sync({
        shape: {
          table: 'valence_logs',
          where: 'valence > $1',
          params: [VALENCE_SYNC_PIVOT]
        }
      });

      await db.sync({ shape: { table: 'users' } });
      await db.sync({ shape: { table: 'progress' } });
      await db.sync({ shape: { table: 'probes' } });
      await db.sync({ shape: { table: 'habitats' } });

      console.log("[SyncEngine] ElectricSQL initialized – high-valence shapes prioritized");

      // Reconnection bloom
      db.on('disconnected', () => {
        this.startReconnectBloom();
      });

      db.on('connected', () => {
        reconnectAttempts = 0;
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        this.flushOfflineQueue();
      });
    } catch (e) {
      console.error("[SyncEngine] Initialization failed", e);
    }
  }

  static async syncWithValencePriority(data: { table: string; row: any }) {
    const actionName = 'Valence-priority sync';
    if (!await mercyGate(actionName)) return;

    const valence = currentValence.get();
    const entry = { ...data, valence };

    if (valence > VALENCE_SYNC_PIVOT) {
      // High valence → sync immediately
      await this.syncImmediate(entry);
    } else {
      // Queue for batch sync
      syncQueue.push(entry);
      if (syncQueue.length > MAX_OFFLINE_QUEUE_SIZE) {
        syncQueue.shift(); // prune oldest low-valence
      }
    }
  }

  private static async syncImmediate(entry: any) {
    try {
      const compressed = await this.compressDelta(entry.row);
      await db.electric.sync.push({
        table: entry.table,
        data: compressed,
        valence: entry.valence
      });
      mercyHaptic.playPattern('cosmicHarmony', entry.valence);
    } catch (e) {
      syncQueue.push(entry);
    }
  }

  private static async flushOfflineQueue() {
    if (syncQueue.length === 0) return;

    // Sort by valence descending (high first)
    syncQueue.sort((a, b) => b.valence - a.valence);

    const batch = syncQueue.splice(0, 50); // batch 50 at a time
    for (const entry of batch) {
      await this.syncImmediate(entry);
    }

    if (syncQueue.length > 0) {
      setTimeout(() => this.flushOfflineQueue(), 5000);
    }
  }

  private static async compressDelta(data: any): Promise<Uint8Array> {
    const json = JSON.stringify(data);
    const binary = new TextEncoder().encode(json);
    return await zstd.compress(binary, 3); // zstd level 3
  }

  private static startReconnectBloom() {
    const delay = RECONNECT_BACKOFF_MS[Math.min(reconnectAttempts, RECONNECT_BACKOFF_MS.length - 1)];
    reconnectAttempts++;
    setTimeout(() => {
      db.reconnect();
    }, delay);
  }

  static getSyncStatus() {
    return {
      isConnected: db?.isConnected || false,
      queueLength: syncQueue.length,
      reconnectAttempts,
      lastValenceSync: currentValence.get()
    };
  }
}

export default MultiplanetarySyncEngine;

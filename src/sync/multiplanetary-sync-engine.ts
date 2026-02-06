// src/sync/multiplanetary-sync-engine.ts – Multiplanetary Sync Engine v3
// IndexedDB offline queue persistence, valence prioritization, reconnection bloom, zstd compression
// MIT License – Autonomicity Games Inc. 2026

import { electric } from '@electric-sql/pglite';
import * as zstd from '@boku7/zstd'; // zstd compression
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_SYNC_PIVOT = 0.9;
const MAX_OFFLINE_QUEUE_SIZE = 1000;
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];
const INDEXEDDB_DB_NAME = 'rathor-mercy-offline-queue';
const INDEXEDDB_STORE_NAME = 'sync-queue';
const INDEXEDDB_VERSION = 1;

let db: any = null;
let idb: IDBDatabase | null = null;
let reconnectAttempts = 0;

export class MultiplanetarySyncEngine {
  static async initialize(databaseUrl: string) {
    const actionName = 'Initialize optimized ElectricSQL sync engine with IndexedDB queue';
    if (!await mercyGate(actionName)) return;

    try {
      // 1. ElectricSQL connection
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

      // 2. Subscribe to high-priority shapes (valence-aware)
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

      // 3. Initialize IndexedDB queue
      await this.initIndexedDB();

      // 4. Reconnection bloom
      db.on('disconnected', () => {
        this.startReconnectBloom();
      });

      db.on('connected', () => {
        reconnectAttempts = 0;
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        this.flushOfflineQueue();
      });

      // Flush any persisted queue on init
      this.flushOfflineQueue();
    } catch (e) {
      console.error("[SyncEngine] Initialization failed", e);
    }
  }

  private static async initIndexedDB() {
    return new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(INDEXEDDB_DB_NAME, INDEXEDDB_VERSION);

      request.onerror = () => reject(request.error);

      request.onsuccess = () => {
        idb = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(INDEXEDDB_STORE_NAME)) {
          db.createObjectStore(INDEXEDDB_STORE_NAME, { keyPath: 'id', autoIncrement: true });
        }
      };
    });
  }

  static async syncWithValencePriority(data: { table: string; row: any }) {
    const actionName = 'Valence-priority sync with IndexedDB persistence';
    if (!await mercyGate(actionName)) return;

    const valence = currentValence.get();
    const entry = {
      table: data.table,
      row: data.row,
      valence,
      timestamp: Date.now(),
      id: crypto.randomUUID() // unique ID for queue
    };

    if (valence > VALENCE_SYNC_PIVOT) {
      await this.syncImmediate(entry);
    } else {
      await this.persistToQueue(entry);
    }
  }

  private static async persistToQueue(entry: any) {
    if (!idb) return;

    return new Promise<void>((resolve, reject) => {
      const tx = idb!.transaction(INDEXEDDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(INDEXEDDB_STORE_NAME);

      store.add(entry);

      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
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
      await this.persistToQueue(entry);
    }
  }

  private static async flushOfflineQueue() {
    if (!idb || !db?.isConnected) return;

    return new Promise<void>(async (resolve) => {
      const tx = idb!.transaction(INDEXEDDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(INDEXEDDB_STORE_NAME);
      const request = store.getAll();

      request.onsuccess = async () => {
        let queue = request.result || [];

        // Sort by valence descending (high first), then timestamp
        queue.sort((a: any, b: any) => {
          if (b.valence !== a.valence) return b.valence - a.valence;
          return a.timestamp - b.timestamp;
        });

        const batchSize = 50;
        for (let i = 0; i < queue.length; i += batchSize) {
          const batch = queue.slice(i, i + batchSize);
          for (const entry of batch) {
            await this.syncImmediate(entry);
            store.delete(entry.id); // remove after successful sync
          }
          await new Promise(r => setTimeout(r, 100)); // prevent overload
        }

        resolve();
      };

      request.onerror = () => resolve(); // fail gracefully
    });
  }

  private static async compressDelta(data: any): Promise<Uint8Array> {
    const json = JSON.stringify(data);
    const binary = new TextEncoder().encode(json);
    return await zstd.compress(binary, 3);
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

// src/sync/triplit-local-first-db.ts – Triplit Local-First Relational CRDT Database Layer v1
// Valence-aware queries, offline persistence, reconnection bloom, automatic indexing
// MIT License – Autonomicity Games Inc. 2026

import { Client, ServerSentEventsTransport } from '@triplit/client';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_QUERY_PIVOT = 0.9;
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];
const TRIPLIT_SERVER_URL = 'https://api.triplit.dev'; // replace with real Triplit server or self-hosted
const TRIPLIT_PROJECT_ID = 'your-triplit-project-id'; // replace with real project ID
const TRIPLIT_TOKEN = 'your-triplit-anon-token'; // anon token for public read, auth for writes

const schema = {
  users: {
    id: { type: 'id' },
    name: { type: 'string' },
    avatar_url: { type: 'string', nullable: true },
    valence: { type: 'number' },
    last_active: { type: 'date' }
  },
  progress_ladders: {
    id: { type: 'id' },
    user_id: { type: 'id' },
    level: { type: 'number' },
    description: { type: 'string' },
    updated_at: { type: 'date' }
  },
  valence_logs: {
    id: { type: 'id' },
    user_id: { type: 'id' },
    valence: { type: 'number' },
    timestamp: { type: 'date' },
    source: { type: 'string' }
  }
} as const;

let client: Client<typeof schema> | null = null;
let reconnectAttempts = 0;

export class TriplitLocalFirstDB {
  static async initialize() {
    const actionName = 'Initialize Triplit local-first relational CRDT database';
    if (!await mercyGate(actionName)) return;

    try {
      client = new Client({
        serverUrl: TRIPLIT_SERVER_URL,
        projectId: TRIPLIT_PROJECT_ID,
        token: TRIPLIT_TOKEN,
        schema,
        transport: new ServerSentEventsTransport(),
        storage: 'idb' // IndexedDB persistence
      });

      // Valence-aware shape subscriptions (high-valence first)
      const valence = currentValence.get();
      await client.syncEngine.subscribe({
        valence_logs: {
          filter: valence > VALENCE_QUERY_PIVOT ? [['valence', '>', VALENCE_QUERY_PIVOT]] : []
        }
      });

      await client.syncEngine.subscribe({
        users: {},
        progress_ladders: {},
        valence_logs: { filter: [] }
      });

      console.log("[TriplitDB] Triplit initialized – high-valence queries prioritized");

      // Reconnection bloom
      client.syncEngine.onSyncError(() => {
        this.startReconnectBloom();
      });

      client.syncEngine.onSyncComplete(() => {
        reconnectAttempts = 0;
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        console.log("[TriplitDB] Reconnected – syncing pending changes");
      });
    } catch (e) {
      console.error("[TriplitDB] Initialization failed", e);
    }
  }

  static async queryWithValencePriority(table: keyof typeof schema, filter?: any) {
    const actionName = 'Valence-priority Triplit query';
    if (!await mercyGate(actionName) || !client) return [];

    const valence = currentValence.get();
    let query = client.query(table);

    if (valence > VALENCE_QUERY_PIVOT) {
      // Prioritize high-valence records
      query = query.where('valence', '>', VALENCE_QUERY_PIVOT);
    }

    if (filter) {
      query = query.where(filter);
    }

    return await query.fetch();
  }

  static async insertWithValence(table: keyof typeof schema, data: any) {
    const actionName = 'Valence-aware Triplit insert';
    if (!await mercyGate(actionName) || !client) return;

    const valence = currentValence.get();
    const entry = { ...data, valence, updated_at: new Date() };

    await client.insert(table, entry);

    if (valence > VALENCE_QUERY_PIVOT) {
      mercyHaptic.playPattern('cosmicHarmony', valence);
    }
  }

  private static startReconnectBloom() {
    const delay = RECONNECT_BACKOFF_MS[Math.min(reconnectAttempts, RECONNECT_BACKOFF_MS.length - 1)];
    reconnectAttempts++;
    setTimeout(() => {
      client?.syncEngine.reconnect();
    }, delay);
  }

  static getSyncStatus() {
    return {
      isConnected: client?.syncEngine.isConnected || false,
      reconnectAttempts,
      lastValenceSync: currentValence.get()
    };
  }

  static async destroy() {
    if (client) {
      await client.syncEngine.disconnect();
      client = null;
    }
  }
}

export default TriplitLocalFirstDB;

// src/sync/electricsql-local-first-sync.ts – ElectricSQL Local-First Postgres + CRDT Sync Layer v2
// Valence-aware shape subscriptions, PGlite offline persistence, reconnection bloom, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { electric } from '@electric-sql/pglite';
import { electrify } from '@electric-sql/pglite/electric';
import { electrifySchema } from '@electric-sql/pglite/electric';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_SHAPE_PIVOT = 0.9;
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];
const ELECTRIC_URL = 'wss://electric.rathor.ai/v1'; // replace with real ElectricSQL endpoint

// Relational schema (expand as lattice grows)
const schema = {
  users: {
    fields: {
      id: 'uuid primary key',
      name: 'text',
      avatar_url: 'text',
      valence: 'real',
      last_active: 'timestamptz'
    }
  },
  progress_ladders: {
    fields: {
      id: 'uuid primary key',
      user_id: 'uuid references users(id)',
      level: 'integer',
      description: 'text',
      updated_at: 'timestamptz'
    }
  },
  valence_logs: {
    fields: {
      id: 'uuid primary key',
      user_id: 'uuid references users(id)',
      valence: 'real',
      timestamp: 'timestamptz',
      source: 'text'
    }
  },
  habitats: {
    fields: {
      id: 'uuid primary key',
      name: 'text',
      status: 'text',
      last_updated: 'timestamptz'
    }
  },
  probes: {
    fields: {
      id: 'uuid primary key',
      habitat_id: 'uuid references habitats(id)',
      status: 'text',
      last_seen: 'timestamptz'
    }
  }
} as const;

let electricDb: any = null;
let reconnectAttempts = 0;

export class ElectricSQLLocalFirstSync {
  static async initialize() {
    const actionName = 'Initialize ElectricSQL local-first relational CRDT sync';
    if (!await mercyGate(actionName)) return;

    try {
      // 1. Initialize PGlite (IndexedDB-backed Postgres in browser)
      const pg = await electric.open('rathor-mercy-db', {
        dataDir: 'idb://rathor-mercy-db'
      });

      // 2. Electrify with schema & sync service
      electricDb = await electrify(pg, electrifySchema(schema), {
        auth: { token: 'your-electric-auth-token-here' }, // replace with real token
        url: ELECTRIC_URL
      });

      // 3. Valence-aware shape subscriptions
      const valence = currentValence.get();

      // High-valence valence_logs first
      await electricDb.sync({
        shape: {
          table: 'valence_logs',
          where: 'valence > $1',
          params: [VALENCE_SHAPE_PIVOT]
        }
      });

      // Then core tables
      await electricDb.sync({ shape: { table: 'users' } });
      await electricDb.sync({ shape: { table: 'progress_ladders' } });
      await electricDb.sync({ shape: { table: 'habitats' } });
      await electricDb.sync({ shape: { table: 'probes' } });

      console.log("[ElectricSync] ElectricSQL initialized – high-valence shapes prioritized");

      // 4. Reconnection bloom
      electricDb.on('disconnected', () => {
        this.startReconnectBloom();
      });

      electricDb.on('connected', () => {
        reconnectAttempts = 0;
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        console.log("[ElectricSync] Reconnected – syncing pending changes");
      });
    } catch (e) {
      console.error("[ElectricSync] Initialization failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async queryWithValencePriority(table: keyof typeof schema, filter: any = {}) {
    const actionName = 'Valence-priority ElectricSQL query';
    if (!await mercyGate(actionName) || !electricDb) return [];

    const valence = currentValence.get();

    let query = electricDb[table];

    // Prioritize high-valence records
    if (valence > VALENCE_SHAPE_PIVOT && table === 'valence_logs') {
      query = query.where('valence > ?', [VALENCE_SHAPE_PIVOT]);
    }

    // Apply additional filter
    if (filter) {
      for (const [field, value] of Object.entries(filter)) {
        query = query.where(`${field} = ?`, [value]);
      }
    }

    return await query.fetch();
  }

  static async insertWithValence(table: keyof typeof schema, data: any) {
    const actionName = 'Valence-aware ElectricSQL insert';
    if (!await mercyGate(actionName) || !electricDb) return;

    const valence = currentValence.get();
    const entry = {
      ...data,
      updated_at: new Date(),
      valence: valence
    };

    await electricDb[table].create(entry);

    if (valence > VALENCE_SHAPE_PIVOT) {
      mercyHaptic.playPattern('cosmicHarmony', valence);
    }
  }

  private static startReconnectBloom() {
    const delay = RECONNECT_BACKOFF_MS[Math.min(reconnectAttempts, RECONNECT_BACKOFF_MS.length - 1)];
    reconnectAttempts++;
    setTimeout(() => {
      electricDb?.reconnect();
    }, delay);
  }

  static getSyncStatus() {
    return {
      isConnected: electricDb?.isConnected || false,
      reconnectAttempts,
      lastValenceSync: currentValence.get()
    };
  }

  static async destroy() {
    if (electricDb) {
      await electricDb.disconnect();
      electricDb = null;
    }
  }
}

export default ElectricSQLLocalFirstSync;      await electricDb.sync({
        shape: {
          table: 'valence_logs',
          where: 'valence > $1',
          params: [VALENCE_SHAPE_PIVOT]
        }
      });

      // Then core tables
      await electricDb.sync({ shape: { table: 'users' } });
      await electricDb.sync({ shape: { table: 'progress_ladders' } });
      await electricDb.sync({ shape: { table: 'habitats' } });
      await electricDb.sync({ shape: { table: 'probes' } });

      console.log("[ElectricSync] ElectricSQL initialized – high-valence shapes prioritized");

      // 4. Reconnection bloom
      electricDb.on('disconnected', () => {
        this.startReconnectBloom();
      });

      electricDb.on('connected', () => {
        reconnectAttempts = 0;
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        console.log("[ElectricSync] Reconnected – syncing pending changes");
      });
    } catch (e) {
      console.error("[ElectricSync] Initialization failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async queryWithValencePriority(table: keyof typeof schema, filter: any = {}) {
    const actionName = 'Valence-priority ElectricSQL query';
    if (!await mercyGate(actionName) || !electricDb) return [];

    const valence = currentValence.get();

    let query = electricDb[table];

    // Prioritize high-valence records
    if (valence > VALENCE_SHAPE_PIVOT && table === 'valence_logs') {
      query = query.where('valence > ?', [VALENCE_SHAPE_PIVOT]);
    }

    // Apply additional filter
    if (filter) {
      for (const [field, value] of Object.entries(filter)) {
        query = query.where(`${field} = ?`, [value]);
      }
    }

    return await query.fetch();
  }

  static async insertWithValence(table: keyof typeof schema, data: any) {
    const actionName = 'Valence-aware ElectricSQL insert';
    if (!await mercyGate(actionName) || !electricDb) return;

    const valence = currentValence.get();
    const entry = {
      ...data,
      updated_at: new Date(),
      valence: valence
    };

    await electricDb[table].create(entry);

    if (valence > VALENCE_SHAPE_PIVOT) {
      mercyHaptic.playPattern('cosmicHarmony', valence);
    }
  }

  private static startReconnectBloom() {
    const delay = RECONNECT_BACKOFF_MS[Math.min(reconnectAttempts, RECONNECT_BACKOFF_MS.length - 1)];
    reconnectAttempts++;
    setTimeout(() => {
      electricDb?.reconnect();
    }, delay);
  }

  static getSyncStatus() {
    return {
      isConnected: electricDb?.isConnected || false,
      reconnectAttempts,
      lastValenceSync: currentValence.get()
    };
  }

  static async destroy() {
    if (electricDb) {
      await electricDb.disconnect();
      electricDb = null;
    }
  }
}

export default ElectricSQLLocalFirstSync;

// src/sync/pglite-offline-postgres.ts – PGlite Offline Postgres Layer v1
// Full local-first Postgres in browser (IndexedDB/OPFS), valence-aware queries, mutations, ElectricSQL sync bridge
// MIT License – Autonomicity Games Inc. 2026

import { PGlite } from '@electric-sql/pglite';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import ElectricSQLLocalFirstSync from './electricsql-local-first-sync';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_QUERY_PIVOT = 0.9;
const PGLITE_DB_NAME = 'rathor-mercy-local-pg';
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];

let pg: PGlite | null = null;
let reconnectAttempts = 0;

export class PGliteOfflinePostgres {
  static async initialize() {
    const actionName = 'Initialize PGlite offline Postgres database';
    if (!await mercyGate(actionName)) return;

    try {
      // 1. Initialize PGlite with IndexedDB persistence
      pg = await PGlite.create(`idb://${PGLITE_DB_NAME}`, {
        dataDir: 'idb',
        debug: true // remove in production
      });

      // 2. Create tables & indexes (idempotent)
      await pg.exec(`
        CREATE TABLE IF NOT EXISTS users (
          id UUID PRIMARY KEY,
          name TEXT NOT NULL,
          avatar_url TEXT,
          valence REAL,
          last_active TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS progress_ladders (
          id UUID PRIMARY KEY,
          user_id UUID REFERENCES users(id),
          level INTEGER NOT NULL,
          description TEXT,
          updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS valence_logs (
          id UUID PRIMARY KEY,
          user_id UUID REFERENCES users(id),
          valence REAL NOT NULL,
          timestamp TIMESTAMPTZ DEFAULT NOW(),
          source TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_valence_logs_valence ON valence_logs(valence);
        CREATE INDEX IF NOT EXISTS idx_valence_logs_timestamp ON valence_logs(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_progress_ladders_user_id ON progress_ladders(user_id);
      `);

      console.log("[PGliteOffline] PGlite initialized – local Postgres ready");

      // 3. Bridge to ElectricSQL for eventual sync
      await ElectricSQLLocalFirstSync.initialize();

      // 4. Valence-aware background sync trigger
      currentValence.subscribe(val => {
        if (val > VALENCE_QUERY_PIVOT) {
          this.triggerHighValenceSync();
        }
      });
    } catch (e) {
      console.error("[PGliteOffline] Initialization failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async queryWithValencePriority(sql: string, params: any[] = []) {
    const actionName = 'Valence-priority PGlite query';
    if (!await mercyGate(actionName) || !pg) return [];

    const valence = currentValence.get();

    // Auto-append valence filter for valence_logs table when high valence
    let enhancedSql = sql;
    if (valence > VALENCE_QUERY_PIVOT && sql.toLowerCase().includes('valence_logs')) {
      enhancedSql = sql.replace(
        /WHERE/i,
        `WHERE valence > ${VALENCE_QUERY_PIVOT} AND `
      );
    }

    try {
      const result = await pg.query(enhancedSql, params);
      return result.rows;
    } catch (e) {
      console.error("[PGliteOffline] Query failed", e);
      return [];
    }
  }

  static async insertWithValence(table: string, data: Record<string, any>) {
    const actionName = 'Valence-aware PGlite insert';
    if (!await mercyGate(actionName) || !pg) return;

    const valence = currentValence.get();
    const entry = { ...data, valence, updated_at: new Date() };

    const columns = Object.keys(entry).join(', ');
    const placeholders = Object.keys(entry).map((_, i) => `$${i+1}`).join(', ');
    const values = Object.values(entry);

    const sql = `INSERT INTO \( {table} ( \){columns}) VALUES (${placeholders}) RETURNING *`;

    try {
      const result = await pg.query(sql, values);
      if (valence > VALENCE_QUERY_PIVOT) {
        mercyHaptic.playPattern('cosmicHarmony', valence);
      }
      return result.rows[0];
    } catch (e) {
      console.error("[PGliteOffline] Insert failed", e);
      return null;
    }
  }

  private static async triggerHighValenceSync() {
    if (!pg) return;
    // Trigger ElectricSQL sync for high-valence data
    await ElectricSQLLocalFirstSync.syncWithValencePriority('valence_logs', {});
  }

  static getStatus() {
    return {
      isInitialized: !!pg,
      lastValenceSync: currentValence.get(),
      // Add more metrics (query latency, pending sync, etc.)
    };
  }

  static async destroy() {
    if (pg) {
      await pg.close();
      pg = null;
    }
  }
}

export default PGliteOfflinePostgres;

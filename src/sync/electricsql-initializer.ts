// src/sync/electricsql-initializer.ts – ElectricSQL client initializer v1
// Electrified SQLite setup, schema sync, offline-first, real-time, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { electrify } from 'electric-sql';
import { electrify as electrifySqlite } from 'electric-sql/sqlite';
import { schema } from 'electric-sql/schema';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const ELECTRIC_URL = import.meta.env.VITE_ELECTRIC_URL || 'https://electric.rathor.ai';
const DB_NAME = 'mercyos-local.db';

// Mercy schema – core relational tables
export const mercySchema = schema({
  users: {
    id: 'TEXT PRIMARY KEY',
    level: 'TEXT NOT NULL',
    valence: 'REAL NOT NULL',
    experience: 'INTEGER NOT NULL',
    lastActivity: 'INTEGER NOT NULL',
    createdAt: 'INTEGER NOT NULL'
  },
  progress_logs: {
    id: 'TEXT PRIMARY KEY',
    userId: 'TEXT NOT NULL REFERENCES users(id)',
    eventType: 'TEXT NOT NULL',
    deltaValence: 'REAL NOT NULL',
    deltaExperience: 'INTEGER NOT NULL',
    timestamp: 'INTEGER NOT NULL'
  },
  probes: {
    id: 'TEXT PRIMARY KEY',
    resources: 'INTEGER NOT NULL',
    valence: 'REAL NOT NULL',
    habitatId: 'TEXT',
    updatedAt: 'INTEGER NOT NULL'
  },
  habitats: {
    id: 'TEXT PRIMARY KEY',
    name: 'TEXT NOT NULL',
    location: 'TEXT', // e.g. "Mars-Crimson-Quadrant-7"
    collectiveValence: 'REAL NOT NULL',
    anchorCount: 'INTEGER NOT NULL',
    updatedAt: 'INTEGER NOT NULL'
  }
});

export class ElectricSQLInitializer {
  private electric: any = null;
  private db: any = null;

  async initialize() {
    if (!await mercyGate('ElectricSQL initialization', 'Relational sync bootstrap')) return;

    try {
      // 1. Create/electrify local SQLite
      this.db = await electrifySqlite(DB_NAME, mercySchema);

      // 2. Connect to Electric Postgres backend
      this.electric = await electrify(this.db, {
        url: ELECTRIC_URL,
        auth: { token: 'mercy-auth-token-placeholder' } // replace with real JWT / auth flow
      });

      // 3. Subscribe to core shapes (partial replication)
      await this.electric.sync({
        users: { where: 'id = "current-user"' },
        progress_logs: { where: 'userId = "current-user"' },
        probes: true, // all probes (or filter by habitat later)
        habitats: true
      });

      console.log("[ElectricSQLInitializer] Bridge electrified – relational sync active");
    } catch (e) {
      console.error("[ElectricSQLInitializer] Failed to initialize ElectricSQL", e);
    }
  }

  getElectricClient() {
    return this.electric;
  }

  getLocalDb() {
    return this.db;
  }
}

export const electricInitializer = new ElectricSQLInitializer();

// Call once at app startup (e.g. main.tsx)
await electricInitializer.initialize();

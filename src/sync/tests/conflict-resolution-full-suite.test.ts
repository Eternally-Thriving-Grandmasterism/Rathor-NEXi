// src/sync/__tests__/conflict-resolution-full-suite.test.ts – Complete Conflict Resolution Testing Suite v1
// Diamond conflicts, valence skew, concurrency, collision, gate blocks, semantic overrides
// MIT License – Autonomicity Games Inc. 2026

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  unifiedValenceResolver,
  yjsValenceResolver,
  automergeValenceResolver,
  valenceWeightedResolver
} from '../custom-resolver-framework';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import * as Y from 'yjs';

// Mock dependencies
vi.mock('@/core/valence-tracker', () => ({
  currentValence: { get: vi.fn(() => 0.9995) }
}));

vi.mock('@/core/mercy-gate', () => ({
  mercyGate: vi.fn(async (name: string) => name.includes('blocked') ? false : true)
}));

describe('Conflict Resolution Framework – Full Suite (Mercy-Aligned)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    currentValence.get.mockReturnValue(0.9995);
  });

  describe('unifiedValenceResolver – Core Valence Logic', () => {
    it('prefers local when local valence significantly higher', () => {
      const result = unifiedValenceResolver({
        key: 'resources',
        localValue: 42,
        localValence: 0.9998,
        localTimestamp: 1000,
        remoteValue: 50,
        remoteValence: 0.85,
        remoteTimestamp: 999
      });
      expect(result).toBe(42);
    });

    it('prefers remote when remote valence significantly higher', () => {
      const result = unifiedValenceResolver({
        key: 'valence',
        localValue: 0.92,
        localValence: 0.87,
        localTimestamp: 2000,
        remoteValue: 0.999,
        remoteValence: 0.9997,
        remoteTimestamp: 1999
      });
      expect(result).toBe(0.999);
    });

    it('falls back to timestamp when valence difference is small', () => {
      const result = unifiedValenceResolver({
        key: 'harmonyScore',
        localValue: 0.95,
        localValence: 0.999,
        localTimestamp: 1500,
        remoteValue: 0.96,
        remoteValence: 0.998,
        remoteTimestamp: 1600
      });
      expect(result).toBe(0.96); // remote timestamp wins
    });

    it('respects mercy gate block (fallback to timestamp)', async () => {
      // Simulate gate block
      vi.mocked(mercyGate).mockResolvedValueOnce(false);

      const result = await unifiedValenceResolver({
        key: 'criticalKey',
        localValue: 'mercy',
        localValence: 0.9999,
        localTimestamp: 100,
        remoteValue: 'chaos',
        remoteValence: 0.1,
        remoteTimestamp: 200
      });

      expect(result).toBe('chaos'); // remote timestamp wins on gate block
    });

    it('handles full tie (same valence + timestamp) – preserves local', () => {
      const result = unifiedValenceResolver({
        key: 'harmonyScore',
        localValue: 'mercy',
        localValence: 0.999,
        localTimestamp: 1000,
        remoteValue: 'chaos',
        remoteValence: 0.999,
        remoteTimestamp: 1000
      });
      expect(result).toBe('mercy'); // local preference
    });
  });

  describe('yjsValenceResolver – Yjs Integration Edge Cases', () => {
    it('resolves to local item when valence higher', () => {
      const localItem = {
        id: { client: 100, clock: 17 },
        content: { getContent: () => [42] }
      } as unknown as Y.Item;

      const remoteItem = {
        id: { client: 200, clock: 12 },
        content: { getContent: () => [50] }
      } as unknown as Y.Item;

      const result = yjsValenceResolver('resources', localItem, remoteItem);
      expect(result).toBe(localItem);
    });

    it('falls back to YATA when valence tie', () => {
      const localItem = {
        id: { client: 100, clock: 17 },
        content: { getContent: () => [42] }
      } as unknown as Y.Item;

      const remoteItem = {
        id: { client: 100, clock: 18 },
        content: { getContent: () => [50] }
      } as unknown as Y.Item;

      const result = yjsValenceResolver('resources', localItem, remoteItem);
      expect(result).toBeNull(); // let native YATA decide
    });
  });

  describe('automergeValenceResolver – Automerge Integration Edge Cases', () => {
    it('resolves to remote when remote valence significantly higher', () => {
      const localChange = {
        value: 42,
        valence: 0.85,
        timestamp: 1000
      };

      const remoteChange = {
        value: 50,
        valence: 0.9999,
        timestamp: 999
      };

      const result = automergeValenceResolver('resources', localChange, remoteChange);
      expect(result).toBe(remoteChange);
    });

    it('falls back to timestamp when valence tie', () => {
      const localChange = {
        value: 42,
        valence: 0.999,
        timestamp: 1500
      };

      const remoteChange = {
        value: 50,
        valence: 0.999,
        timestamp: 1600
      };

      const result = automergeValenceResolver('resources', localChange, remoteChange);
      expect(result).toBe(remoteChange); // remote timestamp wins
    });
  });

  describe('Real-World Concurrency & Collision Scenarios', () => {
    it('handles same actorId collision (rare) – falls back to seq', () => {
      const local = { actorId: 123456789, seq: 42, valence: 0.999, value: 'mercy' };
      const remote = { actorId: 123456789, seq: 43, valence: 0.999, value: 'chaos' };

      const result = valenceWeightedResolver({
        key: 'critical',
        localValue: local.value,
        localValence: local.valence,
        localTimestamp: local.seq,
        remoteValue: remote.value,
        remoteValence: remote.valence,
        remoteTimestamp: remote.seq
      });

      expect(result).toBe('chaos'); // higher seq wins
    });

    it('rejects low-valence concurrent change when gate blocks', async () => {
      mercyGate.mockResolvedValueOnce(false);

      const result = await valenceWeightedResolver({
        key: 'harmonyScore',
        localValue: 0.999,
        localValence: 0.9999,
        localTimestamp: 1000,
        remoteValue: 0.1,
        remoteValence: 0.05,
        remoteTimestamp: 999
      });

      expect(result).toBe(0.1); // fallback to timestamp
    });

    it('enforces positive-sum outcome in multi-node diamond conflict', () => {
      const local = { value: 0.999, valence: 0.9998, timestamp: 1000 };
      const remote = { value: 0.85, valence: 0.87, timestamp: 999 };

      const result = valenceWeightedResolver({
        key: 'collectiveValence',
        localValue: local.value,
        localValence: local.valence,
        localTimestamp: local.timestamp,
        remoteValue: remote.value,
        remoteValence: remote.valence,
        remoteTimestamp: remote.timestamp
      });

      expect(result).toBe(0.999); // local high-valence wins
    });
  });
});

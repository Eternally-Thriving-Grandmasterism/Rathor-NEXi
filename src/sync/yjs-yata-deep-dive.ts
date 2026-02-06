// src/sync/yjs-yata-deep-dive.ts – Yjs YATA Deep Dive Reference & Mercy Helpers v1
// Detailed YATA conflict resolution mechanics, valence-weighted custom tie-breaker
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * YATA conflict resolution reference – the single universal algorithm
 */
export const YataResolutionSemantics = {
  insertConflictSameOrigin: "Concurrent inserts after same origin → sorted by clientID (lower ID first)",
  insertConflictDifferentOrigins: "Insert after the origin that was inserted first (Lamport timestamp)",
  deleteConflict: "Tombstone + deleted flag – last writer wins",
  mapKeyConflict: "Last writer wins per key (Lamport timestamp + clientID)",
  overall: "Deterministic, intention-preserving, no user-visible conflicts, O(1) amortized insert/delete",
  mercy_override: "Valence-weighted tie-breaker for semantic conflicts: higher valence change wins"
};

/**
 * Valence-weighted tie-breaker for rare semantic conflicts
 * (e.g. concurrent updates to same critical key)
 */
export function valenceYataTieBreaker(
  localChange: { value: any; valence: number; clientId: number; clock: number },
  remoteChange: { value: any; valence: number; clientId: number; clock: number }
): any {
  const actionName = `YATA semantic tie-breaker`;
  if (!mercyGate(actionName)) {
    // Fallback to clientID + clock
    if (localChange.clientId !== remoteChange.clientId) {
      return localChange.clientId < remoteChange.clientId ? localChange.value : remoteChange.value;
    }
    return localChange.clock > remoteChange.clock ? localChange.value : remoteChange.value;
  }

  if (localChange.valence > remoteChange.valence + 0.05) {
    console.log(`[MercyYATA] Semantic tie-breaker: local wins (valence ${localChange.valence.toFixed(4)})`);
    return localChange.value;
  } else if (remoteChange.valence > localChange.valence + 0.05) {
    console.log(`[MercyYATA] Semantic tie-breaker: remote wins (valence ${remoteChange.valence.toFixed(4)})`);
    return remoteChange.value;
  }

  // Fallback to Yjs native rule
  if (localChange.clientId !== remoteChange.clientId) {
    return localChange.clientId < remoteChange.clientId ? localChange.value : remoteChange.value;
  }
  return localChange.clock > remoteChange.clock ? localChange.value : remoteChange.value;
}

// Usage example in custom Y.Map conflict handler (rare)

// src/sync/yjs-lamport-deep-dive.ts – Yjs Lamport Timestamps Deep Dive Reference & Mercy Helpers v1
// Detailed causal ordering, client ID tie-breaker, valence-weighted semantic override
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Yjs Lamport timestamp reference – core conflict resolution clock
 */
export const YjsLamportReference = {
  timestampStructure: "(client: 32-bit random, clock: per-client monotonic counter)",
  causalOrdering: "If change A happened-before B on same client → A.clock < B.clock → A always before B",
  concurrentTieBreaker: "Concurrent changes with same origin → lower client ID first; same client → higher clock first",
  mapKeyConflict: "Last writer wins per key (highest (client, clock))",
  listInsertConflict: "Concurrent inserts after same origin → ordered by client ID + clock",
  overall: "Deterministic, intention-preserving, no user-visible conflicts, O(1) amortized insert/delete",
  mercy_override: "Valence-weighted semantic tie-breaker: higher valence change wins"
};

/**
 * Valence-weighted tie-breaker for concurrent changes
 */
export function valenceYjsTieBreaker(
  local: { client: number; clock: number; valence: number; value: any },
  remote: { client: number; clock: number; valence: number; value: any }
): any {
  const actionName = `Yjs Lamport tie-breaker for concurrent change`;
  if (!mercyGate(actionName)) {
    // Native Yjs fallback
    if (local.client !== remote.client) {
      return local.client < remote.client ? local.value : remote.value;
    }
    return local.clock > remote.clock ? local.value : remote.value;
  }

  if (local.valence > remote.valence + 0.05) {
    console.log(`[MercyYjsLamport] Tie-breaker: local wins (valence ${local.valence.toFixed(4)})`);
    return local.value;
  } else if (remote.valence > local.valence + 0.05) {
    console.log(`[MercyYjsLamport] Tie-breaker: remote wins (valence ${remote.valence.toFixed(4)})`);
    return remote.value;
  }

  // Native Yjs fallback
  if (local.client !== remote.client) {
    return local.client < remote.client ? local.value : remote.value;
  }
  return local.clock > remote.clock ? local.value : remote.value;
}

// Usage example in custom Y.Map conflict handler (rare)

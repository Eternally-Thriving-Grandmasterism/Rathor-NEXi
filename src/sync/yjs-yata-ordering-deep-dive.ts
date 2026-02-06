// src/sync/yjs-yata-ordering-deep-dive.ts – Yjs YATA Ordering Deep Dive Reference & Mercy Helpers v1
// Detailed YATA ordering rules, valence-weighted semantic override, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Yjs YATA ordering reference – the single universal algorithm
 */
export const YataOrderingReference = {
  coreStructure: "All types are doubly-linked lists of Item nodes",
  itemFields: "id(client,clock), left, right, origin, originRight, content, deleted",
  insertResolution: "Compare origin first → same origin → sort by clientID (lower first) → same client → higher clock",
  deleteResolution: "Tombstone + deleted flag – insert wins if causally after delete",
  mapKeyResolution: "Last writer wins per key (highest (client,clock))",
  overall: "Deterministic total order, strong intention preservation, no user-visible conflicts",
  mercy_override: "Valence-weighted semantic tie-breaker: higher valence change wins"
};

/**
 * Valence-weighted tie-breaker for rare semantic conflicts in Yjs
 */
export function valenceYataTieBreaker(
  local: { client: number; clock: number; valence: number; value: any },
  remote: { client: number; clock: number; valence: number; value: any }
): any {
  const actionName = `YATA semantic tie-breaker`;
  if (!mercyGate(actionName)) {
    // Native Yjs fallback
    if (local.client !== remote.client) {
      return local.client < remote.client ? local.value : remote.value;
    }
    return local.clock > remote.clock ? local.value : remote.value;
  }

  if (local.valence > remote.valence + 0.05) {
    console.log(`[MercyYATA] Semantic tie-breaker: local wins (valence ${local.valence.toFixed(4)})`);
    return local.value;
  } else if (remote.valence > local.valence + 0.05) {
    console.log(`[MercyYATA] Semantic tie-breaker: remote wins (valence ${remote.valence.toFixed(4)})`);
    return remote.value;
  }

  // Native Yjs fallback
  if (local.client !== remote.client) {
    return local.client < remote.client ? local.value : remote.value;
  }
  return local.clock > remote.clock ? local.value : remote.value;
}

// Usage example in custom Y.Map/Y.Array merge handler (advanced use)

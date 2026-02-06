// src/sync/automerge-conflict-resolution.ts – Automerge Conflict Resolution Deep Dive & Mercy Helpers v1
// Detailed semantics reference, valence-weighted custom resolver, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Valence-weighted conflict resolver for Automerge.Map keys
 * Higher valence change wins in rare concurrent update conflicts
 */
export function valenceWeightedMapResolver(
  key: string,
  localChange: Automerge.Change<any>,
  remoteChange: Automerge.Change<any>
): Automerge.Change<any> | null {
  const localValence = localChange.valence ?? 0.5;
  const remoteValence = remoteChange.valence ?? 0.5;

  if (localValence > remoteValence + 0.05) {
    console.log(`[AutomergeMercy] Valence conflict resolved: local wins (${localValence.toFixed(4)} > ${remoteValence.toFixed(4)})`);
    return localChange;
  } else if (remoteValence > localValence + 0.05) {
    console.log(`[AutomergeMercy] Valence conflict resolved: remote wins (${remoteValence.toFixed(4)} > ${localValence.toFixed(4)})`);
    return remoteChange;
  }

  // Fallback to Lamport timestamp
  return localChange.timestamp > remoteChange.timestamp ? localChange : remoteChange;
}

/**
 * Apply custom resolver to map updates (example)
 */
export function mergeWithValencePriority(
  doc: Automerge.Doc<any>,
  key: string,
  localChange: Automerge.Change<any>,
  remoteChange: Automerge.Change<any>
) {
  const winner = valenceWeightedMapResolver(key, localChange, remoteChange);
  if (winner === localChange) {
    Automerge.change(doc, `Valence merge – keeping local for ${key}`, d => {
      d[key] = localChange.value;
    });
  } else if (winner === remoteChange) {
    Automerge.change(doc, `Valence merge – accepting remote for ${key}`, d => {
      d[key] = remoteChange.value;
    });
  }
}

/**
 * Quick reference: Automerge conflict resolution semantics
 */
export const AutomergeConflictSemantics = {
  lists: "Concurrent inserts → ordered by origin + clientID tie-breaker",
  maps: "Last writer wins per key (Lamport timestamp)",
  counters: "All concurrent increments summed (commutative)",
  text: "Span-based last-writer-wins + tombstones",
  subdocs: "Independent documents → no cross-history conflicts",
  mercy_override: "Valence-weighted custom resolver for critical keys"
};

// Usage example in sync handler
// mergeWithValencePriority(parentDoc, 'probe-001-resources', localChange, remoteChange);

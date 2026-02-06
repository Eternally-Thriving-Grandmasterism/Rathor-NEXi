// src/sync/lseq-concurrent-insertion-examples.ts – LSEQ Concurrent Insertion Examples & Mercy Helpers v1
// Detailed diamond scenarios, adaptive allocation, valence-weighted override
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * LSEQ concurrent insertion examples reference – mercy-aligned scenarios
 */
export const LSEQConcurrentInsertionExamples = {
  simpleConcurrentSingleChars: "Alice inserts 'X' after 'a' (boundary+), Bob inserts 'Y' after 'a' (boundary-) → 'a Y X b' or similar – no interleaving",
  concurrentWordBursts: "Alice types 'mom' after 'a' (boundary+ cluster), Bob types 'dad' after 'a' (boundary- cluster) → 'a dad mom b' – groups preserved",
  massiveGapConcurrency: "100 concurrent inserts in same gap → boundary splitting extends length, places in middle of new level, average length sub-linear",
  deletePlusInsert: "Alice deletes 'b', Bob inserts 'X' after 'b' → 'X' wins (causal after delete), final 'a X c'",
  mercy_override: "Valence-weighted semantic tie-breaker: higher valence change wins in rare conflicts"
};

/**
 * Valence-weighted tie-breaker for rare semantic conflicts in LSEQ
 */
export function valenceLSEQTieBreaker(
  local: { position: number[]; valence: number; value: any },
  remote: { position: number[]; valence: number; value: any }
): any {
  const actionName = `LSEQ semantic tie-breaker for concurrent insertion`;
  if (!mercyGate(actionName)) {
    // Native LSEQ fallback (lexicographical order)
    return comparePositions(local.position, remote.position) < 0 ? local.value : remote.value;
  }

  if (local.valence > remote.valence + 0.05) {
    console.log(`[MercyLSEQ] Semantic tie-breaker: local wins (valence ${local.valence.toFixed(4)})`);
    return local.value;
  } else if (remote.valence > local.valence + 0.05) {
    console.log(`[MercyLSEQ] Semantic tie-breaker: remote wins (valence ${remote.valence.toFixed(4)})`);
    return remote.value;
  }

  // Native LSEQ fallback
  return comparePositions(local.position, remote.position) < 0 ? local.value : remote.value;
}

// Helper: lexicographical comparison of position arrays
function comparePositions(p1: number[], p2: number[]): number {
  const len = Math.min(p1.length, p2.length);
  for (let i = 0; i < len; i++) {
    if (p1[i] < p2[i]) return -1;
    if (p1[i] > p2[i]) return 1;
  }
  return p1.length - p2.length;
}

// Usage example in custom LSEQ merge handler (advanced use)

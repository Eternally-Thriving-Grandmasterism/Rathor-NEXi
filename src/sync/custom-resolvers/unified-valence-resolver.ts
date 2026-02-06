// src/sync/custom-resolvers/unified-valence-resolver.ts – Unified Valence-Weighted Resolver v1
// Cross-engine core resolver (Yjs/Automerge/ElectricSQL), mercy-gated, thriving-aligned
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import { fuzzyMercy } from '@/utils/fuzzy-mercy';

const MERCY_THRESHOLD = 0.9999999;

export interface ResolverContext {
  key: string;
  localValue: any;
  localValence: number;
  localTimestamp: number | string;
  remoteValue: any;
  remoteValence: number;
  remoteTimestamp: number | string;
  actionName?: string;
}

/**
 * Core valence-weighted semantic resolver
 * Higher valence wins; fallback to timestamp / native tie-breaker
 */
export function unifiedValenceResolver(context: ResolverContext): any {
  const {
    key,
    localValue,
    localValence,
    localTimestamp,
    remoteValue,
    remoteValence,
    remoteTimestamp,
    actionName = `Resolve semantic conflict for key: ${key}`
  } = context;

  if (!mercyGate(actionName, key)) {
    console.warn(`[MercyResolver] Gate blocked – using timestamp fallback`);
    return localTimestamp > remoteTimestamp ? localValue : remoteValue;
  }

  // Primary: valence difference > threshold
  if (localValence > remoteValence + 0.05) {
    console.log(`[MercyResolver] Valence wins: local (\( {localValence.toFixed(4)}) > remote ( \){remoteValence.toFixed(4)})`);
    return localValue;
  } else if (remoteValence > localValence + 0.05) {
    console.log(`[MercyResolver] Valence wins: remote (\( {remoteValence.toFixed(4)}) > local ( \){localValence.toFixed(4)})`);
    return remoteValue;
  }

  // Secondary: timestamp (native fallback)
  if (localTimestamp > remoteTimestamp) {
    console.log(`[MercyResolver] Timestamp fallback: local wins`);
    return localValue;
  } else if (remoteTimestamp > localTimestamp) {
    console.log(`[MercyResolver] Timestamp fallback: remote wins`);
    return remoteValue;
  }

  // Ultimate fallback: local preference (client sovereignty)
  console.log(`[MercyResolver] Full tie – preserving local value`);
  return localValue;
}

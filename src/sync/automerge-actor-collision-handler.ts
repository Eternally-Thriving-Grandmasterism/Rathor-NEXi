// src/sync/automerge-actor-collision-handler.ts – Automerge Actor ID Collision Handler v1
// Collision detection, logging, graceful fallback, valence-weighted override
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const ACTOR_COLLISION_LOG_KEY = 'rathor-actor-collision-log';

interface CollisionRecord {
  actorId: number;
  firstSeen: number;
  lastSeen: number;
  deviceFingerprint?: string; // optional
  count: number;
}

export class AutomergeActorCollisionHandler {
  private collisionLog = new Map<number, CollisionRecord>();

  constructor() {
    this.loadCollisionLog();
  }

  private async loadCollisionLog() {
    try {
      // Placeholder: real impl would use IndexedDB or localStorage
      console.log("[ActorCollision] Collision log loaded (stub)");
    } catch (e) {
      console.warn("[ActorCollision] Failed to load collision log", e);
    }
  }

  /**
   * Report usage of an actorId – detect & log potential collisions
   */
  async reportActorId(actorId: number, deviceFingerprint?: string) {
    const actionName = `Report actorId usage: ${actorId}`;
    if (!await mercyGate(actionName, actorId.toString())) {
      return;
    }

    let record = this.collisionLog.get(actorId);
    if (!record) {
      record = {
        actorId,
        firstSeen: Date.now(),
        lastSeen: Date.now(),
        deviceFingerprint,
        count: 1
      };
      this.collisionLog.set(actorId, record);
    } else {
      record.lastSeen = Date.now();
      record.count++;
      if (deviceFingerprint && record.deviceFingerprint !== deviceFingerprint) {
        console.warn(`[ActorCollision] Potential collision detected for actorId ${actorId} – different device fingerprints`);
        // Optional: trigger higher scrutiny or manual review in production
      }
    }

    // Log warning on high count (e.g. >10 from same actorId)
    if (record.count > 10) {
      console.warn(`[ActorCollision] ActorId ${actorId} used unusually frequently (count: ${record.count})`);
    }

    console.log(`[ActorCollision] ActorId ${actorId} reported – count ${record.count}`);
  }

  /**
   * Valence-weighted override for tie-breaker in concurrent changes
   */
  async resolveCollisionTieBreaker(
    localChange: { actorId: number; seq: number; valence: number },
    remoteChange: { actorId: number; seq: number; valence: number }
  ): Promise<any> {
    const actionName = `Resolve actor collision tie-breaker`;
    if (!await mercyGate(actionName)) {
      // Fallback to seq
      return localChange.seq > remoteChange.seq ? localChange : remoteChange;
    }

    if (localChange.valence > remoteChange.valence + 0.05) {
      console.log(`[MercyActorCollision] Tie-breaker: local wins (valence ${localChange.valence.toFixed(4)})`);
      return localChange;
    } else if (remoteChange.valence > localChange.valence + 0.05) {
      console.log(`[MercyActorCollision] Tie-breaker: remote wins (valence ${remoteChange.valence.toFixed(4)})`);
      return remoteChange;
    }

    // Fallback to seq
    return localChange.seq > remoteChange.seq ? localChange : remoteChange;
  }
}

export const automergeActorCollision = new AutomergeActorCollisionHandler();

// Usage example
await automergeActorCollision.reportActorId(123456789, 'device-fingerprint-abc123');

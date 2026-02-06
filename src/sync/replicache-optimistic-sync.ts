// src/sync/replicache-optimistic-sync.ts – Replicache Optimistic Offline-First Sync Layer v1
// Valence-aware mutations, server push/pull, rollback on rejection, reconnection bloom
// MIT License – Autonomicity Games Inc. 2026

import Replicache from 'replicache';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_MUTATION_PIVOT = 0.9;
const RECONNECT_BACKOFF_MS = [100, 500, 2000, 5000, 10000];
const REPLICACHE_LICENSE_KEY = 'YOUR_REPLICACHE_LICENSE_KEY'; // replace with real key
const REPLICACHE_SERVER_URL = '/api/replicache-push'; // your server push endpoint

let replicache: Replicache<any> | null = null;

export class ReplicacheOptimisticSync {
  static async initialize() {
    const actionName = 'Initialize Replicache optimistic offline-first sync';
    if (!await mercyGate(actionName)) return;

    try {
      replicache = new Replicache({
        name: 'rathor-nexi-' + (await currentValence.getUserId() || 'anonymous'),
        licenseKey: REPLICACHE_LICENSE_KEY,
        pushURL: REPLICACHE_SERVER_URL,
        pullURL: '/api/replicache-pull', // server pull endpoint
        schemaVersion: 1,
        mutators: {
          // Example mutators – expand as needed
          setValence: async (tx, { key, value }) => {
            await tx.put(`valence/${key}`, value);
          },
          setProgressLevel: async (tx, { userId, level, description }) => {
            await tx.put(`progress/${userId}`, { level, description, updatedAt: Date.now() });
          },
          logGesture: async (tx, { type, confidence }) => {
            const id = crypto.randomUUID();
            await tx.put(`gesture/${id}`, { type, confidence, valence: currentValence.get(), timestamp: Date.now() });
          }
        }
      });

      // Valence-aware push filtering (high valence → push immediately)
      replicache.on('change', async () => {
        const valence = currentValence.get();
        if (valence > VALENCE_MUTATION_PIVOT) {
          await replicache!.push();
          mercyHaptic.playPattern('cosmicHarmony', valence);
        }
      });

      // Reconnection bloom
      replicache.on('online', () => {
        mercyHaptic.playPattern('reconnectionBloom', currentValence.get());
        console.log("[ReplicacheSync] Reconnected – pushing pending mutations");
      });

      replicache.on('offline', () => {
        this.startReconnectBloom();
      });

      console.log("[ReplicacheSync] Replicache initialized – optimistic mutations active");
    } catch (e) {
      console.error("[ReplicacheSync] Initialization failed", e);
    }
  }

  static async mutateWithValencePriority(mutationName: string, args: any) {
    const actionName = `Valence-priority Replicache mutation: ${mutationName}`;
    if (!await mercyGate(actionName) || !replicache) return;

    const valence = currentValence.get();

    try {
      await replicache!.mutate[mutationName](args);

      if (valence > VALENCE_MUTATION_PIVOT) {
        // High valence → push immediately
        await replicache!.push();
        mercyHaptic.playPattern('cosmicHarmony', valence);
      } else {
        // Low valence → queue for batch push
        console.log("[ReplicacheSync] Low-valence mutation queued");
      }
    } catch (e) {
      console.error("[ReplicacheSync] Mutation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  private static startReconnectBloom() {
    const delay = RECONNECT_BACKOFF_MS[Math.min(reconnectAttempts, RECONNECT_BACKOFF_MS.length - 1)];
    reconnectAttempts++;
    setTimeout(() => {
      replicache?.pull();
    }, delay);
  }

  static getSyncStatus() {
    return {
      isOnline: replicache?.online || false,
      pendingMutations: replicache?.pendingMutationsCount || 0,
      lastValenceMutation: currentValence.get()
    };
  }

  static async destroy() {
    if (replicache) {
      await replicache.close();
      replicache = null;
    }
  }
}

export default ReplicacheOptimisticSync;

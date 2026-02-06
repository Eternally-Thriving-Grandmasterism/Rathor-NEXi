// src/sync/automerge-actor-id-manager.ts – Automerge Actor ID Manager v1
// Generation, persistence, collision awareness, mercy gates, valence-modulated identity
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const ACTOR_ID_KEY = 'rathor-automerge-actor-id';

export class AutomergeActorIdManager {
  private _actorId: number | null = null;

  constructor() {
    this.loadOrGenerateActorId();
  }

  private async loadOrGenerateActorId() {
    try {
      // Try to load from IndexedDB / localStorage
      const stored = localStorage.getItem(ACTOR_ID_KEY);
      if (stored) {
        const id = parseInt(stored, 10);
        if (!isNaN(id) && id >= 0 && id < 0x100000000) {
          this._actorId = id;
          console.log(`[ActorIdManager] Loaded persisted actorId: ${id}`);
          return;
        }
      }

      // Generate new random 32-bit actorId
      this._actorId = Math.floor(Math.random() * 0x100000000);
      localStorage.setItem(ACTOR_ID_KEY, this._actorId.toString());
      console.log(`[ActorIdManager] Generated new actorId: ${this._actorId}`);
    } catch (e) {
      console.warn("[ActorIdManager] Failed to load/generate actorId", e);
      // Fallback: temporary ID
      this._actorId = Math.floor(Math.random() * 0x100000000);
    }
  }

  getActorId(): number {
    if (this._actorId === null) {
      throw new Error("Actor ID not initialized");
    }
    return this._actorId;
  }

  /**
   * Valence-modulated actor ID awareness (future-proof extension)
   */
  async reportActorIdUsage(actionName: string, requiredValence: number = MERCY_THRESHOLD) {
    if (!await mercyGate(actionName, `ActorId usage: ${this.getActorId()}`, requiredValence)) {
      return;
    }
    console.log(`[ActorIdManager] Actor ${this.getActorId()} used for ${actionName} – valence ${currentValence.get().toFixed(8)}`);
  }
}

export const automergeActorId = new AutomergeActorIdManager();

// Usage example
// await automergeActorId.reportActorIdUsage("Probe fleet sync", 0.9995);
// const actor = automergeActorId.getActorId();

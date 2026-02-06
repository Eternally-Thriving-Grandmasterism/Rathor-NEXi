// src/core/valence-tracker.ts
import { openProgressDB } from '@/ui/mercy-onboarding-growth-dashboard';

class ValenceTracker {
  private _valence: number = 0.5;

  constructor() {
    this.loadFromStorage();
  }

  async loadFromStorage() {
    try {
      const progress = await getUserProgress(); // from dashboard db
      this._valence = progress.valence;
    } catch (e) {
      console.warn("[ValenceTracker] Failed to load from storage", e);
    }
  }

  async setValence(newValence: number) {
    this._valence = Math.max(0, Math.min(1, newValence));
    try {
      const progress = await getUserProgress();
      await updateUserProgress(progress.level, this._valence, 0);
    } catch (e) {
      console.warn("[ValenceTracker] Failed to persist valence", e);
    }
  }

  get() {
    return this._valence;
  }

  async addDelta(delta: number) {
    await this.setValence(this._valence + delta);
  }
}

export const currentValence = new ValenceTracker();

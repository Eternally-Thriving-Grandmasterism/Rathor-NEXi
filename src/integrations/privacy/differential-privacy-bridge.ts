// src/integrations/privacy/differential-privacy-bridge.ts – Differential Privacy Bridge v1
// Valence-weighted Gaussian noise, DP-SGD stub, mercy-gated, privacy budget tracking
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const BASE_NOISE_SCALE = 1.0; // calibrated for ε ≈ 1–3 per epoch
const VALENCE_NOISE_PIVOT = 0.9;
const PRIVACY_BUDGET_EPSILON = 3.0; // total ε per training run (adjust per use case)

export class DifferentialPrivacyBridge {
  private currentEpsilonSpent = 0;

  /**
   * Valence-weighted Gaussian noise addition
   * High valence → less noise (higher utility on thriving paths)
   */
  async addValenceWeightedNoise(value: number, sensitivity: number = 1.0): Promise<number> {
    const actionName = 'Add valence-weighted DP noise';
    if (!await mercyGate(actionName)) return value;

    const valence = currentValence.get();
    const noiseScale = BASE_NOISE_SCALE / (1 + Math.max(0, valence - VALENCE_NOISE_PIVOT) * 5); // less noise on high valence

    const noise = tf.randomNormal([1], 0, noiseScale * sensitivity).dataSync()[0];
    const noisyValue = value + noise;

    // Approximate ε spend (simplified moments accountant)
    this.currentEpsilonSpent += Math.log(1 + Math.exp(noiseScale)); // rough estimate

    if (this.currentEpsilonSpent > PRIVACY_BUDGET_EPSILON) {
      console.warn(`[DPBridge] Privacy budget nearly exhausted (ε spent: ${this.currentEpsilonSpent.toFixed(2)})`);
    }

    return noisyValue;
  }

  /**
   * Valence-weighted DP-SGD gradient clipping & noise (stub)
   */
  async applyDPGradient(gradients: tf.Tensor[]): Promise<tf.Tensor[]> {
    if (!await mercyGate('Apply DP-SGD to gradients')) return gradients;

    const clippedGradients = gradients.map(g => {
      const norm = tf.norm(g).dataSync()[0];
      const clipNorm = 1.0; // per-sample clip norm
      if (norm > clipNorm) {
        return tf.mul(g, tf.scalar(clipNorm / norm));
      }
      return g;
    });

    // Add valence-weighted noise to averaged gradients
    const noisyGradients = await Promise.all(clippedGradients.map(async g => {
      const flat = await g.flatten().data();
      const noisyFlat = await Promise.all(flat.map(v => this.addValenceWeightedNoise(v)));
      return tf.tensor(noisyFlat).reshape(g.shape);
    }));

    return noisyGradients;
  }

  /**
   * Check remaining privacy budget
   */
  getRemainingPrivacyBudget(): number {
    return Math.max(0, PRIVACY_BUDGET_EPSILON - this.currentEpsilonSpent);
  }
}

export const dpBridge = new DifferentialPrivacyBridge();

// Usage example in training loop
// const noisyGradients = await dpBridge.applyDPGradient(gradients);
// optimizer.applyGradients(noisyGradients);

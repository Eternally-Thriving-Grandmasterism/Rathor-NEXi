// src/integrations/privacy/federated-dp-bridge.ts – Federated Learning with DP Bridge v1
// Valence-weighted aggregation, DP-SGD stub, privacy budget tracking, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const BASE_NOISE_SCALE = 1.0;
const VALENCE_WEIGHT_PIVOT = 0.9;
const PRIVACY_BUDGET_EPSILON = 4.0; // total ε per training round

export class FederatedDPBridge {
  private currentEpsilonSpent = 0;

  /**
   * Valence-weighted local gradient noise addition (DP-SGD style)
   */
  async addValenceWeightedNoise(gradient: number, sensitivity: number = 1.0): Promise<number> {
    const actionName = 'Add valence-weighted DP noise to gradient';
    if (!await mercyGate(actionName)) return gradient;

    const valence = currentValence.get();
    const noiseScale = BASE_NOISE_SCALE / (1 + Math.max(0, valence - VALENCE_WEIGHT_PIVOT) * 5);

    const noise = tf.randomNormal([1], 0, noiseScale * sensitivity).dataSync()[0];
    const noisyGradient = gradient + noise;

    this.currentEpsilonSpent += Math.log(1 + Math.exp(noiseScale)); // rough ε estimate

    if (this.currentEpsilonSpent > PRIVACY_BUDGET_EPSILON) {
      console.warn(`[FedDPBridge] Privacy budget nearly exhausted (ε spent: ${this.currentEpsilonSpent.toFixed(2)})`);
    }

    return noisyGradient;
  }

  /**
   * Valence-weighted federated aggregation (simulated server-side)
   */
  async aggregateUpdates(localUpdates: number[], localValences: number[]): Promise<number> {
    const actionName = 'Valence-weighted federated aggregation';
    if (!await mercyGate(actionName)) return 0;

    let totalWeight = 0;
    let weightedSum = 0;

    for (let i = 0; i < localUpdates.length; i++) {
      const valence = localValences[i];
      const weight = Math.exp(2.0 * (valence - 0.9)); // exponential boost for high valence
      weightedSum += localUpdates[i] * weight;
      totalWeight += weight;
    }

    const aggregated = totalWeight > 0 ? weightedSum / totalWeight : 0;

    console.log(`[FedDPBridge] Aggregated update: ${aggregated.toFixed(6)} from ${localUpdates.length} nodes`);

    return aggregated;
  }

  /**
   * Check remaining privacy budget
   */
  getRemainingPrivacyBudget(): number {
    return Math.max(0, PRIVACY_BUDGET_EPSILON - this.currentEpsilonSpent);
  }
}

export const fedDPBridge = new FederatedDPBridge();

// Usage example in federated training loop (edge node)
// const noisyGradient = await fedDPBridge.addValenceWeightedNoise(rawGradient);
// send noisyGradient to server

// Server-side (simulated)
// const globalUpdate = await fedDPBridge.aggregateUpdates(localGradients, localValences);

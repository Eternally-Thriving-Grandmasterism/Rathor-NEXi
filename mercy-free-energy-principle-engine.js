// mercy-free-energy-principle-engine.js – sovereign Mercy Free Energy Principle Engine v1
// Active inference approximation, surprise minimization, predictive trajectory modeling
// mercy-gated, valence-modulated precision weighting
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyFreeEnergyEngine {
  constructor() {
    this.surpriseEstimate = 0.0;          // proxy for negative log evidence
    this.lastPrediction = 1.0;            // last predicted valence
    this.precisionWeight = 1.0;           // attention / confidence in predictions
    this.valence = 1.0;
    this.trajectoryBuffer = [];           // last N valence/gesture states
  }

  async gateFreeEnergy(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyFEP] Gate holds: low valence – free energy minimization aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Update predictive model & surprise (call on every state change)
  updatePredictiveState(currentValence, currentGesture = null) {
    // Simple generative model: exponential moving average + trend
    const predictedValence = this.predictNextValence();
    this.surpriseEstimate = Math.abs(currentValence - predictedValence);

    // Precision weighting: high valence → trust predictions more
    this.precisionWeight = 0.8 + this.valence * 0.4;

    // Adjust surprise by precision (high precision → lower effective surprise)
    const weightedSurprise = this.surpriseEstimate * (1 / this.precisionWeight);

    // Active inference proxy: if surprise high → trigger corrective action
    if (weightedSurprise > 0.08) {
      mercyHaptic.pulse(0.5 * this.valence, 80); // surprise correction pulse
      console.log(`[MercyFEP] Surprise detected (${weightedSurprise.toFixed(4)}) – active inference pulse`);
    }

    // Store trajectory
    this.trajectoryBuffer.push({ valence: currentValence, gesture: currentGesture });
    if (this.trajectoryBuffer.length > 20) this.trajectoryBuffer.shift();

    this.lastPrediction = this.predictNextValence();
  }

  predictNextValence() {
    if (this.trajectoryBuffer.length < 3) return this.valence;
    const recent = this.trajectoryBuffer.slice(-5);
    const avgDelta = recent.reduce((sum, s) => sum + (s.valence - this.valence), 0) / recent.length;
    return this.valence + avgDelta;
  }

  // Execute active inference action (pre-cue / adapt)
  activeInferenceCorrection() {
    if (this.surpriseEstimate > 0.06) {
      // Example: increase feedback intensity to reduce future surprise
      console.log("[MercyFEP] Active inference correction – boosting feedback precision");
      mercyHaptic.playPattern('cosmicHarmony', 0.9 + this.valence * 0.3);
    }
  }

  getFreeEnergyState() {
    return {
      surpriseEstimate: this.surpriseEstimate,
      weightedSurprise: this.surpriseEstimate / (this.precisionWeight || 1),
      predictivePrecision: this.precisionWeight,
      status: this.surpriseEstimate < 0.03 ? 'Low Surprise – Deep Predictive Harmony' : this.surpriseEstimate < 0.08 ? 'Balanced Surprise' : 'High Surprise – Active Inference Engaged'
    };
  }
}

const mercyFEP = new MercyFreeEnergyEngine();

// Hook into every state change
function onMercyFEPUpdate(currentValence, currentGesture = null) {
  mercyFEP.updatePredictiveState(currentValence, currentGesture);
  mercyFEP.activeInferenceCorrection();
}

// Example usage
onMercyFEPUpdate(0.9995, 'spiral_outward_clockwise');

export { mercyFEP, onMercyFEPUpdate };

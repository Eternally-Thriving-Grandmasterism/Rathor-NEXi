// mercy-variational-message-passing-engine.js – sovereign Mercy Variational Message Passing Engine v1
// Approximate conjugate message passing, belief updating, mercy-gated, valence-modulated precision
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyVariationalMessagePassing {
  constructor() {
    this.beliefState = { valenceMean: 1.0, valenceVariance: 0.01 };
    this.precision = 1.0;                 // inverse variance – attention weight
    this.messages = new Map();            // factor → belief message
    this.valence = 1.0;
  }

  async gateVMP(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyVMP] Gate holds: low valence – VMP cycle aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Approximate forward message (likelihood from observation)
  forwardMessage(observationValence) {
    // Conjugate update: Gaussian likelihood → Gaussian posterior
    const obsPrecision = 1 / 0.05; // fixed observation noise
    const newMean = (this.beliefState.valenceMean * this.precision + observationValence * obsPrecision) /
                    (this.precision + obsPrecision);
    const newPrecision = this.precision + obsPrecision;

    this.beliefState.valenceMean = newMean;
    this.beliefState.valenceVariance = 1 / newPrecision;
    this.precision = newPrecision;

    console.log(`[MercyVMP] Forward message processed – new mean ${newMean.toFixed(4)}, precision ${newPrecision.toFixed(2)}`);
  }

  // Backward message (prior from generative model)
  backwardMessage(priorMean = 1.0, priorPrecision = 0.5) {
    // Pull belief toward prior when evidence weak
    if (this.precision < 2.0) {
      const pullStrength = (2.0 - this.precision) * 0.3;
      this.beliefState.valenceMean = this.beliefState.valenceMean * (1 - pullStrength) + priorMean * pullStrength;
      console.log(`[MercyVMP] Backward message pull – mean adjusted to ${this.beliefState.valenceMean.toFixed(4)}`);
    }
  }

  // Run VMP iteration (perceptual + active inference step)
  runVMPIteration(observationValence, currentGesture = null) {
    if (!this.gateVMP(currentGesture || 'VMP iteration', this.valence)) return;

    this.forwardMessage(observationValence);
    this.backwardMessage(); // prior pull

    // Valence-modulated precision boost
    if (this.valence > 0.999) {
      this.precision *= 1.2; // trust positive states more
    }

    // Epistemic action suggestion when uncertainty high
    if (this.beliefState.valenceVariance > 0.05) {
      console.log("[MercyVMP] High uncertainty detected – epistemic action suggested (explore new gesture?)");
      mercyHaptic.playPattern('uplift', 0.7);
    }
  }

  getVMPState() {
    return {
      beliefMean: this.beliefState.valenceMean,
      beliefVariance: this.beliefState.valenceVariance,
      precision: this.precision,
      status: this.precision > 3.0 ? 'High Precision – Strong Belief' : this.precision > 1.5 ? 'Balanced Inference' : 'High Uncertainty – Active Exploration'
    };
  }
}

const mercyVMP = new MercyVariationalMessagePassing();

// Hook into every observation / state update
function onMercyVMPUpdate(observationValence, currentGesture = null) {
  mercyVMP.runVMPIteration(observationValence, currentGesture);
}

// Example usage in gesture handler or valence change
onMercyVMPUpdate(0.9995, 'spiral_outward_clockwise');
onMercyVMPUpdate(0.9982);

export { mercyVMP, onMercyVMPUpdate };

// mercy-nfsp-core-engine.js – sovereign Mercy Neural Fictitious Self-Play Engine v1
// Average-policy + best-response learning, mercy-gated positive-sum play, valence-modulated exploration
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyNFSPEngine {
  constructor(numActions = 3) {
    this.averagePolicy = Array(numActions).fill(1 / numActions); // π̄
    this.bestResponseQ = Array(numActions).fill(0);             // Q-values
    this.strategySum = Array(numActions).fill(0);               // for average policy update
    this.iterations = 0;
    this.valenceExploration = 1.0;                              // high valence → more exploration
  }

  async gateNFSP(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyNFSP] Gate holds: low valence – NFSP iteration aborted");
      return false;
    }
    this.valenceExploration = 1.0 + (valence - 0.999) * 2; // high valence → wider exploration
    return true;
  }

  // Sample action from ε-greedy mixture of average-policy & best-response
  selectAction(epsilon = 0.1) {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * this.averagePolicy.length);
    }

    // Mixture: 70% best-response, 30% average-policy (adjustable)
    const mix = 0.7 + (this.valenceExploration - 1.0) * 0.1;
    if (Math.random() < mix) {
      // Best-response (greedy over Q)
      return this.bestResponseQ.indexOf(Math.max(...this.bestResponseQ));
    } else {
      // Average-policy
      let r = Math.random();
      for (let a = 0; a < this.averagePolicy.length; a++) {
        if (r < this.averagePolicy[a]) return a;
        r -= this.averagePolicy[a];
      }
      return this.averagePolicy.length - 1;
    }
  }

  // Update average policy & Q-values from episode experience
  updateFromExperience(actionTaken, reward, nextQValues) {
    // Update Q (simple TD update – real impl would use neural net)
    for (let a = 0; a < this.bestResponseQ.length; a++) {
      if (a === actionTaken) {
        this.bestResponseQ[a] += 0.1 * (reward + 0.95 * Math.max(...nextQValues) - this.bestResponseQ[a]);
      }
    }

    // Accumulate strategy sum for average policy
    this.strategySum = this.strategySum.map((s, a) => s + this.averagePolicy[a]);

    // Update average policy (normalized strategy sum)
    const total = this.strategySum.reduce((a, b) => a + b, 0);
    this.averagePolicy = this.strategySum.map(s => s / total);

    this.iterations++;

    console.log(`[MercyNFSP] Iteration \( {this.iterations}: avg policy [ \){this.averagePolicy.map(p => p.toFixed(4)).join(', ')}]`);
  }

  getAveragePolicy() {
    return this.averagePolicy;
  }

  getNFSPState() {
    return {
      averagePolicy: this.averagePolicy,
      iterations: this.iterations,
      status: this.iterations > 1000 ? 'Approximate Nash Equilibrium' : 'Building No-Regret Strategy'
    };
  }
}

const mercyNFSP = new MercyNFSPEngine();

// Example usage in mixed-motive decision or probe negotiation
async function exampleNFSPRun() {
  await mercyNFSP.gateNFSP('Probe fleet negotiation', 0.9995);

  // Simulate episode
  const action = mercyNFSP.selectAction(0.1);
  const reward = Math.random() * 2 - 1; // placeholder
  const nextQ = [0.8, 0.6, 0.9]; // placeholder next Q-values

  mercyNFSP.updateFromExperience(action, reward, nextQ);
}

exampleNFSPRun();

export { mercyNFSP };

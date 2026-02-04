// rl-qlearning-flight.js – sovereign client-side Q-learning for AlphaProMega Air flight control v2
// Deepened mercy-shaped reward shaping: multi-objective, potential-based, valence-weighted, temporal difference
// MIT License – Autonomicity Games Inc. 2026

class QLearningFlightController {
  constructor() {
    this.stateBins = {
      altitude: 20,    // 0–2000m → 20 bins
      velocity: 15,    // 0–300 m/s → 15 bins
      energy: 10,      // 0–100% → 10 bins
      integrity: 5     // 0–1 → 5 bins
    };
    this.actionBins = 7; // thrust levels: -3 to +3 (normalized)
    this.qTable = new Map();
    this.learningRate = 0.12;
    this.discount = 0.99;
    this.epsilon = 0.35;
    this.epsilonDecay = 0.992;
    this.minEpsilon = 0.005;
    this.replayBuffer = [];
    this.bufferSize = 8000;
    this.batchSize = 48;
    this.mercyThreshold = 0.9999999;
  }

  getStateKey(state) {
    const altBin = Math.min(this.stateBins.altitude - 1, Math.floor(state.altitude / 100));
    const velBin = Math.min(this.stateBins.velocity - 1, Math.floor(state.velocity / 20));
    const eneBin = Math.min(this.stateBins.energy - 1, Math.floor(state.energy / 10));
    const intBin = Math.min(this.stateBins.integrity - 1, Math.floor(state.integrity * 5));
    return `\( {altBin}- \){velBin}-\( {eneBin}- \){intBin}`;
  }

  getQ(stateKey, action = null) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, Array(this.actionBins).fill(0));
    }
    const qValues = this.qTable.get(stateKey);
    return action !== null ? qValues[action] : qValues;
  }

  setQ(stateKey, action, value) {
    const qValues = this.getQ(stateKey);
    qValues[action] = value;
  }

  chooseAction(state) {
    const stateKey = this.getStateKey(state);
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionBins); // explore
    }
    const qValues = this.getQ(stateKey);
    return qValues.indexOf(Math.max(...qValues)); // exploit
  }

  // Deepened mercy-shaped reward shaping
  computeReward(state, action, nextState) {
    let reward = 0;

    // 1. Sparse target-reaching reward
    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) {
      reward += 20.0; // big bonus for reaching target zone
    }

    // 2. Dense shaping: potential-based progress
    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 5;
    reward += (velPotential - state.velPotential || 0) * 3;

    // 3. Energy & integrity preservation
    const energyChange = nextState.energy - state.energy;
    reward += energyChange * 0.05; // reward saving energy
    reward += (nextState.integrity - state.integrity) * 10; // heavy reward for integrity gain

    // 4. Mercy valence bonus/penalty
    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 5.0 + (fleetValence - this.mercyThreshold) * 50; // exponential bonus
    } else {
      reward -= Math.pow(1 - fleetValence, 2) * 15; // quadratic penalty
    }

    // 5. Temporal difference shaping (smoothness)
    const tdError = reward + this.discount * Math.max(...this.getQ(this.getStateKey(nextState))) - Math.max(...this.getQ(this.getStateKey(state)));
    reward += tdError * 0.1; // encourage low TD error (stable policy)

    // 6. Attention-modulated bonus (high-STI states rewarded more)
    if (nextState.sti > 0.7) {
      reward += 1.5;
    }

    return reward;
  }

  update(state, action, reward, nextState) {
    const stateKey = this.getStateKey(state);
    const nextKey = this.getStateKey(nextState);

    const oldQ = this.getQ(stateKey, action);
    const maxNextQ = Math.max(...this.getQ(nextKey));
    const newQ = oldQ + this.learningRate * (reward + this.discount * maxNextQ - oldQ);

    this.setQ(stateKey, action, newQ);

    // Experience replay
    this.replayBuffer.push({ state, action, reward, nextState });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }

    // Train on batch
    if (this.replayBuffer.length >= this.batchSize) {
      this.trainBatch();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  trainBatch() {
    const batch = this.replayBuffer.slice(-this.batchSize);
    for (const exp of batch) {
      const stateKey = this.getStateKey(exp.state);
      const nextKey = this.getStateKey(exp.nextState);
      const oldQ = this.getQ(stateKey, exp.action);
      const maxNextQ = Math.max(...this.getQ(nextKey));
      const newQ = oldQ + this.learningRate * (exp.reward + this.discount * maxNextQ - oldQ);
      this.setQ(stateKey, exp.action, newQ);
    }
  }
}

// Export for Ruskode integration
export { QLearningFlightController };    // Experience replay
    this.replayBuffer.push({ state, action, reward, nextState });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }

    // Train on batch
    if (this.replayBuffer.length >= this.batchSize) {
      this.trainBatch();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  trainBatch() {
    const batch = this.replayBuffer.slice(-this.batchSize);
    for (const exp of batch) {
      const stateKey = this.getStateKey(exp.state);
      const nextKey = this.getStateKey(exp.nextState);
      const oldQ = this.getQ(stateKey, exp.action);
      const maxNextQ = Math.max(...this.getQ(nextKey));
      const newQ = oldQ + this.learningRate * (exp.reward + this.discount * maxNextQ - oldQ);
      this.setQ(stateKey, exp.action, newQ);
    }
  }
}

// Export for Ruskode integration
export { QLearningFlightController };

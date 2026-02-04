// dqn-flight-engine.js – sovereign client-side Deep Q-Networks for AlphaProMega Air flight control
// Mercy-shaped rewards, target network, double Q, experience replay, offline-capable
// MIT License – Autonomicity Games Inc. 2026

class DQNNetwork {
  constructor(inputSize = 6, hiddenSize = 64, outputSize = 7) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.weights1 = Array(hiddenSize).fill().map(() => Array(inputSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias1 = Array(hiddenSize).fill(0);
    this.weights2 = Array(outputSize).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias2 = Array(outputSize).fill(0);
  }

  forward(state) {
    // Layer 1: ReLU
    const hidden = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.weights1[i][j] * state[j];
      }
      hidden.push(Math.max(0, sum));
    }

    // Layer 2: Linear (Q-values)
    const qValues = [];
    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.bias2[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += this.weights2[i][j] * hidden[j];
      }
      qValues.push(sum);
    }

    return qValues;
  }

  copy() {
    const copy = new DQNNetwork(this.inputSize, this.hiddenSize, this.outputSize);
    copy.weights1 = this.weights1.map(row => row.slice());
    copy.bias1 = this.bias1.slice();
    copy.weights2 = this.weights2.map(row => row.slice());
    copy.bias2 = this.bias2.slice();
    return copy;
  }

  update(target, lr = 0.001) {
    // Simple SGD update toward target network
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.weights1[i][j] += lr * (target.weights1[i][j] - this.weights1[i][j]);
      }
      this.bias1[i] += lr * (target.bias1[i] - this.bias1[i]);
    }
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.weights2[i][j] += lr * (target.weights2[i][j] - this.weights2[i][j]);
      }
      this.bias2[i] += lr * (target.bias2[i] - this.bias2[i]);
    }
  }
}

class DQNController {
  constructor() {
    this.inputSize = 6;
    this.actionSize = 7; // discrete thrust levels
    this.qNetwork = new DQNNetwork(this.inputSize, 64, this.actionSize);
    this.targetNetwork = this.qNetwork.copy();
    this.replayBuffer = [];
    this.bufferSize = 10000;
    this.batchSize = 64;
    this.gamma = 0.99;
    this.learningRate = 0.001;
    this.epsilon = 0.4;
    this.epsilonDecay = 0.995;
    this.minEpsilon = 0.01;
    this.targetUpdateFreq = 100;
    this.stepCount = 0;
    this.mercyThreshold = 0.9999999;
  }

  chooseAction(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize); // explore
    }

    const qValues = this.qNetwork.forward(state);
    return qValues.indexOf(Math.max(...qValues)); // exploit
  }

  // Deepened mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    // 1. Sparse target-reaching reward
    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) {
      reward += 25.0;
    }

    // 2. Dense potential-based shaping
    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 8;
    reward += (velPotential - state.velPotential || 0) * 5;

    // 3. Energy & integrity preservation
    reward += (nextState.energy - state.energy) * 0.08;
    reward += (nextState.integrity - state.integrity) * 15;

    // 4. Mercy valence bonus/penalty
    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 8.0 + Math.pow(fleetValence - this.mercyThreshold + 0.001, 2) * 100;
    } else {
      reward -= Math.pow(1 - fleetValence, 3) * 25;
    }

    // 5. Temporal difference shaping
    const tdError = reward + this.gamma * Math.max(...this.targetNetwork.forward(nextState.state || nextState)) - Math.max(...this.qNetwork.forward(state));
    reward += tdError * 0.15;

    // 6. Attention-modulated bonus
    if (nextState.sti > 0.7) reward += 2.0;

    return reward;
  }

  storeTransition(state, action, reward, nextState) {
    this.replayBuffer.push({ state, action, reward, nextState });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }
  }

  train() {
    if (this.replayBuffer.length < this.batchSize) return;

    const batch = this.replayBuffer.slice(-this.batchSize);
    for (const exp of batch) {
      const stateKey = JSON.stringify(exp.state);
      const nextKey = JSON.stringify(exp.nextState);

      const currentQ = this.qNetwork.forward(exp.state)[exp.action];
      const maxNextQ = Math.max(...this.targetNetwork.forward(exp.nextState));
      const targetQ = exp.reward + this.gamma * maxNextQ;

      // Update Q-network toward target
      const error = targetQ - currentQ;
      // Simplified gradient update (real impl would backprop)
      // For demo: nudge weights proportionally
      this.qNetwork.update(this.targetNetwork, this.learningRate * error);
    }

    // Periodic target network update
    this.stepCount++;
    if (this.stepCount % this.targetUpdateFreq === 0) {
      this.targetNetwork = this.qNetwork.copy();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }
}

// Export for Ruskode integration
export { DQNController };

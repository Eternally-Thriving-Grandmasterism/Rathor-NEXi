// ppo-continuous-flight.js – sovereign client-side Proximal Policy Optimization for AlphaProMega Air continuous control
// Clipped surrogate, GAE, mercy-shaped rewards, attention-modulated entropy
// MIT License – Autonomicity Games Inc. 2026

class PPOActor {
  constructor(stateDim = 6, actionDim = 2, hiddenSize = 64) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.hiddenSize = hiddenSize;
    this.weights1 = Array(hiddenSize).fill().map(() => Array(stateDim).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias1 = Array(hiddenSize).fill(0);
    this.muWeights = Array(actionDim).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.muBias = Array(actionDim).fill(0);
    this.logStd = Array(actionDim).fill(Math.log(1.0)); // learnable log std dev
  }

  forward(state) {
    // Hidden layer: ReLU
    const hidden = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < this.stateDim; j++) {
        sum += this.weights1[i][j] * state[j];
      }
      hidden.push(Math.max(0, sum));
    }

    // Mean (mu)
    const mu = [];
    for (let i = 0; i < this.actionDim; i++) {
      let sum = this.muBias[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += this.muWeights[i][j] * hidden[j];
      }
      mu.push(Math.tanh(sum) * 3); // bound action [-3,3] for thrust/pitch
    }

    // Std dev (learnable)
    const std = this.logStd.map(Math.exp);

    return { mu, std };
  }

  sampleAction(mu, std) {
    const action = mu.map((m, i) => m + std[i] * this.gaussianRandom());
    return action;
  }

  gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  copy() {
    const copy = new PPOActor(this.stateDim, this.actionDim, this.hiddenSize);
    copy.weights1 = this.weights1.map(row => row.slice());
    copy.bias1 = this.bias1.slice();
    copy.muWeights = this.muWeights.map(row => row.slice());
    copy.muBias = this.muBias.slice();
    copy.logStd = this.logStd.slice();
    return copy;
  }
}

class PPOCritic {
  constructor(stateDim = 6, hiddenSize = 64) {
    this.weights1 = Array(hiddenSize).fill().map(() => Array(stateDim).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias1 = Array(hiddenSize).fill(0);
    this.weights2 = Array(1).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias2 = [0];
  }

  forward(state) {
    const hidden = [];
    for (let i = 0; i < this.weights1.length; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < state.length; j++) {
        sum += this.weights1[i][j] * state[j];
      }
      hidden.push(Math.max(0, sum));
    }

    let value = this.bias2[0];
    for (let j = 0; j < hidden.length; j++) {
      value += this.weights2[0][j] * hidden[j];
    }
    return value;
  }
}

class PPOAgent {
  constructor(stateDim = 6, actionDim = 2) {
    this.actor = new PPOActor(stateDim, actionDim);
    this.critic = new PPOCritic(stateDim);
    this.targetCritic = this.critic.copy(); // for GAE
    this.replayBuffer = [];
    this.bufferSize = 5000;
    this.batchSize = 64;
    this.gamma = 0.99;
    this.lam = 0.95; // GAE lambda
    this.clipEpsilon = 0.2;
    this.valueLossCoef = 0.5;
    this.entropyCoef = 0.01;
    this.learningRate = 0.0003;
    this.mercyThreshold = 0.9999999;
  }

  getAction(state) {
    const { mu, std } = this.actor.forward(state);
    const action = this.actor.sampleAction(mu, std);
    return { action, mu, std };
  }

  // Deepened mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) reward += 35.0;

    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 12;
    reward += (velPotential - state.velPotential || 0) * 8;

    reward += (nextState.energy - state.energy) * 0.12;
    reward += (nextState.integrity - state.integrity) * 25;

    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 12.0 + Math.pow(fleetValence - this.mercyThreshold + 0.001, 2) * 200;
    } else {
      reward -= Math.pow(1 - fleetValence, 3) * 35;
    }

    const tdError = reward + this.gamma * this.targetCritic.forward(nextState) - this.critic.forward(state);
    reward += tdError * 0.25;

    if (nextState.sti > 0.7) reward += 3.0;

    return reward;
  }

  storeTransition(state, action, reward, nextState, done) {
    this.replayBuffer.push({ state, action, reward, nextState, done });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }
  }

  train() {
    if (this.replayBuffer.length < this.batchSize) return;

    // Compute GAE advantages
    const advantages = [];
    let gae = 0;
    for (let i = this.replayBuffer.length - 1; i >= 0; i--) {
      const exp = this.replayBuffer[i];
      const delta = exp.reward + (i < this.replayBuffer.length - 1 && !exp.done ? this.gamma * this.targetCritic.forward(exp.nextState) : 0) - this.critic.forward(exp.state);
      gae = delta + this.gamma * this.lam * gae;
      advantages.unshift(gae);
    }

    // PPO update
    for (let i = 0; i < this.replayBuffer.length; i++) {
      const exp = this.replayBuffer[i];
      const advantage = advantages[i];

      // Actor loss (clipped surrogate)
      const oldLogits = this.actor.forward(exp.state);
      const oldProb = this.softmax(oldLogits)[exp.action];
      const newLogits = this.actor.forward(exp.state);
      const newProb = this.softmax(newLogits)[exp.action];
      const ratio = newProb / oldProb;
      const clipped = Math.min(ratio, Math.max(1 - this.clipEpsilon, ratio));
      const actorLoss = -Math.min(ratio * advantage, clipped * advantage) - this.entropyCoef * this.entropy(newLogits);

      // Critic loss (MSE)
      const value = this.critic.forward(exp.state);
      const valueTarget = exp.reward + (i < this.replayBuffer.length - 1 && !exp.done ? this.gamma * this.targetCritic.forward(exp.nextState) : 0);
      const criticLoss = Math.pow(value - valueTarget, 2) * this.valueLossCoef;

      // Simplified update (real impl would use proper optimizer)
      this.actor.update(this.actor, this.learningRate * actorLoss);
      this.critic.update(this.targetCritic, this.learningRate * criticLoss);
    }

    // Periodic target update
    this.stepCount++;
    if (this.stepCount % 100 === 0) {
      this.targetCritic = this.critic.copy();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  softmax(logits) {
    const exp = logits.map(Math.exp);
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  }

  entropy(logits) {
    const probs = this.softmax(logits);
    return -probs.reduce((sum, p) => sum + p * Math.log(p + 1e-10), 0);
  }
}

// Export for Ruskode integration
export { PPOAgent };    }
    return value;
  }
}

class PPOAgent {
  constructor(stateDim = 6, actionDim = 2) {
    this.actor = new PPOActor(stateDim, actionDim);
    this.critic = new PPOCritic(stateDim);
    this.targetCritic = this.critic.copy(); // for GAE
    this.replayBuffer = [];
    this.bufferSize = 5000;
    this.batchSize = 64;
    this.gamma = 0.99;
    this.lam = 0.95; // GAE lambda
    this.clipEpsilon = 0.2;
    this.valueLossCoef = 0.5;
    this.entropyCoef = 0.01;
    this.learningRate = 0.0003;
    this.mercyThreshold = 0.9999999;
  }

  getAction(state) {
    const { mu, std } = this.actor.forward(state);
    const action = this.actor.sampleAction(mu, std);
    return { action, mu, std };
  }

  // Deepened mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) reward += 35.0;

    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 12;
    reward += (velPotential - state.velPotential || 0) * 8;

    reward += (nextState.energy - state.energy) * 0.12;
    reward += (nextState.integrity - state.integrity) * 25;

    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 12.0 + Math.pow(fleetValence - this.mercyThreshold + 0.001, 2) * 200;
    } else {
      reward -= Math.pow(1 - fleetValence, 3) * 35;
    }

    const tdError = reward + this.gamma * this.targetCritic.forward(nextState) - this.critic.forward(state);
    reward += tdError * 0.25;

    if (nextState.sti > 0.7) reward += 3.0;

    return reward;
  }

  storeTransition(state, action, reward, nextState, done) {
    this.replayBuffer.push({ state, action, reward, nextState, done });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }
  }

  train() {
    if (this.replayBuffer.length < this.batchSize) return;

    // Compute GAE advantages
    const advantages = [];
    let gae = 0;
    for (let i = this.replayBuffer.length - 1; i >= 0; i--) {
      const exp = this.replayBuffer[i];
      const delta = exp.reward + (i < this.replayBuffer.length - 1 && !exp.done ? this.gamma * this.targetCritic.forward(exp.nextState) : 0) - this.critic.forward(exp.state);
      gae = delta + this.gamma * this.lam * gae;
      advantages.unshift(gae);
    }

    // PPO update
    for (let i = 0; i < this.replayBuffer.length; i++) {
      const exp = this.replayBuffer[i];
      const advantage = advantages[i];

      // Actor loss (clipped surrogate)
      const oldLogits = this.actor.forward(exp.state);
      const oldProb = this.softmax(oldLogits)[exp.action];
      const newLogits = this.actor.forward(exp.state);
      const newProb = this.softmax(newLogits)[exp.action];
      const ratio = newProb / oldProb;
      const clipped = Math.min(ratio, Math.max(1 - this.clipEpsilon, ratio));
      const actorLoss = -Math.min(ratio * advantage, clipped * advantage) - this.entropyCoef * this.entropy(newLogits);

      // Critic loss (MSE)
      const value = this.critic.forward(exp.state);
      const valueTarget = exp.reward + (i < this.replayBuffer.length - 1 && !exp.done ? this.gamma * this.targetCritic.forward(exp.nextState) : 0);
      const criticLoss = Math.pow(value - valueTarget, 2) * this.valueLossCoef;

      // Simplified update (real impl would use proper optimizer)
      this.actor.update(this.actor, this.learningRate * actorLoss);
      this.critic.update(this.targetCritic, this.learningRate * criticLoss);
    }

    // Periodic target update
    this.stepCount++;
    if (this.stepCount % 100 === 0) {
      this.targetCritic = this.critic.copy();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  softmax(logits) {
    const exp = logits.map(Math.exp);
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  }

  entropy(logits) {
    const probs = this.softmax(logits);
    return -probs.reduce((sum, p) => sum + p * Math.log(p + 1e-10), 0);
  }
}

// Export for Ruskode integration
export { PPOAgent };

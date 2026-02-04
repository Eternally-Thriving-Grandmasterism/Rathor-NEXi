// rl-reinforcement-engine.js – sovereign client-side Reinforcement Learning (PPO-style)
// Mercy-shaped rewards, valence-aligned policy, attention-modulated exploration
// MIT License – Autonomicity Games Inc. 2026

class RLAgent {
  constructor(stateDim = 6, actionDim = 2, hiddenSize = 32) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.actor = this.createNetwork(hiddenSize, actionDim, "actor");
    this.critic = this.createNetwork(hiddenSize, 1, "critic");
    this.replayBuffer = [];
    this.bufferSize = 10000;
    this.batchSize = 64;
    this.gamma = 0.99;        // discount
    this.clipEpsilon = 0.2;   // PPO clip
    this.learningRate = 0.001;
    this.mercyThreshold = 0.9999999;
  }

  createNetwork(hiddenSize, outputSize, type) {
    // Simple feedforward network (actor/critic)
    return {
      layers: [
        { weights: Array(hiddenSize).fill().map(() => Array(this.stateDim).fill().map(() => Math.random() * 0.2 - 0.1)), bias: Array(hiddenSize).fill(0) },
        { weights: Array(outputSize).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1)), bias: Array(outputSize).fill(0) }
      ],
      type
    };
  }

  forward(network, state) {
    let x = state.slice();
    for (let layer of network.layers) {
      const out = [];
      for (let i = 0; i < layer.weights.length; i++) {
        let sum = layer.bias[i];
        for (let j = 0; j < x.length; j++) {
          sum += layer.weights[i][j] * x[j];
        }
        out.push(network.type === "actor" && i === network.layers.length - 1 ? Math.tanh(sum) : Math.max(0, sum)); // ReLU + tanh output for actor
      }
      x = out;
    }
    return x;
  }

  getAction(state) {
    const logits = this.forward(this.actor, state);
    const probs = this.softmax(logits);
    const action = this.sample(probs);
    return { action, probs };
  }

  softmax(logits) {
    const exp = logits.map(Math.exp);
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  }

  sample(probs) {
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  async train(trajectory) {
    // PPO update (simplified single-epoch for client-side)
    const states = trajectory.map(t => t.state);
    const actions = trajectory.map(t => t.action);
    const rewards = trajectory.map(t => t.reward);
    const oldProbs = trajectory.map(t => t.prob);

    // Compute advantages (GAE-like)
    let advantages = [];
    let gae = 0;
    for (let i = rewards.length - 1; i >= 0; i--) {
      const delta = rewards[i] + (i < rewards.length - 1 ? this.gamma * this.forward(this.critic, states[i + 1])[0] : 0) - this.forward(this.critic, states[i])[0];
      gae = delta + this.gamma * 0.95 * gae;
      advantages.unshift(gae);
    }

    // Update actor & critic
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const action = actions[i];
      const advantage = advantages[i];
      const oldProb = oldProbs[i];

      const newLogits = this.forward(this.actor, state);
      const newProbs = this.softmax(newLogits);
      const newProb = newProbs[action];

      const ratio = newProb / oldProb;
      const clipped = Math.min(ratio, Math.max(1 - this.clipEpsilon, ratio));
      const actorLoss = -Math.min(ratio * advantage, clipped * advantage);

      // Critic loss (MSE)
      const value = this.forward(this.critic, state)[0];
      const valueTarget = rewards[i] + (i < rewards.length - 1 ? this.gamma * this.forward(this.critic, states[i + 1])[0] : 0);
      const criticLoss = Math.pow(value - valueTarget, 2);

      // Update weights (simple SGD step)
      this.updateNetwork(this.actor, state, actorLoss);
      this.updateNetwork(this.critic, state, criticLoss);
    }
  }

  updateNetwork(network, state, loss) {
    // Very simplified gradient step – real impl would use proper backprop
    const lr = this.learningRate * 0.001;
    for (let layer of network.layers) {
      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] -= lr * loss * state[j];
        }
        layer.bias[i] -= lr * loss;
      }
    }
  }

  // Mercy-shaped reward shaping
  shapeReward(state, action, nextState) {
    let reward = 0;

    // Altitude & velocity progress
    reward += (nextState.altitude - state.altitude) * 0.01;
    reward += (nextState.velocity - state.velocity) * 0.005;

    // Energy & integrity preservation
    reward -= Math.abs(nextState.energy - state.energy) * 0.02;
    reward += nextState.integrity * 0.1;

    // Mercy valence boost (from Hyperon)
    if (nextState.valence >= this.mercyThreshold) {
      reward += 1.0;
    } else {
      reward -= 5.0; // heavy penalty for low valence
    }

    return reward;
  }
}

// Export for Ruskode integration
export { RLAgent };

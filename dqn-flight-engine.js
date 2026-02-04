// dqn-flight-engine.js – sovereign client-side Deep Q-Networks for AlphaProMega Air flight control v3
// Prioritized Experience Replay (PER), mercy-shaped rewards, target network, double Q
// MIT License – Autonomicity Games Inc. 2026

class SumTree {
  constructor(capacity) {
    this.capacity = capacity;
    this.tree = new Array(2 * capacity).fill(0);
    this.data = new Array(capacity).fill(null);
    this.write = 0;
    this.n_entries = 0;
  }

  _propagate(idx, change) {
    let parent = (idx - 1) >> 1;
    this.tree[parent] += change;
    if (parent !== 0) this._propagate(parent, change);
  }

  _retrieve(idx, s) {
    const left = (idx << 1) + 1;
    const right = left + 1;

    if (left >= this.tree.length) return idx - this.capacity;

    if (s <= this.tree[left]) return this._retrieve(left, s);
    else return this._retrieve(right, s - this.tree[left]);
  }

  total() {
    return this.tree[0];
  }

  add(p, data) {
    const idx = this.write + this.capacity;
    this.data[this.write] = data;
    this.update(idx, p);
    this.write = (this.write + 1) % this.capacity;
    if (this.n_entries < this.capacity) this.n_entries++;
  }

  update(idx, p) {
    const change = p - this.tree[idx];
    this.tree[idx] = p;
    this._propagate(idx, change);
  }

  get(s) {
    const idx = this._retrieve(0, s);
    const dataIdx = idx - this.capacity;
    return { priority: this.tree[idx], data: this.data[dataIdx], index: dataIdx };
  }
}

class DQNController {
  constructor() {
    this.inputSize = 6;
    this.actionSize = 7;
    this.qNetwork = new DQNNetwork(this.inputSize, 64, this.actionSize);
    this.targetNetwork = this.qNetwork.copy();
    this.sumTree = new SumTree(10000);
    this.maxPriority = 1.0;
    this.alpha = 0.6;          // prioritization exponent
    this.beta = 0.4;           // importance sampling exponent
    this.betaAnneal = 0.001;   // beta increase per step
    this.maxBeta = 1.0;
    this.epsilon = 0.4;
    this.epsilonDecay = 0.995;
    this.minEpsilon = 0.01;
    this.gamma = 0.99;
    this.learningRate = 0.001;
    this.targetUpdateFreq = 200;
    this.stepCount = 0;
    this.mercyThreshold = 0.9999999;
  }

  chooseAction(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize);
    }
    const qValues = this.qNetwork.forward(state);
    return qValues.indexOf(Math.max(...qValues));
  }

  // Deepened mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) reward += 30.0;

    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 10;
    reward += (velPotential - state.velPotential || 0) * 6;

    reward += (nextState.energy - state.energy) * 0.1;
    reward += (nextState.integrity - state.integrity) * 20;

    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 10.0 + Math.pow(fleetValence - this.mercyThreshold + 0.001, 2) * 150;
    } else {
      reward -= Math.pow(1 - fleetValence, 3) * 30;
    }

    const tdError = reward + this.gamma * Math.max(...this.targetNetwork.forward(nextState)) - Math.max(...this.qNetwork.forward(state));
    reward += tdError * 0.2;

    if (nextState.sti > 0.7) reward += 2.5;

    return reward;
  }

  storeTransition(state, action, reward, nextState) {
    const tdError = Math.abs(reward + this.gamma * Math.max(...this.targetNetwork.forward(nextState)) - Math.max(...this.qNetwork.forward(state)));
    const priority = Math.pow(tdError + 1e-5, this.alpha); // +epsilon to avoid zero priority
    this.sumTree.add(priority, { state, action, reward, nextState });
  }

  train() {
    if (this.sumTree.n_entries < this.batchSize) return;

    const batch = [];
    const priorities = [];
    const indices = [];

    for (let i = 0; i < this.batchSize; i++) {
      const s = Math.random() * this.sumTree.total();
      const { priority, data, index } = this.sumTree.get(s);
      batch.push(data);
      priorities.push(priority);
      indices.push(index);
    }

    const weights = priorities.map(p => Math.pow(this.sumTree.total() * p / this.sumTree.total(), -this.beta));
    const maxWeight = Math.max(...weights);
    const normalizedWeights = weights.map(w => w / maxWeight);

    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      const stateKey = JSON.stringify(exp.state);
      const nextKey = JSON.stringify(exp.nextState);

      const currentQ = this.qNetwork.forward(exp.state)[exp.action];
      const nextQ = this.targetNetwork.forward(exp.nextState);
      const maxNextQ = Math.max(...nextQ);
      const targetQ = exp.reward + this.gamma * maxNextQ;

      const error = targetQ - currentQ;
      // Simplified update
      this.qNetwork.update(this.targetNetwork, this.learningRate * error * normalizedWeights[i]);
    }

    // Anneal beta
    this.beta = Math.min(this.maxBeta, this.beta + this.betaAnneal);

    // Periodic target update
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

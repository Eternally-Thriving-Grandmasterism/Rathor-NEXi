// src/core/mcts-reinforcement-hybrid.ts – MCTS + Reinforcement Learning Hybrid v1.0
// AlphaZero-style self-play + neural improvement + valence-shaped rewards
// MIT License – Autonomicity Games Inc. 2026

import MCTS, { NeuralNetwork } from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const SELF_PLAY_EPISODES_PER_UPDATE = 50;
const TRAINING_BATCH_SIZE = 128;
const LEARNING_RATE = 1e-4;
const VALUE_LOSS_WEIGHT = 0.5;
const POLICY_LOSS_WEIGHT = 0.5;
const VALENCE_REWARD_BONUS = 3.0;      // strong incentive for high-valence outcomes
const MAX_SELF_PLAY_ITERATIONS = 1000;

interface ReplayBufferEntry {
  state: any;
  policy: Map<string, number>;
  value: number;
  reward: number;
  valenceAtStep: number;
}

const replayBuffer: ReplayBufferEntry[] = [];

export class MCTSRLOptimizer {
  private mcts: MCTS;
  private neuralNet: NeuralNetwork & { train: (batch: ReplayBufferEntry[]) => Promise<void> };

  constructor(initialState: any, initialActions: string[], neuralNet: NeuralNetwork & { train: (batch: ReplayBufferEntry[]) => Promise<void> }) {
    this.neuralNet = neuralNet;
    this.mcts = new MCTS(initialState, initialActions, neuralNet);
  }

  /**
   * Run self-play episode → collect trajectories → store in replay buffer
   */
  async selfPlayEpisode(): Promise<void> {
    let state = this.mcts.root.state;
    const trajectory: ReplayBufferEntry[] = [];
    let totalReward = 0;

    while (!this.mcts.isTerminal(state)) {
      const { bestAction, policy } = await this.mcts.search();

      const nextState = this.mcts.applyAction(state, bestAction);
      const stepValence = currentValence.get();

      // Simulate reward (replace with real environment)
      const reward = this.computeReward(nextState, stepValence);

      totalReward += reward;

      trajectory.push({
        state,
        policy,
        value: 0,           // filled later via bootstrapping
        reward,
        valenceAtStep: stepValence
      });

      state = nextState;
    }

    // Bootstrap values with Bellman backup + valence bonus
    let bootstrapValue = this.evaluateTerminal(state);
    for (let i = trajectory.length - 1; i >= 0; i--) {
      const entry = trajectory[i];
      bootstrapValue = entry.reward + 0.99 * bootstrapValue; // discount 0.99
      bootstrapValue += VALENCE_REWARD_BONUS * entry.valenceAtStep;
      entry.value = bootstrapValue;
    }

    replayBuffer.push(...trajectory);

    // Keep buffer bounded
    if (replayBuffer.length > 100000) {
      replayBuffer.splice(0, replayBuffer.length - 100000);
    }

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log(`[MCTSRLOptimizer] Episode complete – length: ${trajectory.length}, total reward: ${totalReward.toFixed(3)}`);
  }

  /**
   * Train neural network from replay buffer
   */
  async trainStep(): Promise<void> {
    if (replayBuffer.length < TRAINING_BATCH_SIZE) return;

    const batch = this.sampleBatch(TRAINING_BATCH_SIZE);
    await this.neuralNet.train(batch);

    console.log(`[MCTSRLOptimizer] Training step complete – batch size: ${batch.length}`);
  }

  private sampleBatch(size: number): ReplayBufferEntry[] {
    const indices = new Set<number>();
    while (indices.size < size && indices.size < replayBuffer.length) {
      indices.add(Math.floor(Math.random() * replayBuffer.length));
    }
    return Array.from(indices).map(i => replayBuffer[i]);
  }

  private computeReward(state: any, valence: number): number {
    // Example reward shaping – customize per domain
    const baseReward = state.isWinning ? 1 : state.isLosing ? -1 : 0;
    return baseReward + valence * 0.8; // valence bonus dominates
  }

  private evaluateTerminal(state: any): number {
    return state.finalValence || currentValence.get();
  }

  /**
   * Full self-play + training loop
   */
  async runTrainingLoop(episodes: number = 10) {
    const actionName = 'Run MCTS-RL training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[MCTSRLOptimizer] Self-play episode \( {e+1}/ \){episodes}`);
      await this.selfPlayEpisode();

      if ((e + 1) % 5 === 0) {
        await this.trainStep();
      }
    }

    console.log("[MCTSRLOptimizer] Training loop complete");
  }
}

// Example usage – replace MockNeuralNet with real WebLLM-backed policy/value net
class MockNeuralNet {
  async predict(state: any) {
    return {
      policy: new Map([
        ['propose-alliance', 0.45],
        ['offer-resources', 0.25],
        ['request-aid', 0.15],
        ['reject', 0.08],
        ['wait', 0.07]
      ]),
      value: currentValence.get()
    };
  }

  async train(batch: ReplayBufferEntry[]) {
    console.log(`[MockNeuralNet] Training on ${batch.length} samples`);
    // Real impl: backprop policy & value heads
  }
}

export default MCTSRLOptimizer;

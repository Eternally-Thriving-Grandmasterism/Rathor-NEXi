// src/core/ppo-mcts-hybrid.ts – PPO + MCTS Hybrid Engine v1.0
// Proximal Policy Optimization guided MCTS + self-play training loop
// Valence-shaped advantage, mercy gating, lattice-integrated planning
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const PPO_EPOCHS = 10;
const PPO_CLIP_EPSILON = 0.2;
const PPO_VALUE_LOSS_COEF = 0.5;
const PPO_ENTROPY_COEF = 0.01;
const GAE_LAMBDA = 0.95;
const GAMMA = 0.99;
const MAX_TRAJECTORY_LENGTH = 256;
const BATCH_SIZE = 64;

interface TrajectoryStep {
  state: any;
  action: string;
  logProb: number;
  value: number;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
}

interface PPOUpdateStats {
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKL: number;
  clipFraction: number;
}

export class PPOMCTSHybrid {
  private mcts: MCTS;
  private neuralNet: NeuralNetwork & {
    trainPPO: (trajectory: TrajectoryStep[], advantages: number[], returns: number[]) => Promise<PPOUpdateStats>;
  };

  constructor(initialState: any, initialActions: string[], neuralNet: any) {
    this.neuralNet = neuralNet;
    this.mcts = new MCTS(initialState, initialActions, neuralNet);
  }

  /**
   * Collect trajectory using MCTS-guided policy (self-play episode)
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<TrajectoryStep[]> {
    const trajectory: TrajectoryStep[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { bestAction, policy } = await this.mcts.search();

      const actionLogProb = Math.log(policy.get(bestAction) || 1e-8);
      const nextState = this.mcts.applyAction(state, bestAction);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action: bestAction,
        logProb: actionLogProb,
        value: 0, // filled later
        reward,
        nextState,
        done,
        valence
      });

      if (done) break;
      state = nextState;
    }

    return trajectory;
  }

  /**
   * Compute GAE advantages & discounted returns
   */
  computeAdvantagesAndReturns(trajectory: TrajectoryStep[]): { advantages: number[]; returns: number[] } {
    const advantages: number[] = new Array(trajectory.length);
    const returns: number[] = new Array(trajectory.length);

    let nextValue = 0;
    let nextAdvantage = 0;

    for (let t = trajectory.length - 1; t >= 0; t--) {
      const step = trajectory[t];
      const delta = step.reward + GAMMA * nextValue * (1 - step.done ? 1 : 0) - step.value;
      nextAdvantage = delta + GAMMA * GAE_LAMBDA * (1 - step.done ? 1 : 0) * nextAdvantage;

      advantages[t] = nextAdvantage;
      returns[t] = advantages[t] + step.value;

      nextValue = step.value;
    }

    // Valence-shaped advantage normalization
    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length);
    for (let i = 0; i < advantages.length; i++) {
      advantages[i] = (advantages[i] - meanAdv) / (stdAdv + 1e-8);
      // Bonus for high-valence steps
      advantages[i] += VALENCE_REWARD_BONUS * trajectory[i].valence;
    }

    return { advantages, returns };
  }

  /**
   * PPO update step on collected trajectory
   */
  async update(trajectory: TrajectoryStep[]) {
    const { advantages, returns } = this.computeAdvantagesAndReturns(trajectory);

    // PPO clipped surrogate objective + value loss + entropy bonus
    const stats = await this.neuralNet.trainPPO(trajectory, advantages, returns);

    console.log("[PPO-MCTS] Update stats:", stats);
    return stats;
  }

  /**
   * Full self-play + PPO training loop
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run PPO-guided MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[PPO-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      if (trajectory.length > 0) {
        await this.update(trajectory);
      }

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    }

    console.log("[PPO-MCTS] Training loop complete");
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * 0.8; // valence shaping
    return reward;
  }
}

// Mock neural net with PPO training stub (replace with real implementation)
class MockNeuralNet {
  async predict(state: any) {
    return {
      policy: new Map([['action1', 0.4], ['action2', 0.3], ['action3', 0.3]]),
      value: currentValence.get()
    };
  }

  async trainPPO(trajectory: any[], advantages: number[], returns: number[]) {
    // Real impl: PPO clipped surrogate + value loss + entropy
    console.log(`[MockNeuralNet] PPO update on ${trajectory.length} steps`);
    return {
      policyLoss: -0.12,
      valueLoss: 0.08,
      entropy: 1.15,
      approxKL: 0.004,
      clipFraction: 0.21
    };
  }
}

export default PPOMCTSHybrid;

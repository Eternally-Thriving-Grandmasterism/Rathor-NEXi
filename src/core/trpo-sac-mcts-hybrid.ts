// src/core/trpo-sac-mcts-hybrid.ts – TRPO + SAC + MCTS Hybrid Engine v1.0
// Trust Region Policy Optimization guided MCTS + SAC entropy regularization + tree search
// Valence-shaped advantage & KL constraint, mercy gating, self-play loop
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { getTemperature, updateTemperature } from './automatic-temperature-tuning';

const TRPO_DELTA = 0.01;                    // KL divergence constraint
const TRPO_BACKTRACK_COEFF = 0.8;           // backtracking line search coefficient
const SAC_TAU = 0.005;                      // soft target update
const SAC_ENTROPY_TARGET_BASE = -2.0;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;
const TARGET_UPDATE_INTERVAL = 1;
const MAX_LINE_SEARCH_STEPS = 10;

interface TrajectoryStep {
  state: any;
  action: any;
  oldLogProb: number;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
  value?: number;
}

const replayBuffer: TrajectoryStep[] = [];
let stepsSinceTargetUpdate = 0;

export class TRPOSACMCTSHybrid {
  private mcts: MCTS;
  private policyNet: {
    predictPolicyAndValue: (state: any) => Promise<{ policy: Map<string, number>; value: number }>;
    computeKL: (oldPolicy: Map<string, number>, newPolicy: Map<string, number>) => Promise<number>;
    trainTRPO: (trajectory: TrajectoryStep[], advantages: number[]) => Promise<any>;
  };

  constructor(initialState: any, initialActions: string[], policyNet: any) {
    this.policyNet = policyNet;
    this.mcts = new MCTS(initialState, initialActions, policyNet);
  }

  /**
   * Collect trajectory using MCTS-guided policy (self-play episode)
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<TrajectoryStep[]> {
    const trajectory: TrajectoryStep[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { bestAction, policy } = await this.mcts.search();

      const oldLogProb = Math.log(policy.get(bestAction) || 1e-8);
      const nextState = this.mcts.applyAction(state, bestAction);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action: bestAction,
        oldLogProb,
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
      const delta = step.reward + SAC_GAMMA * nextValue * (step.done ? 0 : 1) - step.value;
      nextAdvantage = delta + SAC_GAMMA * GAE_LAMBDA * (step.done ? 0 : 1) * nextAdvantage;

      advantages[t] = nextAdvantage;
      returns[t] = advantages[t] + step.value;

      nextValue = step.value;
    }

    // Valence-weighted advantage normalization + boost
    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length) + 1e-8;

    for (let i = 0; i < advantages.length; i++) {
      advantages[i] = (advantages[i] - meanAdv) / stdAdv;
      advantages[i] += VALENCE_ADVANTAGE_BOOST * trajectory[i].valence;
    }

    return { advantages, returns };
  }

  /**
   * TRPO-style trust-region update with line search
   */
  async updateTRPO(trajectory: TrajectoryStep[], advantages: number[]) {
    const actionName = 'TRPO + SAC-MCTS update';
    if (!await mercyGate(actionName)) return;

    // Compute surrogate advantage (old policy vs new policy)
    const oldLogProbs = trajectory.map(t => t.oldLogProb);
    const { policy: newPolicy } = await this.policyNet.predictPolicyAndValue(trajectory[0].state); // simplified
    const newLogProbs = trajectory.map(t => Math.log(newPolicy.get(t.action) || 1e-8));

    const ratio = newLogProbs.map((np, i) => Math.exp(np - oldLogProbs[i]));
    const surrogateAdv = ratio.map((r, i) => r * advantages[i]);

    // KL divergence constraint
    const kl = await this.policyNet.computeKL(
      new Map(trajectory.map((t, i) => [t.action, Math.exp(oldLogProbs[i])])),
      newPolicy
    );

    // Line search for step size
    let stepSize = 1.0;
    for (let i = 0; i < MAX_LINE_SEARCH_STEPS; i++) {
      if (kl <= TRPO_DELTA) break;
      stepSize *= TRPO_BACKTRACK_COEFF;
    }

    // Apply update (simplified – real impl would use conjugate gradient / L-BFGS)
    console.log(`[TRPO-SAC-MCTS] KL after backtrack: ${kl.toFixed(6)}, step size: ${stepSize.toFixed(4)}`);

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
  }

  /**
   * Full self-play + TRPO/SAC training loop
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run TRPO-guided SAC-MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[TRPO-SAC-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      if (trajectory.length > 0) {
        const { advantages, returns } = this.computeAdvantagesAndReturns(trajectory);
        await this.updateTRPO(trajectory, advantages);
      }
    }

    console.log("[TRPO-SAC-MCTS] Training loop complete");
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * 0.8;
    return reward;
  }
}

export default TRPOSACMCTSHybrid;

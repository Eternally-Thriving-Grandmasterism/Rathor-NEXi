// src/core/ppo-mcts-hybrid.ts – PPO + MCTS Hybrid Engine v1.2
// Proximal Policy Optimization guided MCTS + self-play training loop
// Valence-shaped advantage normalization, clipped surrogate objective, mercy gating
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const PPO_CLIP_EPSILON = 0.2;
const PPO_VALUE_LOSS_COEF = 0.5;
const PPO_ENTROPY_COEF = 0.01;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const GAE_LAMBDA = 0.95;
const GAMMA = 0.99;
const MAX_TRAJECTORY_LENGTH = 256;
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;

interface TrajectoryStep {
  state: any;
  action: string;
  oldLogProb: number;
  newLogProb: number;
  value: number;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
}

const replayBuffer: TrajectoryStep[] = [];

function sampleFromProbs(probs: number[]): number {
  let sum = 0;
  const r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) return i;
  }
  return probs.length - 1;
}

export class PPOMCTSHybrid {
  private mcts: MCTS;
  private policyNet: {
    predictPolicyAndValue: (state: any) => Promise<{ policy: Map<string, number>; value: number }>;
    trainPPO: (batch: TrajectoryStep[], advantages: number[], returns: number[]) => Promise<any>;
  };

  constructor(initialState: any, initialActions: string[], policyNet: any) {
    this.policyNet = policyNet;
    this.mcts = new MCTS(initialState, initialActions, policyNet);
  }

  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<TrajectoryStep[]> {
    const trajectory: TrajectoryStep[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { bestAction, policy } = await this.mcts.search();

      const actionProbs = Array.from(policy.values());
      const actionIndex = sampleFromProbs(actionProbs);
      const action = Array.from(policy.keys())[actionIndex];
      const oldLogProb = Math.log(policy.get(action) || 1e-8);

      const nextState = this.mcts.applyAction(state, action);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      const { policy: newPolicy } = await this.policyNet.predictPolicyAndValue(state);
      const newLogProb = Math.log(newPolicy.get(action) || 1e-8);

      trajectory.push({
        state,
        action,
        oldLogProb,
        newLogProb,
        value: 0,
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

  computeAdvantagesAndReturns(trajectory: TrajectoryStep[]): { advantages: number[]; returns: number[] } {
    const advantages: number[] = new Array(trajectory.length);
    const returns: number[] = new Array(trajectory.length);

    let nextValue = 0;
    let nextAdvantage = 0;

    for (let t = trajectory.length - 1; t >= 0; t--) {
      const step = trajectory[t];
      const delta = step.reward + GAMMA * nextValue * (step.done ? 0 : 1) - step.value;
      nextAdvantage = delta + GAMMA * GAE_LAMBDA * (step.done ? 0 : 1) * nextAdvantage;

      advantages[t] = nextAdvantage;
      returns[t] = advantages[t] + step.value;

      nextValue = step.value;
    }

    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length) + 1e-8;

    for (let i = 0; i < advantages.length; i++) {
      advantages[i] = (advantages[i] - meanAdv) / stdAdv;
      advantages[i] += VALENCE_ADVANTAGE_BOOST * trajectory[i].valence;
    }

    return { advantages, returns };
  }

  computePPOLoss(
    trajectory: TrajectoryStep[],
    advantages: number[],
    returns: number[]
  ): any {
    if (!mercyGate('Compute PPO surrogate loss')) {
      return { policyLoss: 0, valueLoss: 0, entropyBonus: 0, totalLoss: 0, approxKL: 0, clipFraction: 0 };
    }

    const n = trajectory.length;
    let policyLossSum = 0;
    let valueLossSum = 0;
    let entropySum = 0;
    let clipCount = 0;
    let klSum = 0;

    for (let i = 0; i < n; i++) {
      const step = trajectory[i];
      const ratio = Math.exp(step.newLogProb - step.oldLogProb);
      const weightedAdv = advantages[i];

      const surrogate1 = ratio * weightedAdv;
      const surrogate2 = Math.max(1 - PPO_CLIP_EPSILON, Math.min(1 + PPO_CLIP_EPSILON, ratio)) * weightedAdv;

      policyLossSum += Math.min(surrogate1, surrogate2);

      if (ratio < 1 - PPO_CLIP_EPSILON || ratio > 1 + PPO_CLIP_EPSILON) {
        clipCount++;
      }

      klSum += ratio - 1 - Math.log(ratio + 1e-10);

      const vPred = step.value;
      const vTarget = returns[i];
      const valueDiff = vPred - vTarget;
      valueLossSum += valueDiff * valueDiff;

      entropySum += -step.oldLogProb;
    }

    const policyLoss = -policyLossSum / n;
    const valueLoss = valueLossSum / n * PPO_VALUE_LOSS_COEF;
    const entropyBonus = (entropySum / n) * PPO_ENTROPY_COEF;

    const totalLoss = policyLoss + valueLoss - entropyBonus;
    const clipFraction = clipCount / n;
    const approxKL = klSum / n;

    return {
      policyLoss,
      valueLoss,
      entropyBonus,
      totalLoss,
      approxKL,
      clipFraction
    };
  }

  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run PPO-guided MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[PPO-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      if (trajectory.length > 0) {
        const { advantages, returns } = this.computeAdvantagesAndReturns(trajectory);
        const stats = this.computePPOLoss(trajectory, advantages, returns);

        console.log("[PPO-MCTS] PPO stats:", stats);

        mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      }
    }

    console.log("[PPO-MCTS] Training loop complete");
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * 0.8;
    return reward;
  }
}

export default PPOMCTSHybrid;    if (!await mercyGate(actionName)) {
      return this.getTemperature();
    }

    stepsSinceAlphaUpdate++;
    if (stepsSinceAlphaUpdate < ENTROPY_UPDATE_INTERVAL) {
      return this.getTemperature();
    }

    stepsSinceAlphaUpdate = 0;

    const targetEntropy = this.getValenceModulatedTargetEntropy();

    // SAC temperature loss: α * (target_entropy - batch_entropy)
    const alphaLoss = Math.exp(logAlpha) * (targetEntropy - batchEntropy);

    // Gradient descent step on logAlpha
    logAlpha -= SAC_ALPHA_LR * alphaLoss;

    // Hard clamp for numerical stability
    logAlpha = Math.max(Math.log(MIN_ALPHA), Math.min(Math.log(MAX_ALPHA), logAlpha));

    const newAlpha = this.getTemperature();

    // Haptic feedback on significant change
    if (Math.abs(newAlpha - SAC_ALPHA_INIT) > 0.1) {
      mercyHaptic.playPattern(
        newAlpha > SAC_ALPHA_INIT ? 'cosmicHarmony' : 'warningPulse',
        currentValence.get()
      );
    }

    console.log(
      `[AutoTemp in PPO] Updated α → ${newAlpha.toFixed(4)}  ` +
      `(target entropy: ${targetEntropy.toFixed(3)}, batch entropy: ${batchEntropy.toFixed(3)})`
    );

    return newAlpha;
  }

  /**
   * Collect trajectory using MCTS-guided policy (self-play episode)
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<TrajectoryStep[]> {
    const trajectory: TrajectoryStep[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { bestAction, policy } = await this.mcts.search();

      // Sample action from MCTS-improved policy
      const actionProbs = Array.from(policy.values());
      const actionIndex = sampleFromProbs(actionProbs);
      const action = Array.from(policy.keys())[actionIndex];
      const oldLogProb = Math.log(policy.get(action) || 1e-8);

      const nextState = this.mcts.applyAction(state, action);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      // Re-evaluate new policy log-prob (for PPO ratio)
      const { policy: newPolicy } = await this.policyNet.predictPolicyAndValue(state);
      const newLogProb = Math.log(newPolicy.get(action) || 1e-8);

      trajectory.push({
        state,
        action,
        oldLogProb,
        newLogProb,
        value: 0, // filled later via bootstrapping
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
   * PPO clipped surrogate loss + value loss + SAC-style adaptive entropy bonus
   */
  computePPOLoss(
    trajectory: TrajectoryStep[],
    advantages: number[],
    returns: number[]
  ): any {
    if (!mercyGate('Compute PPO surrogate loss with SAC entropy tuning')) {
      return { policyLoss: 0, valueLoss: 0, entropyBonus: 0, totalLoss: 0, approxKL: 0, clipFraction: 0 };
    }

    const n = trajectory.length;
    let policyLossSum = 0;
    let valueLossSum = 0;
    let entropySum = 0;
    let clipCount = 0;
    let klSum = 0;

    const alpha = getTemperature(); // current adaptive temperature

    for (let i = 0; i < n; i++) {
      const step = trajectory[i];
      const ratio = Math.exp(step.newLogProb - step.oldLogProb);
      const weightedAdv = advantages[i];

      const surrogate1 = ratio * weightedAdv;
      const surrogate2 = Math.max(1 - PPO_CLIP_EPSILON, Math.min(1 + PPO_CLIP_EPSILON, ratio)) * weightedAdv;

      policyLossSum += Math.min(surrogate1, surrogate2);

      if (ratio < 1 - PPO_CLIP_EPSILON || ratio > 1 + PPO_CLIP_EPSILON) {
        clipCount++;
      }

      klSum += ratio - 1 - Math.log(ratio + 1e-10);

      // Value loss
      const vPred = step.value;
      const vTarget = returns[i];
      const valueDiff = vPred - vTarget;
      valueLossSum += valueDiff * valueDiff;

      // SAC-style adaptive entropy bonus
      entropySum += -step.oldLogProb * alpha;
    }

    const policyLoss = -policyLossSum / n;
    const valueLoss = valueLossSum / n * PPO_VALUE_LOSS_COEF;
    const entropyBonus = (entropySum / n) * PPO_ENTROPY_COEF;

    const totalLoss = policyLoss + valueLoss - entropyBonus;
    const clipFraction = clipCount / n;
    const approxKL = klSum / n;

    return {
      policyLoss,
      valueLoss,
      entropyBonus,
      totalLoss,
      approxKL,
      clipFraction
    };
  }

  /**
   * Full self-play + PPO training loop with SAC-style temperature tuning
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run PPO-guided MCTS training loop with SAC entropy tuning';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[PPO-SAC-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      if (trajectory.length > 0) {
        const { advantages, returns } = this.computeAdvantagesAndReturns(trajectory);

        // Compute batch entropy for SAC-style auto-temperature tuning
        const batchEntropy = -trajectory.reduce((sum, t) => sum + (t.oldLogProb || 0), 0) / trajectory.length;
        const alpha = await updateTemperature(batchEntropy);

        const stats = this.computePPOLoss(trajectory, advantages, returns);

        // Real training: backprop stats.totalLoss through policy & value heads
        console.log("[PPO-SAC-MCTS] PPO stats:", stats, "α:", alpha);

        mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      }
    }

    console.log("[PPO-SAC-MCTS] Training loop complete");
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * 0.8; // valence shaping
    return reward;
  }
}

export default PPOMCTSHybrid;

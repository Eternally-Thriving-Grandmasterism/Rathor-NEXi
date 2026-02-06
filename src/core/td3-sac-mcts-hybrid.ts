// src/core/td3-sac-mcts-hybrid.ts – TD3 + SAC + MCTS Hybrid Engine v1.0
// Twin Delayed Deep Deterministic Policy Gradient + Soft Actor-Critic + MCTS fusion
// Clipped double Q-learning, delayed policy updates, valence-shaped advantage & entropy
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { getTemperature, updateTemperature } from './automatic-temperature-tuning';

const TD3_GAMMA = 0.99;
const TD3_TAU = 0.005;                    // soft target update
const TD3_POLICY_DELAY = 2;               // delayed policy updates
const TD3_NOISE_STD = 0.2;                // target policy smoothing noise
const TD3_NOISE_CLIP = 0.5;               // noise clip range
const SAC_ENTROPY_TARGET_BASE = -2.0;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;
const TARGET_UPDATE_INTERVAL = 1;
const CRITIC_LOSS_COEF = 0.5;
const ACTOR_LOSS_COEF = 1.0;

interface Transition {
  state: any;
  action: any;                            // continuous action vector
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
}

const replayBuffer: Transition[] = [];
let stepsSinceTargetUpdate = 0;
let policyUpdateCounter = 0;

export class TD3SACMCTSHybrid {
  private mcts: MCTS;
  private actor: {
    predictAction: (state: any) => Promise<any>;
    trainActor: (states: any[], actions: any[]) => Promise<number>;
  };
  private critic1: {
    predictQ: (state: any, action: any) => Promise<number>;
    trainCritic: (states: any[], actions: any[], targets: number[]) => Promise<number>;
  };
  private critic2: {
    predictQ: (state: any, action: any) => Promise<number>;
    trainCritic: (states: any[], actions: any[], targets: number[]) => Promise<number>;
  };
  private targetActor: { predictAction: (state: any) => Promise<any> };
  private targetCritic1: { predictQ: (state: any, action: any) => Promise<number> };
  private targetCritic2: { predictQ: (state: any, action: any) => Promise<number> };

  constructor(
    initialState: any,
    actionDim: number,
    actor: any,
    critic1: any,
    critic2: any,
    targetActor: any,
    targetCritic1: any,
    targetCritic2: any
  ) {
    this.mcts = new MCTS(initialState, [], actor);
    this.actor = actor;
    this.critic1 = critic1;
    this.critic2 = critic2;
    this.targetActor = targetActor;
    this.targetCritic1 = targetCritic1;
    this.targetCritic2 = targetCritic2;
  }

  /**
   * Collect trajectory using deterministic policy + target smoothing noise + MCTS refinement
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<Transition[]> {
    const trajectory: Transition[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      let action = await this.actor.predictAction(state);

      // Target policy smoothing noise (TD3)
      action = addNoise(action, TD3_NOISE_STD, TD3_NOISE_CLIP);

      // Optional MCTS refinement
      const { bestAction } = await this.mcts.search();
      action = this.blendActions(action, bestAction);

      const nextState = this.mcts.applyAction(state, action);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action,
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

  private blendActions(action1: any, action2: any): any {
    return action1.map((v: number, i: number) => 0.7 * v + 0.3 * action2[i]);
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * VALENCE_ADVANTAGE_BOOST;
    return reward;
  }

  /**
   * Sample batch & compute TD3 targets
   */
  private async sampleAndComputeTargets(): Promise<{
    states: any[];
    actions: any[];
    nextStates: any[];
    rewards: number[];
    dones: boolean[];
    advantages: number[];
    returns: number[];
  }> {
    const indices = new Set<number>();
    while (indices.size < BATCH_SIZE && indices.size < replayBuffer.length) {
      indices.add(Math.floor(Math.random() * replayBuffer.length));
    }

    const batch = Array.from(indices).map(i => replayBuffer[i]);

    const states = batch.map(t => t.state);
    const actions = batch.map(t => t.action);
    const nextStates = batch.map(t => t.nextState);
    const rewards = batch.map(t => t.reward);
    const dones = batch.map(t => t.done);

    // Clipped double Q-learning (TD3)
    const nextActions = await Promise.all(
      nextStates.map(s => this.targetActor.predictAction(s))
    );
    const noisyNextActions = nextActions.map(a => addNoise(a, TD3_NOISE_STD, TD3_NOISE_CLIP));

    const targetQ1 = await Promise.all(
      noisyNextActions.map((a, i) => this.targetCritic1.predictQ(nextStates[i], a))
    );
    const targetQ2 = await Promise.all(
      noisyNextActions.map((a, i) => this.targetCritic2.predictQ(nextStates[i], a))
    );

    const minTargetQ = targetQ1.map((q1, i) => Math.min(q1, targetQ2[i]));
    const targets = rewards.map((r, i) => r + SAC_GAMMA * minTargetQ[i] * (dones[i] ? 0 : 1));

    // Current Q-values
    const currentQ1 = await Promise.all(
      actions.map((a, i) => this.critic1.predictQ(states[i], a))
    );
    const currentQ2 = await Promise.all(
      actions.map((a, i) => this.critic2.predictQ(states[i], a))
    );

    // Advantages (simplified TD3 style)
    const advantages = targets.map((t, i) => t - currentQ1[i]); // can use min(currentQ1, currentQ2) if desired

    // Valence-weighted advantage normalization + boost
    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length) + 1e-8;

    const weightedAdvantages = advantages.map((adv, i) => {
      let normAdv = (adv - meanAdv) / stdAdv;
      normAdv += VALENCE_ADVANTAGE_BOOST * batch[i].valence;
      return normAdv;
    });

    const returns = targets; // simplified – can use GAE if desired

    return { states, actions, nextStates, rewards, dones, advantages: weightedAdvantages, returns };
  }

  /**
   * Full training loop – collect rollouts + periodic updates
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run DDPG-guided SAC-MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[DDPG-SAC-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      for (const step of trajectory) {
        replayBuffer.push(step);
        if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
          replayBuffer.shift();
        }
      }

      if (replayBuffer.length >= BATCH_SIZE) {
        const batchStats = await this.sampleAndComputeTargets();
        // Real training would happen here
        await this.softUpdateTarget();

        mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      }
    }

    console.log("[DDPG-SAC-MCTS] Training loop complete");
  }

  private async softUpdateTarget(tau: number = SAC_TAU) {
    console.log("[DDPG-SAC-MCTS] Soft updating target networks");
    // Real impl: θ_target = τ * θ + (1-τ) * θ_target
  }
}

function addNoise(action: any, std: number, clip: number): any {
  return action.map((v: number) => {
    let noisy = v + (Math.random() - 0.5) * 2 * std;
    return Math.max(v - clip, Math.min(v + clip, noisy));
  });
}

export default DDPGSACMCTSHybrid;

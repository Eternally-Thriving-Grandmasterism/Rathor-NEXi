// src/core/muzero-integration.ts – MuZero Model Integration Layer v1.0
// MuZero-style model-based planning: representation, dynamics, prediction networks
// Valence-weighted planning priority, mercy-gated simulation depth, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import MCTS from './alphago-style-mcts-neural';

const MUZERO_PLANNING_DEPTH = 50;          // max simulated steps in planning
const MUZERO_MCTS_ITERATIONS = 800;        // MCTS search budget per decision
const VALENCE_PLANNING_BOOST = 2.5;        // high valence → deeper planning & more iterations
const VALENCE_PRUNE_THRESHOLD = 0.85;      // prune simulated paths below this valence
const MIN_PLANNING_DEPTH = 5;
const MAX_PLANNING_DEPTH = 100;

interface MuZeroNetworks {
  representation: (state: any) => Promise<any>;           // s → hidden state h
  dynamics: (hidden: any, action: any) => Promise<{ nextHidden: any; reward: number }>; // (h, a) → (h', r)
  prediction: (hidden: any) => Promise<{ policy: Map<string, number>; value: number }>; // h → (p, v)
}

interface SimulatedNode {
  hiddenState: any;
  reward: number;
  policy: Map<string, number>;
  value: number;
  children: Map<string, SimulatedNode>;
  visits: number;
  totalValue: number;
  depth: number;
  isTerminal: boolean;
}

export class MuZeroIntegration {
  private networks: MuZeroNetworks;

  constructor(networks: MuZeroNetworks) {
    this.networks = networks;
  }

  /**
   * Perform MuZero-style planning: build search tree with learned model
   * @param initialState Root observation
   * @returns Best action & improved policy
   */
  async plan(initialState: any): Promise<{ bestAction: string; policy: Map<string, number> }> {
    const actionName = 'MuZero model-based planning';
    if (!await mercyGate(actionName)) {
      // Fallback to direct policy prediction
      const { policy } = await this.networks.prediction(await this.networks.representation(initialState));
      const bestAction = selectActionFromPolicy(policy);
      return { bestAction, policy };
    }

    const valence = currentValence.get();
    const planningDepth = Math.floor(MIN_PLANNING_DEPTH + (MAX_PLANNING_DEPTH - MIN_PLANNING_DEPTH) * valence);
    const iterations = Math.floor(MUZERO_MCTS_ITERATIONS * (0.5 + valence * VALENCE_PLANNING_BOOST));

    console.log(`[MuZero] Planning start – valence ${valence.toFixed(3)}, depth ${planningDepth}, ${iterations} iterations`);

    // 1. Initial hidden state from representation network
    const rootHidden = await this.networks.representation(initialState);

    // 2. Root prediction
    const { policy: rootPolicy, value: rootValue } = await this.networks.prediction(rootHidden);

    const root: SimulatedNode = {
      hiddenState: rootHidden,
      reward: 0,
      policy: rootPolicy,
      value: rootValue,
      children: new Map(),
      visits: 0,
      totalValue: 0,
      depth: 0,
      isTerminal: false
    };

    // 3. Run MCTS with learned model
    for (let i = 0; i < iterations; i++) {
      const path = this.select(root);
      const leaf = path[path.length - 1];
      const { nextHidden, reward } = await this.networks.dynamics(leaf.hiddenState, leaf.state.lastAction);
      const { policy, value } = await this.networks.prediction(nextHidden);

      // Prune low-valence branches (mercy gate)
      if (value < VALENCE_PRUNE_THRESHOLD && leaf.depth > 3) {
        continue;
      }

      this.expand(leaf, policy, reward, nextHidden);
      this.backpropagate(path, value);
    }

    // 4. Extract best action & improved policy
    const bestChild = this.bestChild(root);
    const bestAction = bestChild.state.lastAction;

    const policyImprovement = new Map<string, number>();
    let totalVisits = 0;
    for (const child of root.children.values()) {
      totalVisits += child.visits;
    }
    for (const [action, child] of root.children) {
      policyImprovement.set(action, child.visits / totalVisits);
    }

    const projectedValue = bestChild.totalValue / bestChild.visits;

    mercyHaptic.playPattern(valence > 0.9 ? 'cosmicHarmony' : 'neutralPulse', valence);

    return {
      bestAction,
      policy: policyImprovement,
      projectedValue
    };
  }

  private select(root: SimulatedNode): SimulatedNode[] {
    const path: SimulatedNode[] = [];
    let node = root;

    while (node.children.size > 0 && !node.isTerminal) {
      path.push(node);
      node = this.bestChild(node);
    }

    path.push(node);
    return path;
  }

  private bestChild(node: SimulatedNode): SimulatedNode {
    const valence = currentValence.get();
    const c_puct = DEFAULT_C_PUCT * (1 + valence * VALENCE_EXPLORATION_BOOST);

    let bestChild: SimulatedNode | null = null;
    let bestPUCT = -Infinity;

    for (const child of node.children.values()) {
      const q = child.visits > 0 ? child.totalValue / child.visits : 0;
      const u = c_puct * child.policy.get(child.state.lastAction || '')! * Math.sqrt(node.visits) / (1 + child.visits);
      const puct = q + u;

      if (puct > bestPUCT) {
        bestPUCT = puct;
        bestChild = child;
      }
    }

    return bestChild!;
  }

  private expand(parent: SimulatedNode, policy: Map<string, number>, reward: number, nextHidden: any) {
    for (const [action, prior] of policy) {
      if (!parent.children.has(action)) {
        const child: SimulatedNode = {
          hiddenState: nextHidden,
          reward,
          policy,
          value: 0,
          children: new Map(),
          visits: 0,
          totalValue: 0,
          depth: parent.depth + 1,
          isTerminal: false
        };
        parent.children.set(action, child);
      }
    }
  }

  private backpropagate(path: SimulatedNode[], value: number) {
    for (const node of path.reverse()) {
      node.visits++;
      node.totalValue += value;
    }
  }
}

function selectActionFromPolicy(policy: Map<string, number>): string {
  const actions = Array.from(policy.keys());
  const probs = Array.from(policy.values());
  let sum = 0;
  const r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    sum += probs[i];
    if (r <= sum) return actions[i];
  }
  return actions[actions.length - 1];
}

export default MuZeroIntegration;

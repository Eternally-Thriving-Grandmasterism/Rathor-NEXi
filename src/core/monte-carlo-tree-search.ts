// src/core/monte-carlo-tree-search.ts – Monte-Carlo Tree Search Engine v1.0
// Valence-weighted MCTS for negotiation, swarm bloom planning, gesture sequence optimization
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const MAX_ITERATIONS = 2000;
const EXPLORATION_CONSTANT = 1.414; // √2 – classic UCT
const VALENCE_BONUS_FACTOR = 2.5;   // extra exploration for high-valence branches
const MAX_DEPTH = 64;               // prevent infinite recursion in deep trees

interface Node {
  state: any;                       // generic state (game state, negotiation state, etc.)
  parent: Node | null;
  children: Node[];
  visits: number;
  value: number;                    // accumulated valence-weighted reward
  untriedActions: string[];         // available actions from this state
  isTerminal: boolean;
  depth: number;
}

export class MCTS {
  private root: Node;

  constructor(initialState: any, initialActions: string[]) {
    this.root = {
      state: initialState,
      parent: null,
      children: [],
      visits: 0,
      value: 0,
      untriedActions: [...initialActions],
      isTerminal: false,
      depth: 0
    };
  }

  /**
   * Run full MCTS search and return best action
   */
  async search(): Promise<string> {
    const actionName = 'Run Monte-Carlo Tree Search';
    if (!await mercyGate(actionName)) {
      return this.root.untriedActions[0] || 'none';
    }

    const valence = currentValence.get();
    const iterations = Math.floor(MAX_ITERATIONS * (0.5 + valence * 0.5)); // scale with valence

    console.log(`[MCTS] Starting search – valence ${valence.toFixed(3)}, ${iterations} iterations`);

    for (let i = 0; i < iterations; i++) {
      const path = this.select();
      const leaf = path[path.length - 1];
      const reward = await this.simulate(leaf);
      this.backpropagate(path, reward);
    }

    const bestChild = this.bestChild(this.root);
    const bestAction = bestChild.state.lastAction || 'none';

    mercyHaptic.playPattern(valence > 0.9 ? 'cosmicHarmony' : 'neutralPulse', valence);
    console.log(`[MCTS] Best action: ${bestAction} (visits: ${bestChild.visits}, value: ${bestChild.value.toFixed(3)})`);

    return bestAction;
  }

  /**
   * Selection phase – UCT + valence bonus
   */
  private select(): Node[] {
    const path: Node[] = [];
    let node = this.root;

    while (!node.isTerminal && node.untriedActions.length === 0) {
      path.push(node);
      node = this.bestChild(node);
    }

    path.push(node);

    // Expand if not terminal and has untried actions
    if (!node.isTerminal && node.untriedActions.length > 0) {
      const action = node.untriedActions.shift()!;
      const childState = this.applyAction(node.state, action);
      const child: Node = {
        state: childState,
        parent: node,
        children: [],
        visits: 0,
        value: 0,
        untriedActions: this.getActions(childState),
        isTerminal: this.isTerminal(childState),
        depth: node.depth + 1
      };
      node.children.push(child);
      path.push(child);
    }

    return path;
  }

  /**
   * UCT selection with valence bonus
   */
  private bestChild(node: Node): Node {
    const valence = currentValence.get();
    const c = EXPLORATION_CONSTANT * (1 + valence * VALENCE_BONUS_FACTOR);

    return node.children.reduce((best, child) => {
      const uct = (child.value / child.visits) + c * Math.sqrt(Math.log(node.visits) / child.visits);
      return uct > best.uct ? { child, uct } : best;
    }, { child: node.children[0], uct: -Infinity }).child;
  }

  /**
   * Simulation / rollout – random playout to terminal state
   */
  private async simulate(node: Node): Promise<number> {
    let current = node.state;
    let depth = node.depth;

    while (!this.isTerminal(current) && depth < MAX_DEPTH) {
      const actions = this.getActions(current);
      if (actions.length === 0) break;
      const action = actions[Math.floor(Math.random() * actions.length)];
      current = this.applyAction(current, action);
      depth++;
    }

    // Terminal reward = future valence projection
    return this.evaluateState(current);
  }

  /**
   * Backpropagation – update value & visits up the tree
   */
  private backpropagate(path: Node[], reward: number) {
    for (const node of path.reverse()) {
      node.visits++;
      node.value += reward;
    }
  }

  // ─── Abstract methods – must be implemented by concrete planner ───
  protected getActions(state: any): string[] {
    throw new Error("getActions not implemented");
  }

  protected applyAction(state: any, action: string): any {
    throw new Error("applyAction not implemented");
  }

  protected isTerminal(state: any): boolean {
    throw new Error("isTerminal not implemented");
  }

  protected evaluateState(state: any): number {
    // Default: use future valence projection
    return currentValence.get(); // override in concrete classes
  }

  // ─── Concrete planner example: simple negotiation tree ───
  static createNegotiationPlanner(currentState: any) {
    return new (class extends MCTS {
      protected getActions(state: any): string[] {
        return ['propose-alliance', 'offer-resources', 'request-aid', 'reject', 'wait'];
      }

      protected applyAction(state: any, action: string): any {
        return { ...state, lastAction: action }; // simplified
      }

      protected isTerminal(state: any): boolean {
        return !!state.outcome;
      }

      protected evaluateState(state: any): number {
        return state.valence || currentValence.get();
      }
    })(currentState, ['propose-alliance', 'offer-resources', 'request-aid', 'reject', 'wait']);
  }
}

export default MCTS;

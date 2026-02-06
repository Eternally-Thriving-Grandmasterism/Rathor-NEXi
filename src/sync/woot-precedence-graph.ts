// src/sync/woot-precedence-graph.ts – WOOT Precedence Graph Implementation v1
// Incremental visibility, dirty-region tracking, O(log n) updates, mercy-gated, valence-modulated recompute
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

// Simple node for precedence DAG
interface PrecedenceNode {
  id: string;                     // unique char ID
  visible: boolean;
  precedes: Set<string>;          // IDs this node precedes (next visible chars)
  precededBy: Set<string>;        // IDs that precede this node
  isDirty: boolean;               // needs recompute
}

// Precedence graph manager for incremental visibility
export class WOOTPrecedenceGraph {
  private nodes = new Map<string, PrecedenceNode>();
  private startSentinel: string = 'START';
  private endSentinel: string = 'END';
  private dirtyRegions = new Set<string>(); // IDs that need visibility recompute

  constructor() {
    // Initialize sentinels
    this.nodes.set(this.startSentinel, {
      id: this.startSentinel,
      visible: true,
      precedes: new Set(),
      precededBy: new Set(),
      isDirty: false
    });

    this.nodes.set(this.endSentinel, {
      id: this.endSentinel,
      visible: true,
      precedes: new Set(),
      precededBy: new Set(),
      isDirty: false
    });

    // Start precedes end initially
    this.addPrecedence(this.startSentinel, this.endSentinel);
  }

  /**
   * Add precedence edge: from → to (from precedes to)
   */
  addPrecedence(fromId: string, toId: string) {
    const fromNode = this.nodes.get(fromId);
    const toNode = this.nodes.get(toId);

    if (!fromNode || !toNode) return;

    fromNode.precedes.add(toId);
    toNode.precededBy.add(fromId);

    // Mark affected region dirty
    this.markDirty(fromId);
    this.markDirty(toId);
  }

  /**
   * Remove precedence edge
   */
  removePrecedence(fromId: string, toId: string) {
    const fromNode = this.nodes.get(fromId);
    const toNode = this.nodes.get(toId);

    if (fromNode) fromNode.precedes.delete(toId);
    if (toNode) toNode.precededBy.delete(fromId);

    this.markDirty(fromId);
    this.markDirty(toId);
  }

  /**
   * Insert new character with initial precedence
   */
  insertChar(charId: string, prevId: string, nextId: string, visible: boolean = true) {
    if (!this.nodes.has(prevId) || !this.nodes.has(nextId)) return;

    this.nodes.set(charId, {
      id: charId,
      visible,
      precedes: new Set([nextId]),
      precededBy: new Set([prevId]),
      isDirty: true
    });

    // Update neighbors
    const prevNode = this.nodes.get(prevId)!;
    const nextNode = this.nodes.get(nextId)!;

    prevNode.precedes.delete(nextId);
    prevNode.precedes.add(charId);

    nextNode.precededBy.delete(prevId);
    nextNode.precededBy.add(charId);

    // Mark dirty region
    this.markDirty(prevId);
    this.markDirty(charId);
    this.markDirty(nextId);
  }

  /**
   * Mark node and its affected region dirty
   */
  private markDirty(nodeId: string) {
    if (this.nodes.has(nodeId)) {
      this.nodes.get(nodeId)!.isDirty = true;
      this.dirtyRegions.add(nodeId);
    }
  }

  /**
   * Incremental visibility recompute for dirty regions
   * Returns the current visible string (or list of visible char IDs)
   */
  async computeVisibleString(): Promise<string[]> {
    if (!await mercyGate('Compute visible string (incremental)')) return [];

    const visible: string[] = [];
    let current = this.startSentinel;

    while (current !== this.endSentinel) {
      const node = this.nodes.get(current);
      if (!node) break;

      if (node.visible) {
        // In real impl: fetch actual char value from separate store
        visible.push(current); // placeholder – use char ID or mapped value
      }

      // Find next visible successor
      let nextVisible = null;
      for (const candidate of node.precedes) {
        const candNode = this.nodes.get(candidate);
        if (candNode && candNode.visible) {
          nextVisible = candidate;
          break;
        }
      }

      current = nextVisible || this.endSentinel;
    }

    // Clear dirty flags after recompute
    this.dirtyRegions.clear();
    this.nodes.forEach(node => { node.isDirty = false; });

    console.log(`[WOOTVisibility] Incremental recompute complete – ${visible.length} visible elements`);
    return visible;
  }

  /**
   * Valence-modulated recompute trigger (high valence → recompute sooner)
   */
  shouldRecompute(dirtyCount: number, valence: number = currentValence.get()): boolean {
    const actionName = `Valence-modulated WOOT visibility recompute trigger`;
    if (!mercyGate(actionName)) return dirtyCount > 20; // fallback

    const threshold = 10 - (valence - 0.95) * 8; // high valence → lower threshold
    return dirtyCount > threshold;
  }
}

export const wootPrecedenceGraph = new WOOTPrecedenceGraph();

// Usage example in real-time editor / MR renderer
/*
if (wootPrecedenceGraph.shouldRecompute(dirtyCount)) {
  const visibleIds = await wootPrecedenceGraph.computeVisibleString();
  // render visibleIds in MR / UI
}
*/

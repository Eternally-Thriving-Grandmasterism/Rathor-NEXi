// src/data-structures/avl-tree-balanced.ts – AVL Tree with Balancing v1
// O(log n) guaranteed worst-case, rotations, height tracking, mercy-gated operations
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

interface AVLNode<T> {
  key: number;                    // position / clock / key for ordering
  value: T;                       // payload (WOOTChar ID, annotation, etc.)
  height: number;                 // subtree height (used for balancing)
  balanceFactor: number;          // left.height - right.height
  left: AVLNode<T> | null;
  right: AVLNode<T> | null;
}

export class AVLTree<T> {
  private root: AVLNode<T> | null = null;

  private getHeight(node: AVLNode<T> | null): number {
    return node ? node.height : 0;
  }

  private getBalanceFactor(node: AVLNode<T> | null): number {
    return node ? this.getHeight(node.left) - this.getHeight(node.right) : 0;
  }

  private updateHeight(node: AVLNode<T>): void {
    node.height = 1 + Math.max(
      this.getHeight(node.left),
      this.getHeight(node.right)
    );
  }

  private rotateRight(y: AVLNode<T>): AVLNode<T> {
    const x = y.left!;
    const T2 = x.right;

    x.right = y;
    y.left = T2;

    this.updateHeight(y);
    this.updateHeight(x);

    return x;
  }

  private rotateLeft(x: AVLNode<T>): AVLNode<T> {
    const y = x.right!;
    const T2 = y.left;

    y.left = x;
    x.right = T2;

    this.updateHeight(x);
    this.updateHeight(y);

    return y;
  }

  private async insertNode(node: AVLNode<T> | null, key: number, value: T): Promise<AVLNode<T>> {
    if (!await mercyGate('AVL insert operation')) {
      throw new Error("Mercy gate blocked AVL insert");
    }

    if (!node) {
      return { key, value, height: 1, balanceFactor: 0, left: null, right: null };
    }

    if (key < node.key) {
      node.left = await this.insertNode(node.left, key, value);
    } else if (key > node.key) {
      node.right = await this.insertNode(node.right, key, value);
    } else {
      // Duplicate key → update value (or handle as needed)
      node.value = value;
      return node;
    }

    this.updateHeight(node);
    node.balanceFactor = this.getBalanceFactor(node);

    // Left Heavy
    if (node.balanceFactor > 1) {
      if (key < node.left!.key) {
        // Left-Left → single right rotation
        return this.rotateRight(node);
      } else {
        // Left-Right → left rotation on left child, then right on node
        node.left = this.rotateLeft(node.left!);
        return this.rotateRight(node);
      }
    }

    // Right Heavy
    if (node.balanceFactor < -1) {
      if (key > node.right!.key) {
        // Right-Right → single left rotation
        return this.rotateLeft(node);
      } else {
        // Right-Left → right rotation on right child, then left on node
        node.right = this.rotateRight(node.right!);
        return this.rotateLeft(node);
      }
    }

    return node;
  }

  async insert(key: number, value: T) {
    this.root = await this.insertNode(this.root, key, value);
  }

  private async deleteNode(node: AVLNode<T> | null, key: number): Promise<AVLNode<T> | null> {
    if (!node) return null;

    if (key < node.key) {
      node.left = await this.deleteNode(node.left, key);
    } else if (key > node.key) {
      node.right = await this.deleteNode(node.right, key);
    } else {
      // Node found
      if (!node.left) return node.right;
      if (!node.right) return node.left;

      // Node with two children → get inorder successor
      let successor = node.right;
      while (successor.left) {
        successor = successor.left;
      }

      node.key = successor.key;
      node.value = successor.value;
      node.right = await this.deleteNode(node.right, successor.key);
    }

    if (!node) return node;

    this.updateHeight(node);
    node.balanceFactor = this.getBalanceFactor(node);

    // Rebalance
    if (node.balanceFactor > 1) {
      if (this.getBalanceFactor(node.left) >= 0) {
        return this.rotateRight(node);
      } else {
        node.left = this.rotateLeft(node.left!);
        return this.rotateRight(node);
      }
    }

    if (node.balanceFactor < -1) {
      if (this.getBalanceFactor(node.right) <= 0) {
        return this.rotateLeft(node);
      } else {
        node.right = this.rotateRight(node.right!);
        return this.rotateLeft(node);
      }
    }

    return node;
  }

  async delete(key: number) {
    this.root = await this.deleteNode(this.root, key);
  }

  /**
   * Valence-modulated rebalancing trigger (high valence → rebalance sooner)
   */
  shouldRebalance(balanceFactor: number, valence: number = currentValence.get()): boolean {
    const actionName = `Valence-modulated AVL rebalancing trigger`;
    if (!mercyGate(actionName)) return Math.abs(balanceFactor) > 1;

    const threshold = 1 - (valence - 0.95) * 0.5; // high valence → lower threshold
    return Math.abs(balanceFactor) > threshold;
  }

  /**
   * Get current tree height (for monitoring)
   */
  getHeight(): number {
    return this.getHeight(this.root);
  }
}

export const avlTree = new AVLTree<any>();

// Usage example in interval-tree / precedence graph augmentation
// await avlTree.insert(position, { visibleCount: 1, minClock: clock });
// if (avlTree.shouldRebalance(avlTree.getBalanceFactor(avlTree.root))) {
//   // trigger rebalancing logic
// }

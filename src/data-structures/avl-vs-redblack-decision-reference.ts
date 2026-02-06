// src/data-structures/avl-vs-redblack-decision-reference.ts – AVL vs Red-Black Decision Reference & Mercy Helpers v1
// Guidelines for choosing AVL vs Red-Black, valence-modulated switching
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * AVL vs Red-Black decision reference – mercy-aligned guidelines
 */
export const AVLvsRedBlackDecisionReference = {
  chooseAVLWhen: [
    "Worst-case lookup latency is critical (MR rendering, live collective valence queries)",
    "Read-heavy workload (dashboard sync monitoring, gesture range searches)",
    "Predictable O(log n) matters more than amortized write throughput",
    "Simpler balance condition desired for debugging & lattice auditing"
  ],
  chooseRedBlackWhen: [
    "Write-heavy workload (probe command bursts, swarm progression updates)",
    "Fewer rotations desired (lower constant factor in mixed read/write)",
    "Standard-library compatibility needed (Java TreeMap, C++ std::map patterns)",
    "Multi-device / multiplanetary sync delta volume is high"
  ],
  mercy_valence_switch: "High valence (>0.95) → prefer AVL (stability & thriving convergence); low valence (<0.9) → prefer Red-Black (survival exploration & write throughput)"
};

/**
 * Valence-modulated tree choice helper
 */
export function chooseTreeTypeForWorkload(
  isReadHeavy: boolean,
  valence: number = currentValence.get()
): 'AVL' | 'RedBlack' {
  const actionName = `Valence-modulated AVL vs Red-Black choice`;
  if (!mercyGate(actionName)) return 'RedBlack'; // fallback to write-friendly

  if (valence > 0.95) {
    // High valence → prefer stability (AVL)
    return 'AVL';
  } else if (valence < 0.9) {
    // Low valence → prefer write throughput (Red-Black)
    return 'RedBlack';
  } else {
    // Balanced valence → choose based on workload
    return isReadHeavy ? 'AVL' : 'RedBlack';
  }
}

// Usage example when deciding tree type for new structure
// const treeType = chooseTreeTypeForWorkload(isReadHeavyQuery);
// if (treeType === 'AVL') {
//   useAVLTree();
// } else {
//   useRedBlackTree();
// }

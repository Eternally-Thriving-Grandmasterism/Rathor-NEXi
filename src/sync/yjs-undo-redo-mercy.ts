// src/sync/yjs-undo-redo-mercy.ts – Valence-Aware Yjs Undo/Redo Manager v1
// Mercy-gated undo/redo, high-valence auto-snapshot, low-valence prune
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { UndoManager } from 'y-undo-manager';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_SNAPSHOT_PIVOT = 0.95;
const VALENCE_UNDO_GATE = 0.9;
const SNAPSHOT_INTERVAL_MS = 5000; // snapshot every 5s when high valence

let undoManager: UndoManager | null = null;
let snapshotInterval: NodeJS.Timeout | null = null;

export class YjsUndoRedoMercy {
  static initialize(ydoc: Y.Doc, trackedTypes: Y.AbstractType<any>[]) {
    const actionName = 'Initialize valence-aware Yjs undo/redo manager';
    if (!mercyGate(actionName)) return;

    undoManager = new UndoManager(trackedTypes, {
      captureTimeout: 300, // group rapid changes into one undo step
      trackedOrigins: new Set([true]) // only local changes
    });

    // Valence-weighted snapshot on high valence
    snapshotInterval = setInterval(() => {
      if (currentValence.get() > VALENCE_SNAPSHOT_PIVOT && undoManager) {
        undoManager.snapshot();
        mercyHaptic.playPattern('eternalHarmony', currentValence.get());
        console.log("[YjsUndoRedoMercy] High-valence snapshot captured");
      }
    }, SNAPSHOT_INTERVAL_MS);

    // Mercy gate on undo/redo
    undoManager.on('stack-item-added', () => {
      const valence = currentValence.get();
      if (valence < VALENCE_UNDO_GATE) {
        undoManager!.clear();
        console.warn("[YjsUndoRedoMercy] Undo stack cleared – low valence detected");
      }
    });

    console.log("[YjsUndoRedoMercy] Initialized – valence-aware undo/redo active");
  }

  static canUndo(): boolean {
    return undoManager?.canUndo() && currentValence.get() >= VALENCE_UNDO_GATE;
  }

  static canRedo(): boolean {
    return undoManager?.canRedo() ?? false;
  }

  static undo() {
    if (!undoManager || !this.canUndo()) return;

    const valenceBefore = currentValence.get();
    undoManager.undo();
    mercyHaptic.playPattern('timeReverse', valenceBefore);
    console.log("[YjsUndoRedoMercy] Undo performed");
  }

  static redo() {
    if (!undoManager || !this.canRedo()) return;

    undoManager.redo();
    mercyHaptic.playPattern('timeForward', currentValence.get());
    console.log("[YjsUndoRedoMercy] Redo performed");
  }

  static destroy() {
    if (snapshotInterval) clearInterval(snapshotInterval);
    if (undoManager) {
      undoManager.destroy();
      undoManager = null;
    }
  }

  static getUndoStackSize(): number {
    return undoManager?.undoStack.length ?? 0;
  }

  static getRedoStackSize(): number {
    return undoManager?.redoStack.length ?? 0;
  }
}

export default YjsUndoRedoMercy;

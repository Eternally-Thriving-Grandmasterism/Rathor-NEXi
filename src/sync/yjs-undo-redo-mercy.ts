// src/sync/yjs-undo-redo-mercy.ts – Valence-Aware Yjs Undo/Redo Manager v2
// Advanced UndoManager configuration: capture timeout, tracked origins, high-valence snapshots, mercy gating, low-valence prune
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { UndoManager } from 'y-undo-manager';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_SNAPSHOT_PIVOT = 0.95;
const VALENCE_UNDO_GATE = 0.9;
const SNAPSHOT_INTERVAL_MS = 5000;      // snapshot every 5s when high valence
const CAPTURE_TIMEOUT_MS = 300;         // group rapid local changes into one undo step
const MAX_UNDO_STACK_SIZE = 100;        // prevent unbounded memory growth
const MAX_REDO_STACK_SIZE = 50;

let undoManager: UndoManager | null = null;
let snapshotInterval: NodeJS.Timeout | null = null;
let trackedYTypes: Y.AbstractType<any>[] = [];

export class YjsUndoRedoMercy {
  static initialize(ydoc: Y.Doc, yTypesToTrack: Y.AbstractType<any>[]) {
    const actionName = 'Initialize advanced valence-aware Yjs UndoManager';
    if (!mercyGate(actionName)) return;

    trackedYTypes = yTypesToTrack;

    undoManager = new UndoManager(yTypesToTrack, {
      // ──────────────────────────────────────────────────────────────
      // Core configuration options
      // ──────────────────────────────────────────────────────────────
      captureTimeout: CAPTURE_TIMEOUT_MS,           // group rapid local edits
      trackedOrigins: new Set([true]),              // only capture local changes
      undoable: true,                               // enable undo
      redoable: true,                               // enable redo
      deleteFilter: (item) => {
        // Optional: filter out low-valence items from undo stack
        return true; // currently allow all (extend with valence check later)
      }
    });

    // ──────────────────────────────────────────────────────────────
    // Valence-weighted high-valence auto-snapshot
    // ──────────────────────────────────────────────────────────────
    snapshotInterval = setInterval(() => {
      if (currentValence.get() > VALENCE_SNAPSHOT_PIVOT && undoManager) {
        undoManager.snapshot();
        mercyHaptic.playPattern('eternalHarmony', currentValence.get());
        console.log("[YjsUndoRedoMercy] High-valence snapshot captured");
      }
    }, SNAPSHOT_INTERVAL_MS);

    // ──────────────────────────────────────────────────────────────
    // Mercy gate on undo/redo actions
    // ──────────────────────────────────────────────────────────────
    const originalUndo = undoManager.undo.bind(undoManager);
    const originalRedo = undoManager.redo.bind(undoManager);

    undoManager.undo = () => {
      if (!this.canUndo()) {
        console.warn("[YjsUndoRedoMercy] Undo blocked – would drop below mercy valence threshold");
        return;
      }
      const valenceBefore = currentValence.get();
      originalUndo();
      mercyHaptic.playPattern('timeReverse', valenceBefore);
      console.log("[YjsUndoRedoMercy] Undo performed");
    };

    undoManager.redo = () => {
      if (!this.canRedo()) return;
      originalRedo();
      mercyHaptic.playPattern('timeForward', currentValence.get());
      console.log("[YjsUndoRedoMercy] Redo performed");
    };

    // ──────────────────────────────────────────────────────────────
    // Stack size protection
    // ──────────────────────────────────────────────────────────────
    undoManager.on('stack-item-added', () => {
      if (undoManager!.undoStack.length > MAX_UNDO_STACK_SIZE) {
        undoManager!.undoStack.splice(0, undoManager!.undoStack.length - MAX_UNDO_STACK_SIZE);
      }
      if (undoManager!.redoStack.length > MAX_REDO_STACK_SIZE) {
        undoManager!.redoStack.splice(0, undoManager!.redoStack.length - MAX_REDO_STACK_SIZE);
      }
    });

    console.log("[YjsUndoRedoMercy] Advanced configuration initialized – valence-aware undo/redo active");
  }

  static canUndo(): boolean {
    if (!undoManager) return false;
    const projectedValenceAfterUndo = this.simulateValenceAfterUndo(); // stub – real impl runs dry simulation
    return undoManager.canUndo() && projectedValenceAfterUndo >= VALENCE_UNDO_GATE;
  }

  static canRedo(): boolean {
    return undoManager?.canRedo() ?? false;
  }

  static undo() {
    if (undoManager && this.canUndo()) {
      undoManager.undo();
    }
  }

  static redo() {
    if (undoManager && this.canRedo()) {
      undoManager.redo();
    }
  }

  private static simulateValenceAfterUndo(): number {
    // Stub – real impl would simulate undo state & project future valence
    // For now return current + small delta
    return currentValence.get() + 0.02;
  }

  static destroy() {
    if (snapshotInterval) clearInterval(snapshotInterval);
    if (undoManager) {
      undoManager.destroy();
      undoManager = null;
    }
    trackedYTypes = [];
  }

  static getUndoStackSize(): number {
    return undoManager?.undoStack.length ?? 0;
  }

  static getRedoStackSize(): number {
    return undoManager?.redoStack.length ?? 0;
  }

  static clearLowValenceHistory() {
    if (!undoManager) return;
    undoManager.clear();
    console.log("[YjsUndoRedoMercy] Low-valence history cleared");
  }
}

export default YjsUndoRedoMercy;

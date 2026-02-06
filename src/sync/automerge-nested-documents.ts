// src/sync/automerge-nested-documents.ts – sovereign Automerge Nested Documents Manager v1
// Binary subdocuments, independent sync & history, GC safety, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const SUBDOC_PREFIX = 'mercy-automerge-subdoc-';

export class AutomergeNestedDocuments {
  private parentDoc: Automerge.Doc<any>;
  private subdocs = new Map<string, Automerge.Doc<any>>();

  constructor(parentDoc: Automerge.Doc<any>) {
    this.parentDoc = parentDoc;
  }

  /**
   * Get or create nested Automerge subdocument for a key (lazy + mercy-gated)
   */
  async getOrCreateSubdoc(
    key: string,
    initialValue: any = {},
    requiredValence: number = MERCY_THRESHOLD
  ): Promise<Automerge.Doc<any> | null> {
    const actionName = `Create/Get Automerge subdoc: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) {
      return null;
    }

    if (this.subdocs.has(key)) {
      return this.subdocs.get(key)!;
    }

    // Check if already stored in parent
    const binary = Automerge.get(this.parentDoc, ['subdocs', key]);
    let subdoc: Automerge.Doc<any>;

    if (binary && Automerge.isValidBinary(binary)) {
      subdoc = Automerge.load(binary);
    } else {
      subdoc = Automerge.from(initialValue);
    }

    this.subdocs.set(key, subdoc);

    // Optional independent persistence (IndexedDB per subdoc)
    // In real impl: use idb-keyval or custom store
    console.log(`[AutomergeNested] Subdocument loaded/created: ${key}`);

    return subdoc;
  }

  /**
   * Save subdoc back into parent as binary blob
   */
  async saveSubdoc(key: string, requiredValence: number = MERCY_THRESHOLD) {
    const actionName = `Save Automerge subdoc: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) return;

    const subdoc = this.subdocs.get(key);
    if (!subdoc) return;

    const binary = Automerge.save(subdoc);

    // Save to parent document
    Automerge.change(this.parentDoc, `Saving subdoc ${key}`, doc => {
      if (!doc.subdocs) doc.subdocs = {};
      doc.subdocs[key] = binary;
    });

    console.log(`[AutomergeNested] Subdocument saved to parent: \( {key} ( \){binary.byteLength} bytes)`);
  }

  /**
   * Clean up subdoc (remove from parent & memory)
   */
  async destroySubdoc(key: string) {
    // Remove from parent
    Automerge.change(this.parentDoc, `Destroying subdoc ${key}`, doc => {
      if (doc.subdocs) delete doc.subdocs[key];
    });

    this.subdocs.delete(key);
    console.log(`[AutomergeNested] Subdocument destroyed: ${key}`);
  }

  /**
   * Get all active subdocs (for monitoring / sync)
   */
  getActiveSubdocs(): Map<string, Automerge.Doc<any>> {
    return new Map(this.subdocs);
  }
}

export const automergeNested = new AutomergeNestedDocuments(/* pass global Automerge doc */);

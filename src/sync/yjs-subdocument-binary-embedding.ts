// src/sync/yjs-subdocument-binary-embedding.ts – Yjs Subdocument Binary Embedding Manager v1
// Creation, serialization, lazy loading, independent sync, GC safety, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { IndexeddbPersistence } from 'y-indexeddb';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const SUBDOC_PREFIX = 'mercy-yjs-subdoc-';

export class YjsSubdocumentBinaryEmbedding {
  private parentDoc: Y.Doc;
  private subdocs = new Map<string, Y.Doc>();
  private persistences = new Map<string, IndexeddbPersistence>();

  constructor(parentDoc: Y.Doc) {
    this.parentDoc = parentDoc;
  }

  /**
   * Get or create subdocument (lazy load from binary if exists)
   */
  async getOrCreate(
    key: string,
    initialValue: any = {},
    requiredValence: number = MERCY_THRESHOLD
  ): Promise<Y.Doc | null> {
    const actionName = `Create/Get Yjs subdoc: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) {
      return null;
    }

    if (this.subdocs.has(key)) {
      return this.subdocs.get(key)!;
    }

    const parentMap = this.parentDoc.getMap('subdocs');
    const binary = parentMap.get(key);

    let subdoc: Y.Doc;

    if (binary instanceof Uint8Array) {
      subdoc = new Y.Doc();
      Y.applyUpdate(subdoc, binary);
      console.log(`[YjsSubdocBinary] Loaded subdoc from binary: \( {key} ( \){binary.byteLength} bytes)`);
    } else {
      subdoc = new Y.Doc();
      // Initialize with initial value if needed
      if (Object.keys(initialValue).length > 0) {
        Y.transact(subdoc, () => {
          const map = subdoc.getMap('state');
          Object.entries(initialValue).forEach(([k, v]) => map.set(k, v));
        });
      }
      console.log(`[YjsSubdocBinary] Created new subdoc: ${key}`);
    }

    // Independent persistence
    const persistenceName = `\( {SUBDOC_PREFIX} \){key}`;
    const persistence = new IndexeddbPersistence(persistenceName, subdoc);
    this.persistences.set(key, persistence);

    // Optional independent relay (high valence only)
    if (currentValence.get() > 0.95 && navigator.onLine) {
      // e.g. new WebsocketProvider(RELAY_URL, persistenceName, subdoc);
    }

    this.subdocs.set(key, subdoc);
    return subdoc;
  }

  /**
   * Save subdoc back to parent as binary blob
   */
  async save(key: string, requiredValence: number = MERCY_THRESHOLD) {
    const actionName = `Save Yjs subdoc binary: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) return;

    const subdoc = this.subdocs.get(key);
    if (!subdoc) return;

    const binary = Y.encodeStateAsUpdate(subdoc);

    // Save to parent
    Y.transact(this.parentDoc, () => {
      const parentMap = this.parentDoc.getMap('subdocs');
      parentMap.set(key, binary);
    });

    console.log(`[YjsSubdocBinary] Subdoc saved as binary: \( {key} ( \){binary.byteLength} bytes)`);
  }

  /**
   * Destroy subdoc (detach + cleanup persistence)
   */
  async destroy(key: string) {
    const subdoc = this.subdocs.get(key);
    if (!subdoc) return;

    // Detach from parent
    Y.transact(this.parentDoc, () => {
      const parentMap = this.parentDoc.getMap('subdocs');
      parentMap.delete(key);
    });

    // Cleanup persistence
    const persistence = this.persistences.get(key);
    if (persistence) {
      await persistence.destroy();
      this.persistences.delete(key);
    }

    this.subdocs.delete(key);
    console.log(`[YjsSubdocBinary] Subdoc destroyed: ${key}`);
  }
}

export const yjsSubdocBinary = new YjsSubdocumentBinaryEmbedding(/* pass global ydoc */);

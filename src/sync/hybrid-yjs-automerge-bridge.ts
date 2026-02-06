// src/sync/hybrid-yjs-automerge-bridge.ts – Hybrid Yjs + Automerge Bridge v1
// Real-time Yjs UI view + durable Automerge history, bidirectional delta sync
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import * as Automerge from '@automerge/automerge';
import { mercyGate } from '@/core/mercy-gate';

export class HybridYjsAutomergeBridge {
  private ydoc: Y.Doc;
  private automergeRoot: Automerge.Doc<any>;

  constructor(ydoc: Y.Doc, automergeRoot: Automerge.Doc<any>) {
    this.ydoc = ydoc;
    this.automergeRoot = automergeRoot;
  }

  async syncAutomergeToYjs(key: string) {
    if (!await mercyGate(`Sync Automerge → Yjs: ${key}`)) return;

    const binary = Automerge.get(this.automergeRoot, ['subdocs', key]);
    if (!(binary instanceof Uint8Array)) return;

    const yMap = this.ydoc.getMap('subdocs');
    yMap.set(key, binary);

    console.log(`[HybridBridge] Automerge → Yjs sync: \( {key} ( \){binary.byteLength} bytes)`);
  }

  async syncYjsToAutomerge(key: string) {
    if (!await mercyGate(`Sync Yjs → Automerge: ${key}`)) return;

    const binary = this.ydoc.getMap('subdocs').get(key);
    if (!(binary instanceof Uint8Array)) return;

    Automerge.change(this.automergeRoot, `Sync from Yjs ${key}`, doc => {
      if (!doc.subdocs) doc.subdocs = {};
      doc.subdocs[key] = binary;
    });

    console.log(`[HybridBridge] Yjs → Automerge sync: ${key}`);
  }
}

export const hybridBridge = new HybridYjsAutomergeBridge(/* ydoc, automergeRoot */);

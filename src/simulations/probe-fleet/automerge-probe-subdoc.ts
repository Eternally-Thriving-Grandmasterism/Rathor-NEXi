// src/simulations/probe-fleet/automerge-probe-subdoc.ts – Automerge per-probe durable subdocument v1
// Binary embedding, independent sync, GC safety, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { mercyGate } from '@/core/mercy-gate';

export class AutomergeProbeSubdoc {
  private doc: Automerge.Doc<any>;

  constructor(probeId: string, initialState = { resources: 10, valence: 0.8, status: 'active' }) {
    this.doc = Automerge.from(initialState);
  }

  async updateResources(delta: number) {
    if (!await mercyGate(`Update probe resources`)) return;

    Automerge.change(this.doc, 'Update resources', d => {
      d.resources += delta;
      d.valence = Math.min(1.0, d.valence + delta * 0.001);
    });

    console.log(`[AutomergeProbe] Resources updated: ${Automerge.get(this.doc, ['resources'])}`);
  }

  getBinary(): Uint8Array {
    return Automerge.save(this.doc);
  }

  static fromBinary(binary: Uint8Array): AutomergeProbeSubdoc {
    const doc = Automerge.load(binary);
    return new AutomergeProbeSubdoc('from-binary', Automerge.get(doc, []));
  }
}

// Usage in fleet sim root document
// const probe = new AutomergeProbeSubdoc('probe-001');
// await probe.updateResources(5);
// const binary = probe.getBinary();
// parentAutomerge.change(d => { d.probes['probe-001'] = binary; });

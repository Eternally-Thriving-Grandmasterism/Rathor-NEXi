// src/simulations/probe-fleet/electricsql-probe-sync.ts – ElectricSQL per-probe metadata sync v1
// Relational durability for probe state, mercy-gated updates
// MIT License – Autonomicity Games Inc. 2026

import { electricInitializer } from '@/sync/electricsql-initializer';
import { mercyGate } from '@/core/mercy-gate';

export async function syncProbeMetadataToElectric(probeId: string, data: { resources: number; valence: number; habitatId?: string }) {
  const actionName = `Sync probe metadata: ${probeId}`;
  if (!await mercyGate(actionName)) return;

  const electric = electricInitializer.getElectricClient();
  if (!electric) return;

  await electric.db.probes.upsert({
    id: probeId,
    resources: data.resources,
    valence: data.valence,
    habitatId: data.habitatId,
    updatedAt: Date.now()
  });

  await electric.sync();

  console.log(`[ElectricProbeSync] Probe ${probeId} metadata synced – resources ${data.resources}, valence ${data.valence.toFixed(4)}`);
}

// Example usage in probe fleet sim
// await syncProbeMetadataToElectric('probe-001', { resources: 42, valence: 0.999 });

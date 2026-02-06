// src/simulations/probe-fleet/subdoc-per-probe.ts
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb';
import { mercyGate } from '@/core/mercy-gate';

export async function createProbeSubdoc(probeId: string) {
  if (!await mercyGate(`Create probe subdoc: ${probeId}`)) return null;

  const subdoc = new Y.Doc();

  // Independent persistence per probe
  const persistence = new IndexeddbPersistence(`probe-${probeId}`, subdoc);

  // Independent sync channel per probe
  const provider = new WebsocketProvider(
    'wss://relay.rathor.ai',
    `probe-${probeId}`,
    subdoc
  );

  // Initialize probe state
  const state = subdoc.getMap('state');
  state.set('resources', 10);
  state.set('valence', currentValence.get());
  state.set('status', 'active');

  console.log(`[ProbeSubdoc] Created & synced probe-${probeId}`);

  return { subdoc, persistence, provider };
}

// Usage in fleet sim
// const probe = await createProbeSubdoc('probe-001');
// probe.subdoc.getMap('state').set('resources', 15);

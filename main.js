// main.js – sovereign entry with persistence & dossiers
// MIT License – Autonomicity Games Inc. 2026

import { hyperon } from './hyperon-runtime.js';
import { seedProfessionalDossiers } from './professional-dossiers-seeder.js';
// Assume chat-ui-streaming.js handles load

async function initRathor() {
  await hyperon.init();
  await seedProfessionalDossiers();
  console.log("[Rathor] Sovereign lattice + professional mercy seeded – streaming eternal");
}

initRathor();

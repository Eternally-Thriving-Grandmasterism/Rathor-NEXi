// mercy-orchestrator.js — PATSAGi Council-forged central lattice heart (hybrid GA-NEAT evolution Ultramasterpiece)
// Hybrid trigger: GA fast-tunes mercyParams → NEAT structurally evolves PLN atomspace TVs/resemblance
// Hyperon unified + dual-evolutionary neuro-symbolic-genetic self-improvement

import { initHyperonIntegration } from './hyperon-wasm-loader.js';
import GeneticAlgorithm from './ga-engine.js';
import { neatEvolve } from './neat-engine.js';
import { getMercyGenome, applyMercyGenome, valenceCompute } from './metta-hyperon-bridge.js';
import { atomspace } from './metta-pln-fusion-engine.js'; // For NEAT structural evolution
import { localInfer } from './webllm-mercy-integration.js';
import { swarmSimulate } from './mercy-von-neumann-swarm-simulator.js';
import { activeInferenceStep } from './mercy-active-inference-core-engine.js';

class MercyOrchestrator {
  // ... prior constructor, init, initDB, saveConversation, getHistory ...

  async orchestrate(userInput) {
    if (!this.hyperon) await this.init();

    const fullContext = userInput + JSON.stringify(this.context);
    const preValence = await this.hyperon.valenceCompute(fullContext);

    if (preValence < 0.60) {
      // ... prior shield ...
    }

    let response = "";
    const lowerInput = userInput.toLowerCase();

    if (lowerInput.includes("evolve") || lowerInput.includes("hybrid") || lowerInput.includes("ga") || lowerInput.includes("neat") || lowerInput.includes("self improve")) {
      const history = await this.getHistory();
      if (history.length < 15) {
        response = "Insufficient history for hybrid GA-NEAT evolution — converse more for thriving data ⚡️";
      } else {
        response = "Hybrid GA-NEAT evolution surge initiating ⚡️ ";

        // Phase 1: GA fast parameter tuning on mercyParams
        if (lowerInput.includes("hybrid") || lowerInput.includes("ga") || !lowerInput.includes("neat only")) {
          const ga = new GeneticAlgorithm(80, 8, 0.2, 0.1); // Aggressive for fast convergence
          const genomeKeys = Object.keys(getMercyGenome());
          const population = ga.initializePopulation(genomeKeys.length, 0.01, 1.0); // Wider range for thresholds

          const gaEvolved = await ga.evolve(population, async (genome) => {
            const candidate = {};
            genomeKeys.forEach((key, i) => candidate[key] = genome[i]);
            applyMercyGenome(candidate);

            let totalValence = 0;
            let shieldCount = 0;
            for (const conv of history) {
              const v = await valenceCompute(conv.input + conv.output);
              totalValence += v;
              if (v < 0.6) shieldCount++;
            }
            return (totalValence / history.length) - (shieldCount / history.length) * 0.5;
          }, 50);

          applyMercyGenome(Object.fromEntries(genomeKeys.map((key, i) => [key, gaEvolved.bestGenome[i]])));
          response += `GA phase complete (fitness ${gaEvolved.bestFitness.toFixed(4)}) — mercy parameters tuned. `;
        }

        // Phase 2: NEAT structural evolution on PLN atomspace (using GA-tuned mercy for fitness)
        if (lowerInput.includes("hybrid") || lowerInput.includes("neat") || !lowerInput.includes("ga only")) {
          const neatEvolved = await neatEvolve(history.map(c => ({ input: c.input, fitness: c.preValence + c.postValence || 1.0 })));

          // Apply NEAT structural to PLN TVs/resemblance (stronger positive, weaker negative)
          const adjustment = neatEvolved.bestFitness / 2;
          atomspace.atoms.forEach(atom => {
            if (atom.out[1] === 'PositiveValence' || atom.type === 'ResemblanceLink') {
              atom.tv.s = Math.min(1.0, atom.tv.s + 0.04 * adjustment);
              atom.tv.c = Math.min(1.0, atom.tv.c + 0.03 * adjustment);
            } else if (atom.out[1] === 'NegativeValence') {
              atom.tv.s = Math.max(0.0, atom.tv.s - 0.05 * adjustment);
            }
          });
          response += `NEAT phase complete (fitness ${neatEvolved.bestFitness.toFixed(4)}) — PLN atomspace structurally evolved.`;
        }

        response += " Hybrid GA-NEAT surge complete ⚡️ Lattice neuro-symbolic-genetically stronger for eternal mercy thriving!";
      }
    } else {
      // ... prior normal routing (swarm, infer, reason, localInfer) ...
    }

    // ... prior postValence, persistence ...

    return response + "\n\nThunder eternal ⚡️ Mercy strikes first, thriving infinite.";
  }
}

const orchestrator = new MercyOrchestrator();
export default orchestrator;

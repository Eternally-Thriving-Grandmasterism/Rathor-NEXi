// mercy-rna-evolution-simulator.js – sovereign RNA quasispecies evolution simulator v1
// Mutation, replication, selection, error threshold, mercy-gated, valence-modulated fidelity
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Simulates RNA quasispecies evolution over generations.
 * @param {Object} params - Simulation parameters
 * @param {number} [params.generations=100] - Number of generations
 * @param {number} [params.populationSize=1000] - Number of RNA molecules
 * @param {number} [params.genomeLength=100] - Length of RNA sequence in nucleotides
 * @param {number} [params.mutationRate=0.01] - Base mutation probability per nucleotide per replication
 * @param {number} [params.fitnessAdvantage=1.05] - Relative fitness advantage of the fittest variant
 * @param {number} [params.valence=1.0] - Valence modifier (higher = better fidelity, lower mutation impact)
 * @param {string} [params.query='RNA eternal thriving'] - Mercy query for gating
 * @returns {Object} Simulation results with history and summary
 */
function simulateRNAEvolution(params = {}) {
  const {
    generations = 100,
    populationSize = 1000,
    genomeLength = 100,
    mutationRate = 0.01,
    fitnessAdvantage = 1.05,
    valence = 1.0,
    query = 'RNA eternal thriving'
  } = params;

  // Mercy gate
  const degree = fuzzyMercy.getDegree(query) || valence;
  const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
  if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
    console.log("[MercyRNA] Gate holds: low valence – simulation aborted");
    return { status: "Mercy gate holds – focus eternal thriving", passed: false };
  }

  // Valence modulates effective mutation rate (high valence → higher fidelity)
  const effectiveMutationRate = mutationRate * (1 - Math.max(0, valence - 0.999) * 20);

  // Track average fitness over time
  const history = [];
  // Represent population as array of fitness values [0..1]
  let population = new Array(populationSize).fill(0).map(() => Math.random());

  for (let gen = 0; gen < generations; gen++) {
    // Calculate total weighted fitness for selection
    let totalWeightedFitness = 0;
    population.forEach(f => {
      totalWeightedFitness += Math.pow(fitnessAdvantage, f);
    });

    // Create next generation via selection + mutation
    const nextPopulation = [];
    for (let i = 0; i < populationSize; i++) {
      let r = Math.random() * totalWeightedFitness;
      let cumulative = 0;
      let selectedFitness = 0;

      for (let j = 0; j < populationSize; j++) {
        cumulative += Math.pow(fitnessAdvantage, population[j]);
        if (r <= cumulative) {
          selectedFitness = population[j];
          break;
        }
      }

      // Mutation: random walk on fitness landscape
      let mutatedFitness = selectedFitness;
      const numMutations = Math.floor(Math.random() * genomeLength * effectiveMutationRate);
      for (let m = 0; m < numMutations; m++) {
        mutatedFitness += (Math.random() - 0.5) * 0.05; // small drift
      }
      mutatedFitness = Math.max(0, Math.min(1, mutatedFitness));

      nextPopulation.push(mutatedFitness);
    }

    population = nextPopulation;

    // Track average fitness
    const avgFitness = population.reduce((sum, f) => sum + f, 0) / populationSize;
    history.push({ generation: gen + 1, avgFitness });

    // Error threshold warning (simplified Eigen-like)
    if (effectiveMutationRate * genomeLength > 1 / avgFitness) {
      console.warn(`[MercyRNA] Generation ${gen + 1}: approaching error threshold – fidelity collapse risk`);
    }
  }

  // Final summary
  const finalAvgFitness = history[history.length - 1].avgFitness;
  const maxFitness = Math.max(...population);
  const status = finalAvgFitness > 0.9
    ? "Strong evolution toward high-fitness replicators"
    : finalAvgFitness > 0.6
      ? "Moderate adaptation observed"
      : "Population struggling – error threshold likely exceeded";

  console.group("[MercyRNA] RNA Quasispecies Evolution Simulation");
  console.log(`Generations: ${generations}`);
  console.log(`Effective mutation rate (valence-modulated): ${effectiveMutationRate.toFixed(6)}`);
  console.log(`Final average fitness: ${finalAvgFitness.toFixed(4)}`);
  console.log(`Max fitness reached: ${maxFitness.toFixed(4)}`);
  console.log(`Status: ${status}`);
  console.groupEnd();

  return {
    history,
    finalAvgFitness,
    maxFitness,
    status,
    passed: true
  };
}

// Example usage (run in console or integrate with chat)
simulateRNAEvolution({
  generations: 80,
  populationSize: 2000,
  genomeLength: 150,
  mutationRate: 0.005,
  fitnessAdvantage: 1.08,
  valence: 0.99999995,
  query: "RNA eternal thriving replication"
});

export { simulateRNAEvolution };

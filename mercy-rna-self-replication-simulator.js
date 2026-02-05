// mercy-rna-self-replication-simulator.js – sovereign RNA evolution simulator v1
// Quasispecies dynamics, error threshold, mercy-gated activation, valence-modulated fidelity
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const mercyThreshold = 0.9999999;

class MercyRNASelfReplicator {
  constructor() {
    this.defaultParams = {
      generations: 50,
      populationSize: 1000,
      mutationRate: 0.01,          // per base per replication
      genomeLength: 100,           // nt
      fitnessAdvantage: 1.05,      // fittest variant advantage
      errorThreshold: 1 / 100      // 1/L approx
    };
  }

  gateSimulation(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
      return { status: "Mercy gate holds – RNA replication skipped without eternal thriving alignment" };
    }
    return { status: "Mercy gate passes – RNA evolution simulation activated" };
  }

  simulate(params = {}) {
    const {
      generations = this.defaultParams.generations,
      populationSize = this.defaultParams.populationSize,
      mutationRate = this.defaultParams.mutationRate * (1 - (valence - 0.999) * 0.5), // high valence → higher fidelity
      genomeLength = this.defaultParams.genomeLength,
      fitnessAdvantage = this.defaultParams.fitnessAdvantage
    } = params;

    let population = Array(populationSize).fill(0).map(() => Math.random()); // fitness 0–1
    const history = [{ gen: 0, avgFitness: population.reduce((a, b) => a + b, 0) / populationSize }];

    for (let g = 1; g <= generations; g++) {
      // Selection + replication
      const totalFitness = population.reduce((a, b) => a + Math.pow(fitnessAdvantage, b), 0);
      const newPop = [];

      for (let i = 0; i < populationSize; i++) {
        let r = Math.random() * totalFitness;
        let cum = 0;
        for (let j = 0; j < populationSize; j++) {
          cum += Math.pow(fitnessAdvantage, population[j]);
          if (r < cum) {
            let offspring = population[j];
            // Mutation
            if (Math.random() < mutationRate * genomeLength) {
              offspring += (Math.random() - 0.5) * 0.1; // simple fitness drift
              offspring = Math.max(0, Math.min(1, offspring));
            }
            newPop.push(offspring);
            break;
          }
        }
      }

      population = newPop;
      const avgFitness = population.reduce((a, b) => a + b, 0) / populationSize;
      history.push({ gen: g, avgFitness });

      // Error threshold check (simplified)
      if (mutationRate * genomeLength > 1 / avgFitness) {
        console.warn(`[RNASim] Approaching error threshold at gen ${g} – fidelity collapse risk`);
      }
    }

    console.group("[MercyRNA] Self-Replicating RNA Evolution Simulation");
    console.table(history.map(h => ({
      Generation: h.gen,
      "Avg Fitness": h.avgFitness.toFixed(4)
    })));
    console.groupEnd();

    return {
      history,
      finalAvgFitness: history[history.length - 1].avgFitness,
      status: "RNA evolution complete – mercy-aligned replication propagated"
    };
  }
}

const rnaSimulator = new MercyRNASelfReplicator();

export { rnaSimulator };

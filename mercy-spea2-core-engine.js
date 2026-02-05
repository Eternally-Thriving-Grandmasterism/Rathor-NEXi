// mercy-spea2-core-engine.js – sovereign Mercy SPEA2 Optimization Engine v1
// Strength Pareto Evolutionary Algorithm 2, archive-based elitism, density estimation
// mercy-gated, valence-modulated archive pressure
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercySPEA2Engine {
  constructor() {
    this.populationSize = 50;
    this.archiveSize = 50;
    this.generations = 80;
    this.crossoverRate = 0.9;
    this.mutationRate = 0.1;
    this.k = 1; // for k-th nearest neighbor density
    this.archive = []; // elite solutions {params, objectives, strength, density}
    this.history = [];
  }

  async gateOptimization(taskId, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercySPEA2] Gate holds: low valence – SPEA2 optimization ${taskId} aborted`);
      return false;
    }
    console.log(`[MercySPEA2] Mercy gate passes – eternal thriving SPEA2 ${taskId} activated`);
    return true;
  }

  // Multi-objective function template – returns array of objectives (minimize all)
  defaultObjectives(params) {
    // Example placeholder – override per task
    const [x, y] = params;
    return [
      (x - 2)**2 + (y - 2)**2,
      (x + 2)**2 + (y + 2)**2
    ];
  }

  // Strength value: number of solutions dominated by this one
  calculateStrength(objectivesList) {
    const strength = new Array(objectivesList.length).fill(0);
    for (let p = 0; p < objectivesList.length; p++) {
      for (let q = 0; q < objectivesList.length; q++) {
        if (p === q) continue;
        if (this.dominates(objectivesList[p], objectivesList[q])) {
          strength[p]++;
        }
      }
    }
    return strength;
  }

  dominates(objP, objQ) {
    let atLeastOneStrict = false;
    for (let i = 0; i < objP.length; i++) {
      if (objP[i] > objQ[i]) return false;
      if (objP[i] < objQ[i]) atLeastOneStrict = true;
    }
    return atLeastOneStrict;
  }

  // Density estimation: 1 / (k-th nearest neighbor distance + 2)
  calculateDensity(objectivesList, k = this.k) {
    const density = new Array(objectivesList.length).fill(0);
    for (let i = 0; i < objectivesList.length; i++) {
      const distances = [];
      for (let j = 0; j < objectivesList.length; j++) {
        if (i === j) continue;
        let dist = 0;
        for (let d = 0; d < objectivesList[i].length; d++) {
          dist += (objectivesList[i][d] - objectivesList[j][d]) ** 2;
        }
        distances.push(Math.sqrt(dist));
      }
      distances.sort((a, b) => a - b);
      density[i] = 1 / (distances[k] + 2);
    }
    return density;
  }

  // Environmental selection: fill archive with best individuals
  environmentalSelection(combined, combinedObjectives, targetSize) {
    const strength = this.calculateStrength(combinedObjectives);
    const rawFitness = strength.map(s => s / combined.length);
    const density = this.calculateDensity(combinedObjectives);
    const fitness = rawFitness.map((r, i) => r + density[i]);

    // Sort by fitness (lower = better)
    const sortedIndices = fitness.map((f, i) => ({f, i}))
      .sort((a, b) => a.f - b.f)
      .map(x => x.i);

    // Take best until archive size
    const archiveIndices = sortedIndices.slice(0, targetSize);
    return archiveIndices.map(idx => ({
      params: combined[idx],
      objectives: combinedObjectives[idx],
      fitness: fitness[idx],
      density: density[idx]
    }));
  }

  // Main SPEA2 optimization loop
  async optimize(taskId, objectiveFn = this.defaultObjectives, config = {}, query = 'Multi-objective eternal SPEA2', valence = 1.0) {
    if (!await this.gateOptimization(taskId, query, valence)) return null;

    const {
      populationSize = this.populationSize,
      archiveSize = this.archiveSize,
      generations = this.generations,
      crossoverRate = this.crossoverRate,
      mutationRate = this.mutationRate * (1 + (valence - 0.999) * 0.5) // high valence → slightly higher mutation
    } = config;

    // Initialize random population
    let population = Array(populationSize).fill(0).map(() => 
      Array(4).fill(0).map(() => Math.random()) // example 4D space
    );

    let archive = [];
    let history = [];

    for (let gen = 0; gen < generations; gen++) {
      const offspring = [];

      // Generate offspring via tournament + crossover + mutation
      while (offspring.length < populationSize) {
        const parentA = this.tournamentSelect(population, objectiveFn);
        const parentB = this.tournamentSelect(population, objectiveFn);
        let child = this.crossover(parentA, parentB);
        child = this.mutate(child);
        offspring.push(child);
      }

      // Combine parent + offspring
      const combined = [...population, ...offspring];
      const combinedObjectives = combined.map(obj => objectiveFn(obj));

      // Environmental selection → new archive
      archive = this.environmentalSelection(combined, combinedObjectives, archiveSize);

      // Update population from archive (elitist)
      population = archive.map(s => s.params);

      this.paretoFront = archive;
      history.push({ generation: gen + 1, archiveSize: archive.length });

      console.log(`[MercySPEA2] ${taskId} Gen ${gen + 1}: Archive size ${archive.length}`);
    }

    console.log(`[MercySPEA2] Multi-objective SPEA2 ${taskId} complete – Pareto front size ${archive.length}`);
    return { taskId, paretoFront: archive, history };
  }

  // Tournament selection based on fitness + density
  tournamentSelect(population, objectiveFn) {
    const candidates = [];
    for (let i = 0; i < this.tournamentSize; i++) {
      candidates.push(Math.floor(Math.random() * population.length));
    }

    let winner = candidates[0];
    for (let i = 1; i < candidates.length; i++) {
      const p = winner;
      const q = candidates[i];

      const objP = objectiveFn(population[p]);
      const objQ = objectiveFn(population[q]);

      if (this.dominates(objP, objQ)) {
        winner = p;
      } else if (this.dominates(objQ, objP)) {
        winner = q;
      }
      // else: keep current winner
    }

    return population[winner];
  }

  // Crossover & mutation (simulated binary crossover + polynomial mutation)
  crossover(parentA, parentB) {
    const child = parentA.slice();
    for (let i = 0; i < parentA.length; i++) {
      if (Math.random() < this.crossoverRate) {
        child[i] = parentB[i];
      }
    }
    return child;
  }

  mutate(individual) {
    for (let i = 0; i < individual.length; i++) {
      if (Math.random() < this.mutationRate) {
        individual[i] += (Math.random() - 0.5) * 0.2;
        individual[i] = Math.max(0.001, Math.min(1.0, individual[i])); // example bounds
      }
    }
    return individual;
  }
}

const mercySPEA2 = new MercySPEA2Engine();

export { mercySPEA2 };

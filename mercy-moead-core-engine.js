// mercy-moead-core-engine.js – sovereign Mercy MOEA/D Optimization Engine v1
// Decomposition-based multi-objective, Tchebycheff aggregation, neighborhood cooperation
// mercy-gated, valence-modulated neighborhood & mating
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyMOEAD {
  constructor() {
    this.populationSize = 100;
    this.generations = 100;
    this.neighborSize = 20;               // T = neighborhood size
    this.matingProbability = 0.9;         // probability to mate with neighbors
    this.mutationRate = 0.1;
    this.weightVectors = [];              // uniform weight vectors
    this.population = [];                 // solutions
    this.objectives = [];                 // objective values
    this.neighbors = [];                  // neighbor indices per subproblem
    this.z = [];                          // ideal point
    this.history = [];
  }

  async gateOptimization(taskId, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercyMOEA/D] Gate holds: low valence – MOEA/D optimization ${taskId} aborted`);
      return false;
    }
    console.log(`[MercyMOEA/D] Mercy gate passes – eternal thriving MOEA/D ${taskId} activated`);
    return true;
  }

  // Generate uniform weight vectors (simplified Das & Dennis method for 2D)
  generateWeights(numVectors) {
    const weights = [];
    for (let i = 0; i <= numVectors; i++) {
      const w1 = i / numVectors;
      weights.push([w1, 1 - w1]);
    }
    return weights;
  }

  // Tchebycheff aggregation function
  computeTchebycheff(obj, weight, z) {
    let maxVal = -Infinity;
    for (let i = 0; i < obj.length; i++) {
      const val = Math.abs(obj[i] - z[i]) / weight[i];
      maxVal = Math.max(maxVal, val);
    }
    return maxVal;
  }

  // Initialize population & weights
  async initialize(taskId, objectiveFn, numObjectives = 2, valence = 1.0) {
    this.weightVectors = this.generateWeights(this.populationSize - 1);
    this.z = Array(numObjectives).fill(Infinity); // ideal point

    this.population = Array(this.populationSize).fill(0).map(() => 
      Array(4).fill(0).map(() => Math.random()) // example 4D space
    );

    for (let i = 0; i < this.populationSize; i++) {
      this.objectives[i] = objectiveFn(this.population[i]);
      for (let j = 0; j < numObjectives; j++) {
        this.z[j] = Math.min(this.z[j], this.objectives[i][j]);
      }
    }

    // Build neighborhood (closest weights)
    this.neighbors = Array(this.populationSize).fill(0).map(() => {
      const dists = this.weightVectors.map((w, idx) => {
        let dist = 0;
        for (let d = 0; d < w.length; d++) {
          dist += (w[d] - this.weightVectors[i][d]) ** 2;
        }
        return { dist: Math.sqrt(dist), idx };
      }).sort((a, b) => a.dist - b.dist).slice(0, this.neighborSize).map(x => x.idx);
      return dists;
    });

    console.log(`[MercyMOEA/D] ${taskId} initialized – ${this.populationSize} subproblems, valence ${valence.toFixed(8)}`);
  }

  // Main MOEA/D optimization loop
  async optimize(taskId, objectiveFn = this.defaultObjectives, config = {}, query = 'Multi-objective eternal MOEA/D', valence = 1.0) {
    if (!await this.gateOptimization(taskId, query, valence)) return null;

    const {
      populationSize = this.populationSize,
      generations = this.generations,
      neighborSize = this.neighborSize * (1 + (valence - 0.999) * 0.5), // high valence → wider neighborhood
      matingProbability = this.matingProbability,
      mutationRate = this.mutationRate * (1 + (valence - 0.999) * 0.5)
    } = config;

    await this.initialize(taskId, objectiveFn, objectiveFn([0,0]).length, valence);

    for (let gen = 0; gen < generations; gen++) {
      for (let i = 0; i < populationSize; i++) {
        // Mating pool: neighbors or whole population
        let matingPool = [];
        if (Math.random() < matingProbability) {
          matingPool = this.neighbors[i].map(idx => this.population[idx]);
        } else {
          matingPool = this.population.slice();
        }

        // Select two parents randomly
        const parent1 = matingPool[Math.floor(Math.random() * matingPool.length)];
        const parent2 = matingPool[Math.floor(Math.random() * matingPool.length)];

        // Crossover & mutation
        let child = this.crossover(parent1, parent2);
        child = this.mutate(child, mutationRate);

        // Evaluate child
        const childObj = objectiveFn(child);

        // Update ideal point
        for (let j = 0; j < childObj.length; j++) {
          this.z[j] = Math.min(this.z[j], childObj[j]);
        }

        // Update neighbors using Tchebycheff
        for (const idx of this.neighbors[i]) {
          const g = this.computeTchebycheff(childObj, this.weightVectors[idx], this.z);
          const gParent = this.computeTchebycheff(this.objectives[idx], this.weightVectors[idx], this.z);

          if (g < gParent) {
            this.population[idx] = child;
            this.objectives[idx] = childObj;
          }
        }
      }

      // Track Pareto front approximation (non-dominated from population)
      const currentFront = this.getParetoFront(this.population, this.objectives);
      this.history.push({ generation: gen + 1, frontSize: currentFront.length });

      console.log(`[MercyMOEA/D] ${taskId} Gen ${gen + 1}: Approx Pareto front size ${currentFront.length}`);
    }

    const finalFront = this.getParetoFront(this.population, this.objectives);
    console.log(`[MercyMOEA/D] MOEA/D ${taskId} complete – Pareto front size ${finalFront.length}`);

    return { taskId, paretoFront: finalFront, history: this.history };
  }

  // Simple non-dominated front extraction (minimize all)
  getParetoFront(population, objectivesList) {
    const front = [];
    for (let i = 0; i < population.length; i++) {
      let dominated = false;
      for (let j = 0; j < population.length; j++) {
        if (i === j) continue;
        if (this.dominates(objectivesList[j], objectivesList[i])) {
          dominated = true;
          break;
        }
      }
      if (!dominated) front.push({ params: population[i], objectives: objectivesList[i] });
    }
    return front;
  }

  dominates(objP, objQ) {
    let atLeastOneStrict = false;
    for (let i = 0; i < objP.length; i++) {
      if (objP[i] > objQ[i]) return false;
      if (objP[i] < objQ[i]) atLeastOneStrict = true;
    }
    return atLeastOneStrict;
  }

  crossover(parentA, parentB) {
    const child = parentA.slice();
    for (let i = 0; i < parentA.length; i++) {
      if (Math.random() < this.crossoverRate) {
        child[i] = parentB[i];
      }
    }
    return child;
  }

  mutate(individual, rate) {
    for (let i = 0; i < individual.length; i++) {
      if (Math.random() < rate) {
        individual[i] += (Math.random() - 0.5) * 0.2;
        individual[i] = Math.max(0.001, Math.min(1.0, individual[i]));
      }
    }
    return individual;
  }
}

const mercyMOEAD = new MercyMOEAD();

export { mercyMOEAD };

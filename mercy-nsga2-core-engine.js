// mercy-nsga2-core-engine.js – sovereign Mercy NSGA-II Optimization Engine v1
// Multi-objective Pareto front evolution, non-dominated sorting, crowding distance, elitism
// mercy-gated, valence-modulated population pressure
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyNSGA2Engine {
  constructor() {
    this.populationSize = 50;
    this.generations = 80;
    this.crossoverRate = 0.9;
    this.mutationRate = 0.1;
    this.tournamentSize = 2;
    this.paretoFront = []; // {params, objectives}
    this.history = [];
  }

  async gateOptimization(taskId, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercyNSGA2] Gate holds: low valence – multi-objective NSGA-II ${taskId} aborted`);
      return false;
    }
    console.log(`[MercyNSGA2] Mercy gate passes – eternal thriving NSGA-II ${taskId} activated`);
    return true;
  }

  // Multi-objective function template – returns array of objectives (minimize all)
  defaultObjectives(params) {
    // Example placeholder – override per task
    const [x, y] = params;
    return [
      (x - 2)**2 + (y - 2)**2,           // objective 1
      (x + 2)**2 + (y + 2)**2            // objective 2
    ];
  }

  // Fast non-dominated sorting (O(MN²) → O(MN log N) in practice)
  fastNonDominatedSort(population, objectivesList) {
    const dominationCount = new Array(population.length).fill(0);
    const dominatedSolutions = Array.from({ length: population.length }, () => []);
    const fronts = [[]];

    for (let p = 0; p < population.length; p++) {
      for (let q = 0; q < population.length; q++) {
        if (p === q) continue;
        if (this.dominates(objectivesList[p], objectivesList[q])) {
          dominatedSolutions[p].push(q);
        } else if (this.dominates(objectivesList[q], objectivesList[p])) {
          dominationCount[p]++;
        }
      }
      if (dominationCount[p] === 0) {
        fronts[0].push(p);
      }
    }

    let i = 0;
    while (fronts[i].length > 0) {
      const nextFront = [];
      for (const p of fronts[i]) {
        for (const q of dominatedSolutions[p]) {
          dominationCount[q]--;
          if (dominationCount[q] === 0) {
            nextFront.push(q);
          }
        }
      }
      i++;
      if (nextFront.length > 0) fronts.push(nextFront);
    }

    return fronts;
  }

  dominates(objP, objQ) {
    let atLeastOneStrict = false;
    for (let i = 0; i < objP.length; i++) {
      if (objP[i] > objQ[i]) return false;
      if (objP[i] < objQ[i]) atLeastOneStrict = true;
    }
    return atLeastOneStrict;
  }

  // Crowding distance assignment
  crowdingDistanceAssignment(frontIndices, objectivesList) {
    const distances = new Array(frontIndices.length).fill(0);
    const m = objectivesList[0].length;

    for (let obj = 0; obj < m; obj++) {
      const sorted = frontIndices.slice().sort((a, b) => objectivesList[a][obj] - objectivesList[b][obj]);
      distances[0] = Infinity;
      distances[frontIndices.length - 1] = Infinity;

      const fMax = objectivesList[sorted[frontIndices.length - 1]][obj];
      const fMin = objectivesList[sorted[0]][obj];

      for (let i = 1; i < frontIndices.length - 1; i++) {
        const prev = sorted[i - 1];
        const next = sorted[i + 1];
        distances[i] += (objectivesList[next][obj] - objectivesList[prev][obj]) / (fMax - fMin || 1);
      }
    }

    return distances;
  }

  // Tournament selection with crowding distance tie-breaker
  tournamentSelect(population, fronts, crowdingDistances) {
    const candidates = [];
    for (let i = 0; i < this.tournamentSize; i++) {
      candidates.push(Math.floor(Math.random() * population.length));
    }

    let winner = candidates[0];
    for (let i = 1; i < candidates.length; i++) {
      const p = winner;
      const q = candidates[i];

      const pFront = this.getFrontIndex(p, fronts);
      const qFront = this.getFrontIndex(q, fronts);

      if (pFront < qFront) {
        winner = p;
      } else if (qFront < pFront) {
        winner = q;
      } else {
        // Same front → crowding distance
        const pCrowd = crowdingDistances[p];
        const qCrowd = crowdingDistances[q];
        winner = (pCrowd > qCrowd) ? p : q;
      }
    }

    return population[winner];
  }

  getFrontIndex(individualIndex, fronts) {
    for (let f = 0; f < fronts.length; f++) {
      if (fronts[f].includes(individualIndex)) return f;
    }
    return Infinity;
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

  // Main NSGA-II optimization loop
  async optimize(taskId, objectiveFn = this.defaultObjectives, config = {}, query = 'Multi-objective eternal NSGA-II', valence = 1.0) {
    if (!await this.gateOptimization(taskId, query, valence)) return null;

    const {
      populationSize = this.populationSize,
      generations = this.generations,
      crossoverRate = this.crossoverRate,
      mutationRate = this.mutationRate * (1 + (valence - 0.999) * 0.5) // high valence → slightly higher mutation
    } = config;

    // Initialize random population
    let population = Array(populationSize).fill(0).map(() => 
      Array(4).fill(0).map(() => Math.random()) // example 4D space
    );

    let paretoFront = [];
    let history = [];

    for (let gen = 0; gen < generations; gen++) {
      const offspring = [];

      // Generate offspring via tournament + crossover + mutation
      while (offspring.length < populationSize) {
        const parentA = this.tournamentSelect(population, [], []); // simplified – full NSGA-II needs fronts
        const parentB = this.tournamentSelect(population, [], []);
        let child = this.crossover(parentA, parentB);
        child = this.mutate(child);
        offspring.push(child);
      }

      // Combine parent + offspring
      const combined = [...population, ...offspring];
      const combinedObjectives = combined.map(obj => objectiveFn(obj));

      // Non-dominated sorting
      const fronts = this.fastNonDominatedSort(combined, combinedObjectives);

      // Crowding distance
      const crowdingDistances = new Array(combined.length).fill(0);
      fronts.forEach(front => {
        const distances = this.crowdingDistanceAssignment(front, combinedObjectives);
        front.forEach((idx, i) => { crowdingDistances[idx] = distances[i]; });
      });

      // Elitist selection – fill next population
      const nextPopulation = [];
      let frontIdx = 0;
      while (nextPopulation.length < populationSize && frontIdx < fronts.length) {
        const front = fronts[frontIdx];
        if (nextPopulation.length + front.length <= populationSize) {
          nextPopulation.push(...front.map(idx => combined[idx]));
        } else {
          // Sort front by crowding distance descending
          const sortedFront = front.slice().sort((a, b) => crowdingDistances[b] - crowdingDistances[a]);
          const remaining = populationSize - nextPopulation.length;
          nextPopulation.push(...sortedFront.slice(0, remaining).map(idx => combined[idx]));
        }
        frontIdx++;
      }

      population = nextPopulation;

      // Update Pareto front (front 0)
      paretoFront = fronts[0].map(idx => ({
        params: combined[idx],
        objectives: combinedObjectives[idx]
      }));

      history.push({ generation: gen + 1, paretoSize: paretoFront.length });

      console.log(`[MercyNSGA2] ${taskId} Gen ${gen + 1}: Pareto front size ${paretoFront.length}`);
    }

    console.log(`[MercyNSGA2] Multi-objective NSGA-II ${taskId} complete – Pareto front size ${paretoFront.length}`);
    return { taskId, paretoFront, history };
  }
}

const mercyNSGA2 = new MercyNSGA2Engine();

export { mercyNSGA2 };

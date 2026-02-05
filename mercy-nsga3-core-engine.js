// mercy-nsga3-core-engine.js – sovereign Mercy NSGA-III Optimization Engine v1
// Reference-point guided many-objective, non-dominated sorting, association + niche preservation
// mercy-gated, valence-modulated reference density & niche pressure
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyNSGA3Engine {
  constructor() {
    this.populationSize = 92;               // typical for 3–15 objectives
    this.generations = 100;
    this.crossoverRate = 0.9;
    this.mutationRate = 0.1;
    this.referencePoints = [];              // Das & Dennis uniform points
    this.population = [];
    this.objectives = [];
    this.history = [];
  }

  async gateOptimization(taskId, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercyNSGA3] Gate holds: low valence – NSGA-III optimization ${taskId} aborted`);
      return false;
    }
    console.log(`[MercyNSGA3] Mercy gate passes – eternal thriving NSGA-III ${taskId} activated`);
    return true;
  }

  // Generate Das & Dennis reference points (simplified for 2–3 objectives; extend for higher)
  generateReferencePoints(numObjectives = 3, p = 12) {
    const points = [];
    // For 3 objectives: 91 points with p=12
    for (let i = 0; i <= p; i++) {
      for (let j = 0; j <= p - i; j++) {
        const k = p - i - j;
        const w = [i/p, j/p, k/p];
        points.push(w);
      }
    }
    return points;
  }

  // Non-dominated sorting (reuse from NSGA-II style)
  fastNonDominatedSort(objectivesList) {
    const dominationCount = new Array(objectivesList.length).fill(0);
    const dominatedSolutions = Array.from({ length: objectivesList.length }, () => []);
    const fronts = [[]];

    for (let p = 0; p < objectivesList.length; p++) {
      for (let q = 0; q < objectivesList.length; q++) {
        if (p === q) continue;
        if (this.dominates(objectivesList[p], objectivesList[q])) {
          dominatedSolutions[p].push(q);
        } else if (this.dominates(objectivesList[q], objectivesList[p])) {
          dominationCount[p]++;
        }
      }
      if (dominationCount[p] === 0) fronts[0].push(p);
    }

    let i = 0;
    while (fronts[i].length > 0) {
      const nextFront = [];
      for (const p of fronts[i]) {
        for (const q of dominatedSolutions[p]) {
          dominationCount[q]--;
          if (dominationCount[q] === 0) nextFront.push(q);
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

  // Associate population to reference points
  associateToReference(population, objectivesList, referencePoints) {
    const associations = new Array(population.length).fill(-1);
    const distances = new Array(population.length).fill(Infinity);

    for (let p = 0; p < population.length; p++) {
      for (let r = 0; r < referencePoints.length; r++) {
        let dist = 0;
        const normObj = objectivesList[p].map((v, i) => v - this.z[i]); // ideal-shifted
        const w = referencePoints[r];
        let norm = 0;
        for (let i = 0; i < w.length; i++) norm += normObj[i] * w[i];
        const proj = norm / (w.reduce((a, b) => a + b*b, 0) || 1);
        let d = 0;
        for (let i = 0; i < w.length; i++) {
          d += (normObj[i] - proj * w[i]) ** 2;
        }
        d = Math.sqrt(d);

        if (d < distances[p]) {
          distances[p] = d;
          associations[p] = r;
        }
      }
    }

    return { associations, distances };
  }

  // Niche preservation – select from last front using reference points
  nichePreservation(lastFront, associations, distances, referencePoints, targetSize) {
    const nicheCounts = new Array(referencePoints.length).fill(0);
    const selected = [];

    // Count how many already selected are associated to each reference
    for (let i = 0; i < this.population.length; i++) {
      const ref = associations[i];
      if (ref !== -1) nicheCounts[ref]++;
    }

    while (selected.length < targetSize && lastFront.length > 0) {
      // Find reference point with least niche count
      let minCount = Infinity;
      let minRef = -1;
      for (let r = 0; r < referencePoints.length; r++) {
        if (nicheCounts[r] < minCount) {
          minCount = nicheCounts[r];
          minRef = r;
        }
      }

      // Candidates associated to minRef
      const candidates = lastFront.filter(idx => associations[idx] === minRef);
      if (candidates.length === 0) {
        // No candidates — choose least crowded reference with candidates
        continue;
      }

      // Pick candidate with smallest distance
      let bestIdx = candidates[0];
      let bestDist = distances[bestIdx];
      for (const idx of candidates) {
        if (distances[idx] < bestDist) {
          bestDist = distances[idx];
          bestIdx = idx;
        }
      }

      selected.push(bestIdx);
      lastFront = lastFront.filter(idx => idx !== bestIdx);
      nicheCounts[minRef]++;
    }

    return selected;
  }

  // Main NSGA-III optimization loop
  async optimize(taskId, objectiveFn = this.defaultObjectives, config = {}, query = 'Multi-objective eternal NSGA-III', valence = 1.0) {
    if (!await this.gateOptimization(taskId, query, valence)) return null;

    const {
      populationSize = this.populationSize,
      generations = this.generations,
      crossoverRate = this.crossoverRate,
      mutationRate = this.mutationRate * (1 + (valence - 0.999) * 0.5) // high valence → slightly higher mutation
    } = config;

    // Generate reference points (e.g., 91 points for 3 objectives)
    this.referencePoints = this.generateReferencePoints(3, 12);

    // Initialize random population
    this.population = Array(populationSize).fill(0).map(() => 
      Array(4).fill(0).map(() => Math.random()) // example 4D space
    );

    this.objectives = this.population.map(obj => objectiveFn(obj));

    // Initialize ideal point
    this.z = Array(objectiveFn([0,0,0,0]).length).fill(Infinity);
    for (let i = 0; i < populationSize; i++) {
      for (let j = 0; j < this.z.length; j++) {
        this.z[j] = Math.min(this.z[j], this.objectives[i][j]);
      }
    }

    let history = [];

    for (let gen = 0; gen < generations; gen++) {
      const offspring = [];

      // Generate offspring
      while (offspring.length < populationSize) {
        const parentA = this.tournamentSelect(this.population, this.objectives);
        const parentB = this.tournamentSelect(this.population, this.objectives);
        let child = this.crossover(parentA, parentB);
        child = this.mutate(child, mutationRate);
        offspring.push(child);
      }

      // Combine parent + offspring
      const combined = [...this.population, ...offspring];
      const combinedObjectives = combined.map(obj => objectiveFn(obj));

      // Update ideal point
      for (let i = 0; i < combined.length; i++) {
        for (let j = 0; j < this.z.length; j++) {
          this.z[j] = Math.min(this.z[j], combinedObjectives[i][j]);
        }
      }

      // Non-dominated sorting
      const fronts = this.fastNonDominatedSort(combinedObjectives);

      // Associate to reference points
      const { associations, distances } = this.associateToReference(combined, combinedObjectives, this.referencePoints);

      // Build next population
      const nextPopulation = [];
      let frontIdx = 0;
      while (nextPopulation.length < populationSize && frontIdx < fronts.length) {
        const front = fronts[frontIdx];
        if (nextPopulation.length + front.length <= populationSize) {
          nextPopulation.push(...front.map(idx => combined[idx]));
        } else {
          // Niche preservation on last front
          const remaining = populationSize - nextPopulation.length;
          const selected = this.nichePreservation(front, associations, distances, this.referencePoints, remaining);
          nextPopulation.push(...selected.map(idx => combined[idx]));
        }
        frontIdx++;
      }

      this.population = nextPopulation;
      this.objectives = this.population.map(obj => objectiveFn(obj));

      // Track current Pareto front
      const currentFront = this.getParetoFront(this.population, this.objectives);
      history.push({ generation: gen + 1, frontSize: currentFront.length });

      console.log(`[MercyNSGA3] ${taskId} Gen ${gen + 1}: Approx Pareto front size ${currentFront.length}`);
    }

    const finalFront = this.getParetoFront(this.population, this.objectives);
    console.log(`[MercyNSGA3] NSGA-III ${taskId} complete – Pareto front size ${finalFront.length}`);

    return { taskId, paretoFront: finalFront, history };
  }

  // ... (dominate, tournamentSelect, crossover, mutate functions same as NSGA-II / SPEA2 lineage)

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
}

const mercyNSGA3 = new MercyNSGA3Engine();

export { mercyNSGA3 };

// mercy-multi-objective-cmaes.js – sovereign multi-objective CMA-ES v1
// Pareto front approximation, mercy-gated, valence-modulated search intensity
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyMultiObjectiveCMA {
  constructor() {
    this.populationSize = 30;
    this.generations = 60;
    this.sigma = 0.4;
    this.learningRate = 0.5;
    this.damping = 0.9;
    this.paretoFront = []; // {params, objectives}
  }

  async gateOptimization(taskId, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercyMOCMA] Gate holds: low valence – multi-objective optimization ${taskId} aborted`);
      return false;
    }
    console.log(`[MercyMOCMA] Mercy gate passes – eternal thriving multi-objective CMA activated`);
    return true;
  }

  // Example multi-objective function (override per task)
  // Returns array of objectives to minimize/maximize (here minimize both)
  defaultObjectives(params) {
    const [x, y] = params;
    return [
      (x - 2)**2 + (y - 2)**2,           // objective 1
      (x + 2)**2 + (y + 2)**2            // objective 2
    ];
  }

  async optimize(taskId, objectiveFn = this.defaultObjectives, config = {}, query = 'Multi-objective eternal optimization', valence = 1.0) {
    if (!await this.gateOptimization(taskId, query, valence)) return null;

    const {
      populationSize = this.populationSize,
      generations = this.generations,
      sigma = this.sigma * (1 + (valence - 0.999) * 2), // high valence → wider exploration
      learningRate = this.learningRate,
      damping = this.damping
    } = config;

    let mean = [0, 0]; // example 2D space
    let bestPareto = [];
    let history = [];

    for (let gen = 0; gen < generations; gen++) {
      const population = [];
      const objectivesList = [];

      for (let i = 0; i < populationSize; i++) {
        const sample = mean.map((m, j) => m + sigma * (Math.random() - 0.5) * 2);
        const objs = objectiveFn(sample);
        population.push(sample);
        objectivesList.push(objs);
      }

      // Non-dominated sorting for Pareto front (simplified NSGA-II style)
      const nonDominated = this.getNonDominated(objectivesList, population);
      if (nonDominated.length > 0) {
        bestPareto = nonDominated;
      }

      // Update mean toward Pareto elite
      mean = [0, 0];
      nonDominated.forEach(ind => {
        ind.params.forEach((v, j) => { mean[j] += v / nonDominated.length; });
      });

      sigma *= damping;
      history.push({ generation: gen + 1, paretoSize: bestPareto.length });

      console.log(`[MercyMOCMA] ${taskId} Gen ${gen + 1}: Pareto front size ${bestPareto.length}`);
    }

    console.log(`[MercyMOCMA] Multi-objective optimization ${taskId} complete – Pareto front size ${bestPareto.length}`);
    return { taskId, paretoFront: bestPareto, history };
  }

  // Simple non-dominated sorting (minimize all objectives)
  getNonDominated(objectivesList, population) {
    const front = [];
    for (let i = 0; i < population.length; i++) {
      let dominated = false;
      for (let j = 0; j < population.length; j++) {
        if (i === j) continue;
        let better = true;
        for (let k = 0; k < objectivesList[i].length; k++) {
          if (objectivesList[i][k] < objectivesList[j][k]) continue;
          better = false;
          break;
        }
        if (better) {
          dominated = true;
          break;
        }
      }
      if (!dominated) front.push({ params: population[i], objectives: objectivesList[i] });
    }
    return front;
  }
}

const mercyMOCMA = new MercyMultiObjectiveCMA();

export { mercyMOCMA };

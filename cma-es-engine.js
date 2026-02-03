// cma-es-engine.js – sovereign client-side Covariance Matrix Adaptation Evolution Strategy
// MIT License – Autonomicity Games Inc. 2026

class CMAESIndividual {
  constructor(dimension) {
    this.params = new Float64Array(dimension);
    for (let i = 0; i < dimension; i++) {
      this.params[i] = Math.random() * 2 - 1; // [-1, 1] init
    }
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  copy() {
    const copy = new CMAESIndividual(this.params.length);
    copy.params.set(this.params);
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }
}

class CMAES {
  constructor(dimension = 10, populationSize = 50, generations = 80) {
    this.dimension = dimension;
    this.lambda = populationSize;   // offspring population size
    this.mu = Math.floor(this.lambda / 2); // parents
    this.generations = generations;

    // Strategy parameters (default CMA-ES settings)
    this.mean = new Float64Array(dimension); // initial mean
    this.sigma = 0.3;                        // global step-size
    this.C = this.identityMatrix(dimension); // covariance matrix
    this.ps = new Float64Array(dimension);   // evolution path (step-size)
    this.pc = new Float64Array(dimension);   // evolution path (covariance)
    this.cc = 4 / (dimension + 4);           // learning rate for pc
    this.cs = 4 / (dimension + 4);           // learning rate for ps
    this.c1 = 2 / ((dimension + 1.3)**2 + this.mu); // rank-1 update
    this.cmu = Math.min(1 - this.c1, 2 * (this.mu - 2 + 1 / this.mu) / ((dimension + 2)**2 + this.mu));
    this.damps = 1 + 2 * Math.max(0, Math.sqrt((this.mu - 1) / (dimension + 1)) - 1) + this.cs;

    this.weights = new Float64Array(this.mu);
    let sumw = 0;
    for (let i = 0; i < this.mu; i++) {
      this.weights[i] = Math.log(this.mu + 0.5) - Math.log(i + 1);
      sumw += this.weights[i];
    }
    for (let i = 0; i < this.mu; i++) {
      this.weights[i] /= sumw;
    }

    this.population = Array.from({ length: this.lambda }, () => new CMAESIndividual(dimension));
  }

  identityMatrix(n) {
    const mat = Array(n).fill().map(() => new Float64Array(n).fill(0));
    for (let i = 0; i < n; i++) mat[i][i] = 1;
    return mat;
  }

  async evolve(fitnessFunction) {
    for (let gen = 0; gen < this.generations; gen++) {
      // Sample offspring
      const offspring = [];
      for (let i = 0; i < this.lambda; i++) {
        const z = new Float64Array(this.dimension);
        for (let j = 0; j < this.dimension; j++) {
          z[j] = gaussianRandom();
        }
        const y = new Float64Array(this.dimension);
        for (let j = 0; j < this.dimension; j++) {
          let sum = 0;
          for (let k = 0; k < this.dimension; k++) {
            sum += this.C[j][k] * z[k];
          }
          y[j] = this.mean[j] + this.sigma * sum;
        }
        const ind = new CMAESIndividual(this.dimension);
        ind.params.set(y);
        ind.fitness = await fitnessFunction(y);
        ind.sti = Math.min(1.0, ind.fitness * 0.6 + ind.sti * 0.4);
        offspring.push({ ind, z, y });
      }

      // Sort by fitness + STI
      offspring.sort((a, b) => (b.ind.fitness + b.ind.sti) - (a.ind.fitness + a.ind.sti));

      // Update mean (weighted average of best μ)
      const newMean = new Float64Array(this.dimension);
      for (let i = 0; i < this.mu; i++) {
        const w = this.weights[i];
        const y = offspring[i].y;
        for (let j = 0; j < this.dimension; j++) {
          newMean[j] += w * y[j];
        }
      }
      this.mean = newMean;

      // Update evolution paths
      const zmean = new Float64Array(this.dimension);
      for (let i = 0; i < this.mu; i++) {
        const w = this.weights[i];
        const z = offspring[i].z;
        for (let j = 0; j < this.dimension; j++) {
          zmean[j] += w * z[j];
        }
      }
      for (let j = 0; j < this.dimension; j++) {
        this.ps[j] = (1 - this.cs) * this.ps[j] + Math.sqrt(this.cs * (2 - this.cs) * this.mu) * zmean[j];
        this.pc[j] = (1 - this.cc) * this.pc[j] + Math.sqrt(this.cc * (2 - this.cc) * this.mu) * zmean[j];
      }

      // Adapt covariance matrix
      const rank1 = this.outerProduct(this.pc, this.pc);
      const rankMu = this.zeroMatrix(this.dimension);
      for (let i = 0; i < this.mu; i++) {
        const w = this.weights[i];
        const y = offspring[i].y;
        const diff = new Float64Array(this.dimension);
        for (let j = 0; j < this.dimension; j++) {
          diff[j] = (y[j] - this.mean[j]) / this.sigma;
        }
        const op = this.outerProduct(diff, diff);
        for (let j = 0; j < this.dimension; j++) {
          for (let k = 0; k < this.dimension; k++) {
            rankMu[j][k] += w * op[j][k];
          }
        }
      }

      for (let j = 0; j < this.dimension; j++) {
        for (let k = 0; k < this.dimension; k++) {
          this.C[j][k] = (1 - this.c1 - this.cmu) * this.C[j][k] + this.c1 * rank1[j][k] + this.cmu * rankMu[j][k];
        }
      }

      // Adapt step-size
      const psNorm = Math.sqrt(this.ps.reduce((sum, v) => sum + v * v, 0));
      this.sigma *= Math.exp((this.cs / this.damps) * (psNorm / Math.sqrt(this.dimension) - 1));

      // Restart if sigma too small/large or covariance ill-conditioned (optional)
      if (this.sigma < 1e-10 || this.sigma > 1e10) {
        this.sigma = 0.3;
        this.C = this.identityMatrix(this.dimension);
        this.ps.fill(0);
        this.pc.fill(0);
      }
    }

    // Return best individual
    this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));
    const best = this.population[0];
    return {
      params: Array.from(best.params),
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4),
      sigma: this.sigma.toFixed(4)
    };
  }

  outerProduct(v1, v2) {
    const mat = Array(this.dimension).fill().map(() => new Float64Array(this.dimension));
    for (let i = 0; i < this.dimension; i++) {
      for (let j = 0; j < this.dimension; j++) {
        mat[i][j] = v1[i] * v2[j];
      }
    }
    return mat;
  }

  zeroMatrix(n) {
    return Array(n).fill().map(() => new Float64Array(n).fill(0));
  }
}

// Example fitness – higher = better
async function exampleFitness(params) {
  // Dummy: reward vector close to [0.618, 0.618, ...] (golden ratio vibe)
  let score = 0;
  const golden = (1 + Math.sqrt(5)) / 2 - 1;
  for (const p of params) {
    score -= Math.abs(p - golden);
  }
  // Attention boost
  // const highAtt = await updateAttention(params.join(","));
  // score += highAtt.length * 0.1;
  return Math.max(0, Math.min(1, (score + params.length * golden) / (params.length * golden)));
}

// Export for index.html integration
export { CMAES, exampleFitness };

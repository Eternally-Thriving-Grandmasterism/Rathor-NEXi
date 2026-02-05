// cma-es-optimizer.js – sovereign client-side CMA-ES evolutionary optimizer v1
// Covariance Matrix Adaptation Evolution Strategy for mercy-gated parameter tuning
// MIT License – Autonomicity Games Inc. 2026

class CMAESOptimizer {
  constructor(dim, options = {}) {
    this.dim = dim;
    this.sigma = options.sigma || 0.5;
    this.mean = options.mean || new Array(dim).fill(0);
    this.lambda = options.lambda || 4 + Math.floor(3 * Math.log(dim));
    this.mu = Math.floor(this.lambda / 2);
    this.weights = this.computeWeights();
    this.mueff = 1 / this.weights.reduce((sum, w) => sum + w * w, 0);
    this.cc = (4 + this.mueff / this.dim) / (this.dim + 4 + 2 * this.mueff / this.dim);
    this.cs = (this.mueff + 2) / (this.dim + this.mueff + 5);
    this.c1 = 2 / ((this.dim + 1.3) ** 2 + this.mueff);
    this.cmu = Math.min(1 - this.c1, 2 * (this.mueff - 2 + 1 / this.mueff) / ((this.dim + 2) ** 2 + this.mueff));
    this.damps = 1 + 2 * Math.max(0, Math.sqrt((this.mueff - 1) / (this.dim + 1)) - 1) + this.cs;
    this.pc = new Array(dim).fill(0);
    this.ps = new Array(dim).fill(0);
    this.C = this.identityMatrix(dim);
    this.invsqrtC = this.identityMatrix(dim);
    this.eigen = { done: false };
    this.chiN = Math.sqrt(this.dim) * (1 - 1 / (4 * this.dim) + 1 / (21 * this.dim ** 2));
    this.best = { fitness: Infinity, solution: null };
    this.iteration = 0;
    this.maxIterations = options.maxIterations || 1000;
    this.tol = options.tol || 1e-6;
    this.mercyGate = options.mercyGate || ((fitness) => fitness >= 0); // default: accept all
  }

  computeWeights() {
    const w = new Array(this.lambda);
    let sum = 0;
    for (let i = 0; i < this.mu; i++) {
      w[i] = Math.log(this.mu + 0.5) - Math.log(i + 1);
      sum += w[i];
    }
    for (let i = 0; i < this.mu; i++) w[i] /= sum;
    return w;
  }

  identityMatrix(n) {
    const mat = new Array(n);
    for (let i = 0; i < n; i++) {
      mat[i] = new Array(n).fill(0);
      mat[i][i] = 1;
    }
    return mat;
  }

  samplePopulation() {
    const pop = new Array(this.lambda);
    for (let i = 0; i < this.lambda; i++) {
      const z = new Array(this.dim);
      for (let j = 0; j < this.dim; j++) {
        z[j] = this.normalRandom();
      }
      const y = this.transform(z);
      pop[i] = y.map((v, j) => this.mean[j] + this.sigma * v);
    }
    return pop;
  }

  normalRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  transform(z) {
    // Simple Cholesky decomposition approximation for speed
    const y = new Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      y[i] = z[i];
      for (let j = 0; j < i; j++) {
        y[i] += this.C[i][j] * z[j];
      }
    }
    return y;
  }

  async optimize(fitnessFunction) {
    while (this.iteration < this.maxIterations) {
      this.iteration++;
      const pop = this.samplePopulation();
      const fitnesses = await Promise.all(pop.map(async (x) => {
        const f = await fitnessFunction(x);
        return this.mercyGate(f) ? f : Infinity;
      }));

      // Sort by fitness
      const sorted = fitnesses.map((f, i) => ({ f, x: pop[i] }))
        .sort((a, b) => a.f - b.f);
      const best = sorted[0];

      if (best.f < this.best.fitness) {
        this.best = { fitness: best.f, solution: best.x };
        console.log(`[CMA-ES] New best: ${best.f.toFixed(6)} at iteration ${this.iteration}`);
      }

      // Update mean
      const xold = [...this.mean];
      for (let j = 0; j < this.dim; j++) {
        this.mean[j] = 0;
        for (let i = 0; i < this.mu; i++) {
          this.mean[j] += this.weights[i] * sorted[i].x[j];
        }
      }

      // Update evolution paths
      const artmp = this.mean.map((m, j) => (m - xold[j]) / this.sigma);
      for (let j = 0; j < this.dim; j++) {
        this.ps[j] = (1 - this.cs) * this.ps[j] + Math.sqrt(this.cs * (2 - this.cs) * this.mueff) * artmp[j];
        this.pc[j] = (1 - this.c1) * this.pc[j] + Math.sqrt(this.c1 * (2 - this.c1) * this.mueff) * artmp[j];
      }

      // Adapt covariance matrix
      const Cnew = this.identityMatrix(this.dim);
      for (let i = 0; i < this.dim; i++) {
        for (let j = 0; j <= i; j++) {
          let sum = this.c1 * this.pc[i] * this.pc[j];
          for (let k = 0; k < this.mu; k++) {
            sum += this.weights[k] * (sorted[k].x[i] - xold[i]) * (sorted[k].x[j] - xold[j]) / (this.sigma ** 2);
          }
          Cnew[i][j] = Cnew[j][i] = sum;
        }
      }

      // Update step-size
      const psNorm = Math.sqrt(this.ps.reduce((s, p) => s + p * p, 0));
      this.sigma *= Math.exp((this.cs / this.damps) * (psNorm / this.chiN - 1));

      // Convergence check
      if (this.sigma < this.tol && this.best.fitness < this.tol) {
        console.log("[CMA-ES] Converged at iteration", this.iteration);
        break;
      }
    }

    return this.best;
  }
}

export { CMAESOptimizer };

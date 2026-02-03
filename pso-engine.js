// pso-engine.js – sovereign client-side Particle Swarm Optimization
// MIT License – Autonomicity Games Inc. 2026

class PSOParticle {
  constructor(dimension, bounds = [-5, 5]) {
    this.dimension = dimension;
    this.position = new Float64Array(dimension);
    this.velocity = new Float64Array(dimension);
    this.bestPosition = new Float64Array(dimension);
    this.bestFitness = -Infinity;
    this.sti = 0.1; // attention from Hyperon

    for (let i = 0; i < dimension; i++) {
      this.position[i] = bounds[0] + Math.random() * (bounds[1] - bounds[0]);
      this.velocity[i] = (Math.random() - 0.5) * 0.2;
      this.bestPosition[i] = this.position[i];
    }
  }

  updateVelocity(globalBest, w = 0.729, c1 = 1.496, c2 = 1.496) {
    for (let i = 0; i < this.dimension; i++) {
      const r1 = Math.random();
      const r2 = Math.random();
      this.velocity[i] = 
        w * this.velocity[i] +
        c1 * r1 * (this.bestPosition[i] - this.position[i]) +
        c2 * r2 * (globalBest.position[i] - this.position[i]);
      
      // Velocity clamping
      this.velocity[i] = Math.max(-1, Math.min(1, this.velocity[i]));
    }
  }

  updatePosition(bounds = [-5, 5]) {
    for (let i = 0; i < this.dimension; i++) {
      this.position[i] += this.velocity[i];
      // Bound clamping
      this.position[i] = Math.max(bounds[0], Math.min(bounds[1], this.position[i]));
    }
  }

  updateBest(fitness) {
    if (fitness > this.bestFitness) {
      this.bestFitness = fitness;
      this.bestPosition.set(this.position);
      this.sti = Math.min(1.0, fitness * 0.6 + this.sti * 0.4);
    }
  }
}

class PSOEngine {
  constructor(dimension = 10, swarmSize = 60, maxIterations = 120, inertia = 0.729, c1 = 1.496, c2 = 1.496) {
    this.dimension = dimension;
    this.swarmSize = swarmSize;
    this.maxIterations = maxIterations;
    this.w = inertia;
    this.c1 = c1;
    this.c2 = c2;

    this.swarm = Array.from({ length: swarmSize }, () => new PSOParticle(dimension));
    this.globalBest = this.swarm[0];
  }

  async optimize(fitnessFunction) {
    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Evaluate all particles
      for (const particle of this.swarm) {
        const fitness = await fitnessFunction(particle.position);
        particle.updateBest(fitness);

        if (fitness > this.globalBest.bestFitness) {
          this.globalBest = particle.copy();
          this.globalBest.bestFitness = fitness;
        }
      }

      // Update velocities & positions
      for (const particle of this.swarm) {
        particle.updateVelocity(this.globalBest, this.w, this.c1, this.c2);
        particle.updatePosition();
      }

      // Linear inertia decay
      this.w = 0.9 - (0.9 - 0.4) * (iter / this.maxIterations);
    }

    return {
      bestPosition: Array.from(this.globalBest.bestPosition),
      bestFitness: this.globalBest.bestFitness.toFixed(4),
      sti: this.globalBest.sti.toFixed(4),
      iterations: this.maxIterations
    };
  }
}

// Example fitness function (higher = better)
async function exampleFitness(position) {
  // Dummy: sphere function inverted (minimize distance to origin)
  let sum = 0;
  for (const p of position) {
    sum += p * p;
  }
  const score = -sum; // negative for maximization
  // Attention boost from Hyperon (optional integration)
  // const highAtt = await updateAttention(position.join(","));
  // score += highAtt.length * 0.1;
  return Math.max(-1000, Math.min(0, score));
}

// Export for index.html integration
export { PSOEngine, exampleFitness };

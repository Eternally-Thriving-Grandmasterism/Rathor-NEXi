// afso-engine.js – sovereign client-side Artificial Fish Swarm Optimization
// MIT License – Autonomicity Games Inc. 2026

class ArtificialFish {
  constructor(dimension, bounds = [-5, 5]) {
    this.dimension = dimension;
    this.position = new Float64Array(dimension);
    this.fitness = -Infinity;
    this.sti = 0.1; // attention from Hyperon

    for (let i = 0; i < dimension; i++) {
      this.position[i] = bounds[0] + Math.random() * (bounds[1] - bounds[0]);
    }
  }

  copy() {
    const copy = new ArtificialFish(this.dimension);
    copy.position.set(this.position);
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }
}

class AFSAEngine {
  constructor(dimension = 10, fishNum = 60, maxIter = 100, visual = 1.0, step = 0.5, crowdFactor = 0.8) {
    this.dimension = dimension;
    this.fishNum = fishNum;
    this.maxIter = maxIter;
    this.visual = visual;       // perception range
    this.step = step;           // max move distance
    this.crowdFactor = crowdFactor; // crowding threshold

    this.school = Array.from({ length: fishNum }, () => new ArtificialFish(dimension));
    this.bestFish = this.school[0];
  }

  async optimize(fitnessFunction) {
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (const fish of this.school) {
        fish.fitness = await fitnessFunction(fish.position);
        fish.sti = Math.min(1.0, fish.fitness * 0.6 + fish.sti * 0.4);

        if (fish.fitness > this.bestFish.fitness) {
          this.bestFish = fish.copy();
        }
      }

      // Each fish performs one behavior (prey / swarm / follow)
      for (const fish of this.school) {
        const behavior = this.chooseBehavior(fish);
        let newPos;

        if (behavior === "prey") {
          newPos = this.preyBehavior(fish, fitnessFunction);
        } else if (behavior === "swarm") {
          newPos = this.swarmBehavior(fish);
        } else {
          newPos = this.followBehavior(fish);
        }

        // Clamp position
        for (let i = 0; i < this.dimension; i++) {
          newPos[i] = Math.max(-5, Math.min(5, newPos[i]));
        }

        const newFitness = await fitnessFunction(newPos);
        if (newFitness > fish.fitness) {
          fish.position.set(newPos);
          fish.fitness = newFitness;
          fish.sti = Math.min(1.0, newFitness * 0.6 + fish.sti * 0.4);
        }
      }
    }

    return {
      bestPosition: Array.from(this.bestFish.position),
      bestFitness: this.bestFish.fitness.toFixed(4),
      sti: this.bestFish.sti.toFixed(4),
      iterations: this.maxIter
    };
  }

  chooseBehavior(fish) {
    // Probability-based choice (can be tuned)
    const r = Math.random();
    if (r < 0.4) return "prey";
    if (r < 0.7) return "swarm";
    return "follow";
  }

  preyBehavior(fish, fitnessFunction) {
    // Random search within visual range
    const newPos = fish.position.slice();
    for (let i = 0; i < this.dimension; i++) {
      newPos[i] += (Math.random() - 0.5) * 2 * this.step;
    }
    return newPos;
  }

  swarmBehavior(fish) {
    // Move toward center of nearby fish + avoid crowding
    const nearby = [];
    for (const other of this.school) {
      if (other === fish) continue;
      let dist = 0;
      for (let i = 0; i < this.dimension; i++) {
        dist += Math.pow(fish.position[i] - other.position[i], 2);
      }
      dist = Math.sqrt(dist);
      if (dist <= this.visual) nearby.push(other);
    }

    if (nearby.length === 0) return fish.position.slice();

    // Center of mass
    const center = new Float64Array(this.dimension);
    let crowding = nearby.length / this.fishNum;
    for (const n of nearby) {
      for (let i = 0; i < this.dimension; i++) {
        center[i] += n.position[i];
      }
    }
    for (let i = 0; i < this.dimension; i++) {
      center[i] /= nearby.length;
    }

    // Move toward center if not crowded
    const newPos = fish.position.slice();
    if (crowding < this.crowdFactor) {
      for (let i = 0; i < this.dimension; i++) {
        newPos[i] += this.step * (center[i] - fish.position[i]);
      }
    }

    return newPos;
  }

  followBehavior(fish) {
    // Follow the best fish in visual range
    let bestNearby = null;
    let bestFitness = -Infinity;
    let distToBest = Infinity;

    for (const other of this.school) {
      if (other === fish) continue;
      let dist = 0;
      for (let i = 0; i < this.dimension; i++) {
        dist += Math.pow(fish.position[i] - other.position[i], 2);
      }
      dist = Math.sqrt(dist);

      if (dist <= this.visual && other.fitness > bestFitness) {
        bestFitness = other.fitness;
        bestNearby = other;
        distToBest = dist;
      }
    }

    if (!bestNearby) return fish.position.slice();

    const newPos = fish.position.slice();
    for (let i = 0; i < this.dimension; i++) {
      newPos[i] += this.step * (bestNearby.position[i] - fish.position[i]);
    }

    return newPos;
  }
}

// Example fitness function (higher = better)
async function exampleFitness(position) {
  // Dummy: reward vector close to [0, 0, ..., 0] (minimize distance to origin)
  let sum = 0;
  for (const p of position) {
    sum += p * p;
  }
  const score = -sum;
  return Math.max(-1000, Math.min(0, score + 50));
}

// Export for index.html integration
export { AFSAEngine, exampleFitness };

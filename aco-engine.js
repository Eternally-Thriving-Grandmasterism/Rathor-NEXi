// aco-engine.js – sovereign client-side Ant Colony Optimization
// MIT License – Autonomicity Games Inc. 2026

class ACOAnt {
  constructor(problemSize) {
    this.path = new Array(problemSize).fill(0);
    this.visited = new Set();
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  constructSolution(graph, alpha = 1, beta = 2) {
    this.visited.clear();
    let current = Math.floor(Math.random() * graph.length);
    this.path[0] = current;
    this.visited.add(current);

    for (let step = 1; step < graph.length; step++) {
      const probabilities = [];
      let totalProb = 0;

      for (let next = 0; next < graph.length; next++) {
        if (this.visited.has(next)) continue;
        const pheromone = Math.pow(graph[current][next].pheromone || 0.1, alpha);
        const heuristic = Math.pow(graph[current][next].heuristic || 1, beta);
        const prob = pheromone * heuristic;
        probabilities.push({ next, prob });
        totalProb += prob;
      }

      let r = Math.random() * totalProb;
      let selected = null;
      for (const p of probabilities) {
        r -= p.prob;
        if (r <= 0) {
          selected = p.next;
          break;
        }
      }

      if (!selected) selected = probabilities[0].next; // fallback
      this.path[step] = selected;
      this.visited.add(selected);
      current = selected;
    }
  }
}

class ACOEngine {
  constructor(problemSize = 10, ants = 30, iterations = 100, alpha = 1, beta = 2, rho = 0.1, Q = 100) {
    this.problemSize = problemSize;
    this.ants = ants;
    this.iterations = iterations;
    this.alpha = alpha;   // pheromone importance
    this.beta = beta;     // heuristic importance
    this.rho = rho;       // evaporation rate
    this.Q = Q;           // pheromone deposit constant

    // Graph representation (pheromone matrix + heuristic)
    this.graph = Array(problemSize).fill().map(() =>
      Array(problemSize).fill().map(() => ({ pheromone: 1.0, heuristic: 1.0 }))
    );

    this.bestSolution = null;
    this.bestFitness = -Infinity;
  }

  async optimize(fitnessFunction) {
    for (let iter = 0; iter < this.iterations; iter++) {
      const antPopulation = Array.from({ length: this.ants }, () => new ACOAnt(this.problemSize));

      // Each ant constructs a solution
      for (const ant of antPopulation) {
        ant.constructSolution(this.graph, this.alpha, this.beta);
        ant.fitness = await fitnessFunction(ant.path);
        ant.sti = Math.min(1.0, ant.fitness * 0.6 + ant.sti * 0.4);

        if (ant.fitness > this.bestFitness) {
          this.bestFitness = ant.fitness;
          this.bestSolution = [...ant.path];
        }
      }

      // Global pheromone update (best-so-far)
      this.evaporatePheromone();
      for (const ant of antPopulation) {
        const deposit = this.Q / (1 + Math.abs(ant.fitness - this.bestFitness));
        for (let i = 0; i < this.problemSize - 1; i++) {
          const from = ant.path[i];
          const to = ant.path[i + 1];
          this.graph[from][to].pheromone += deposit * (ant.sti || 0.1);
          this.graph[to][from].pheromone += deposit * (ant.sti || 0.1); // symmetric
        }
      }

      // Local pheromone update on best solution
      const deposit = this.Q / (1 + Math.abs(this.bestFitness));
      for (let i = 0; i < this.problemSize - 1; i++) {
        const from = this.bestSolution[i];
        const to = this.bestSolution[i + 1];
        this.graph[from][to].pheromone += deposit;
        this.graph[to][from].pheromone += deposit;
      }
    }

    return {
      bestPath: this.bestSolution,
      bestFitness: this.bestFitness.toFixed(4),
      iterations: this.iterations
    };
  }

  evaporatePheromone() {
    for (let i = 0; i < this.problemSize; i++) {
      for (let j = 0; j < this.problemSize; j++) {
        this.graph[i][j].pheromone *= (1 - this.rho);
        this.graph[i][j].pheromone = Math.max(0.01, this.graph[i][j].pheromone);
      }
    }
  }
}

// Example fitness function (higher = better) – TSP-like
async function exampleTSPFitness(path) {
  // Dummy distances – in real use replace with actual heuristic
  let total = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const d = Math.abs(path[i] - path[i + 1]) + 1;
    total += d;
  }
  total += Math.abs(path[path.length - 1] - path[0]) + 1;
  return -total; // negative for maximization
}

// Export for index.html integration
export { ACOEngine, exampleTSPFitness };

// moses-evolutionary-engine.js – sovereign client-side MOSES evolutionary program synthesis
// Meta-optimizing semantic search, combinatory DSL, attention-modulated fitness
// MIT License – Autonomicity Games Inc. 2026

// Tiny combinatory DSL for evolved programs (OpenCog MOSES inspired)
const MOSES_DSL = [
  { name: "add", arity: 2, fn: (a,b) => a + b },
  { name: "mul", arity: 2, fn: (a,b) => a * b },
  { name: "and", arity: 2, fn: (a,b) => a && b ? 1 : 0 },
  { name: "or",  arity: 2, fn: (a,b) => a || b ? 1 : 0 },
  { name: "not", arity: 1, fn: (a) => a ? 0 : 1 },
  { name: "if",  arity: 3, fn: (c,t,f) => c ? t : f },
  { name: "eq",  arity: 2, fn: (a,b) => a === b ? 1 : 0 },
  { name: "gt",  arity: 2, fn: (a,b) => a > b ? 1 : 0 },
  { name: "const1", arity: 0, fn: () => 1 },
  { name: "const0", arity: 0, fn: () => 0 }
];

// Individual = tree program (nested array prefix notation)
class MosesIndividual {
  constructor(tree = null, maxDepth = 5) {
    this.tree = tree || this.randomTree(maxDepth);
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  randomTree(maxDepth) {
    if (maxDepth <= 0 || Math.random() < 0.3) {
      return Math.random() < 0.5 ? ["input"] : ["const", Math.random() < 0.5 ? 0 : 1];
    }

    const prim = MOSES_DSL[Math.floor(Math.random() * MOSES_DSL.length)];
    const children = [];
    for (let i = 0; i < prim.arity; i++) {
      children.push(this.randomTree(maxDepth - 1));
    }
    return [prim.name, ...children];
  }

  evaluate(inputs) {
    function exec(node) {
      if (!Array.isArray(node)) return node;
      const op = node[0];
      if (op === "input") return inputs.shift() || 0;
      if (op === "const") return node[1];

      const prim = MOSES_DSL.find(p => p.name === op);
      if (!prim) return 0;

      const args = node.slice(1).map(exec);
      return prim.fn(...args);
    }
    return exec(this.tree);
  }

  toString() {
    function str(node) {
      if (!Array.isArray(node)) return node.toString();
      const op = node[0];
      const args = node.slice(1).map(str);
      return `(${op} ${args.join(" ")})`;
    }
    return str(this.tree);
  }
}

// Population + MOSES evolution loop
class MosesEngine {
  constructor(popSize = 50, generations = 30, maxDepth = 7) {
    this.popSize = popSize;
    this.generations = generations;
    this.maxDepth = maxDepth;
    this.population = Array.from({ length: popSize }, () => new MosesIndividual(null, maxDepth));
  }

  async evolve(taskFitnessFn) {
    for (let gen = 0; gen < this.generations; gen++) {
      // Evaluate fitness (PLN TV proxy + attention)
      for (const ind of this.population) {
        ind.fitness = await taskFitnessFn(ind);
        ind.sti = Math.min(1.0, ind.fitness * 0.5 + ind.sti * 0.5);
      }

      // Sort by fitness + STI
      this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));

      // Elitism: keep top 20%
      const elite = this.population.slice(0, Math.floor(this.popSize * 0.2));

      // Breed rest
      const nextGen = [...elite];
      while (nextGen.length < this.popSize) {
        const p1 = this.tournamentSelect();
        const p2 = this.tournamentSelect();
        let child = this.crossover(p1.tree, p2.tree);
        child = this.mutate(child);
        nextGen.push(new MosesIndividual(child, this.maxDepth));
      }

      this.population = nextGen;
    }

    const best = this.population[0];
    return {
      program: best.toString(),
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4)
    };
  }

  tournamentSelect(size = 5) {
    let best = this.population[0];
    for (let i = 1; i < size; i++) {
      const cand = this.population[Math.floor(Math.random() * this.population.length)];
      if (cand.fitness + cand.sti > best.fitness + best.sti) best = cand;
    }
    return best;
  }

  crossover(tree1, tree2) {
    if (Math.random() < 0.5) return tree1;
    return tree2;
  }

  mutate(tree) {
    if (typeof tree !== "object") return tree;
    if (Math.random() < 0.3) {
      return new MosesIndividual(null, this.maxDepth).tree;
    }
    return [tree[0], ...tree.slice(1).map(child => this.mutate(child))];
  }
}

// Example fitness – can be task-specific (higher = better)
async function exampleFitness(ind) {
  const str = ind.toString();
  let score = 0;
  if (str.includes("Mercy")) score += 0.6;
  if (str.includes("Valence")) score += 0.7;
  if (str.includes("Harm")) score -= 0.8;

  // Boost by current Hyperon attention
  const highAtt = await updateAttention(str);
  score += highAtt.length * 0.05;

  return Math.max(0, Math.min(1, score));
}

// Export for index.html integration
export { MosesEngine, exampleFitness };

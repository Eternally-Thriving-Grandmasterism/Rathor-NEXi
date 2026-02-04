// neat-neuroevolution-engine.js – sovereign client-side Neuroevolution of Augmenting Topologies (NEAT)
// Expanded for AlphaProMega Air flight simulation: multi-aircraft swarm, physics dynamics, mercy-gated fitness
// MIT License – Autonomicity Games Inc. 2026

let globalInnovation = 0;

class NeuronGene {
  constructor(id, type = "hidden") {
    this.id = id;
    this.type = type;
    this.activation = Math.tanh;
  }
}

class ConnectionGene {
  constructor(inNode, outNode, weight = Math.random() * 2 - 1, enabled = true, innovation = 0) {
    this.in = inNode;
    this.out = outNode;
    this.weight = weight;
    this.enabled = enabled;
    this.innovation = innovation;
  }
}

class Genome {
  constructor() {
    this.nodes = new Map();
    this.connections = [];
    this.fitness = 0;
    this.adjustedFitness = 0;
    this.species = -1;
    this.sti = 0.1;
  }

  copy() {
    const copy = new Genome();
    copy.nodes = new Map(this.nodes);
    copy.connections = this.connections.map(c => new ConnectionGene(c.in, c.out, c.weight, c.enabled, c.innovation));
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }

  mutate() {
    this.connections.forEach(c => {
      if (Math.random() < 0.8) c.weight += (Math.random() - 0.5) * 0.1;
    });

    if (Math.random() < 0.05) {
      const inNode = Array.from(this.nodes.keys())[Math.floor(Math.random() * this.nodes.size)];
      const outNode = Array.from(this.nodes.keys())[Math.floor(Math.random() * this.nodes.size)];
      if (inNode !== outNode && !this.hasConnection(inNode, outNode)) {
        this.connections.push(new ConnectionGene(inNode, outNode, Math.random() * 2 - 1, true, globalInnovation++));
      }
    }

    if (Math.random() < 0.03 && this.connections.length > 0) {
      const conn = this.connections[Math.floor(Math.random() * this.connections.length)];
      if (conn.enabled) {
        conn.enabled = false;
        const newNodeId = this.nodes.size;
        this.nodes.set(newNodeId, new NeuronGene(newNodeId));
        this.connections.push(new ConnectionGene(conn.in, newNodeId, 1.0, true, globalInnovation++));
        this.connections.push(new ConnectionGene(newNodeId, conn.out, conn.weight, true, globalInnovation++));
      }
    }
  }

  hasConnection(inNode, outNode) {
    return this.connections.some(c => c.in === inNode && c.out === outNode);
  }

  evaluate(inputs) {
    const registers = new Float64Array(this.nodes.size);
    inputs.forEach((v, i) => registers[i] = v);

    const sortedNodes = Array.from(this.nodes.keys()).sort((a, b) => a - b);
    for (const nodeId of sortedNodes) {
      const node = this.nodes.get(nodeId);
      if (node.type === "input") continue;

      let sum = 0;
      for (const conn of this.connections) {
        if (conn.out === nodeId && conn.enabled) {
          sum += registers[conn.in] * conn.weight;
        }
      }
      registers[nodeId] = node.activation(sum);
    }

    return Array.from({ length: this.outputs }, (_, i) => registers[this.inputs + i]);
  }
}

class Species {
  constructor(id, representative) {
    this.id = id;
    this.genomes = [representative];
    this.representative = representative;
    this.maxFitness = representative.fitness;
    this.adjustedFitnessTotal = 0;
  }

  updateRepresentative() {
    this.representative = this.genomes[0];
  }
}

class NEAT {
  constructor(inputs = 6, outputs = 2, popSize = 200, maxGenerations = 120) {
    this.inputs = inputs;    // altitude, velocity, energy, integrity, wind, target dist
    this.outputs = outputs;  // thrust, pitch
    this.popSize = popSize;
    this.maxGenerations = maxGenerations;
    this.population = [];
    this.species = [];
    this.compatibilityThreshold = 3.0;
    this.targetSpeciesCount = Math.floor(popSize / 12) || 8;
    this.thresholdAdjustmentRate = 0.12;
    this.c1 = 1.0;
    this.c2 = 1.0;
    this.c3 = 0.4;
    this.speciesIdCounter = 0;

    this.initPopulation();
  }

  initPopulation() {
    for (let i = 0; i < this.popSize; i++) {
      const genome = new Genome();
      for (let inp = 0; inp < this.inputs; inp++) {
        genome.nodes.set(inp, new NeuronGene(inp, "input"));
      }
      for (let out = 0; out < this.outputs; out++) {
        const outId = this.inputs + out;
        genome.nodes.set(outId, new NeuronGene(outId, "output"));
        for (let inp = 0; inp < this.inputs; inp++) {
          genome.connections.push(new ConnectionGene(inp, outId, Math.random() * 2 - 1, true, globalInnovation++));
        }
      }
      this.population.push(genome);
    }
  }

  async evolve(flightSimulator) {
    for (let gen = 0; gen < this.maxGenerations; gen++) {
      for (const genome of this.population) {
        genome.fitness = flightSimulator.evaluate(genome);
        genome.sti = Math.min(1.0, genome.fitness * 0.6 + genome.sti * 0.4);
      }

      this.speciateWithDynamicThreshold();
      const totalAdjusted = this.adjustFitness();

      this.species.sort((a, b) => b.maxFitness - a.maxFitness);

      const nextPopulation = [];
      for (const species of this.species) {
        if (species.adjustedFitnessTotal <= 0) continue;

        const offspringCount = Math.floor((species.adjustedFitnessTotal / totalAdjusted) * this.popSize);
        if (offspringCount <= 0) continue;

        const elite = species.genomes[0].copy();
        nextPopulation.push(elite);

        for (let i = 0; i < offspringCount; i++) {
          let parent1 = species.tournamentSelect();
          let parent2 = parent1;
          if (Math.random() < 0.7 && species.genomes.length > 1) {
            parent2 = species.tournamentSelect();
          }
          let child = this.crossover(parent1, parent2);
          child.mutate();
          nextPopulation.push(child);
        }
      }

      while (nextPopulation.length < this.popSize) {
        const species = this.species[Math.floor(Math.random() * this.species.length)];
        const parent = species.tournamentSelect();
        const child = parent.copy();
        child.mutate();
        nextPopulation.push(child);
      }

      this.population = nextPopulation;
    }

    const best = this.population.reduce((a, b) => (a.fitness + a.sti > b.fitness + b.sti ? a : b));
    return {
      genome: best,
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4)
    };
  }

  speciateWithDynamicThreshold() {
    this.species = [];
    for (const genome of this.population) {
      let found = false;
      for (const species of this.species) {
        if (this.compatibilityDistance(genome, species.representative) < this.compatibilityThreshold) {
          species.genomes.push(genome);
          found = true;
          break;
        }
      }
      if (!found) {
        this.species.push(new Species(this.speciesIdCounter++, genome));
      }
    }

    // Dynamic threshold adjustment
    const currentSpeciesCount = this.species.length;
    const speciesDelta = currentSpeciesCount - this.targetSpeciesCount;

    if (speciesDelta > 0) {
      this.compatibilityThreshold += this.thresholdAdjustmentRate * speciesDelta;
    } else if (speciesDelta < 0) {
      this.compatibilityThreshold = Math.max(1.0, this.compatibilityThreshold - this.thresholdAdjustmentRate * Math.abs(speciesDelta));
    }

    // Mercy-gate boost: high-STI genomes get speciation preference
    this.population.forEach(genome => {
      if (genome.sti > 0.7) {
        this.compatibilityThreshold = Math.max(2.0, this.compatibilityThreshold * 0.95);
      }
    });

    this.species.forEach(s => s.updateRepresentative());
  }

  compatibilityDistance(g1, g2) {
    const genes1 = new Set(g1.connections.map(c => c.innovation));
    const genes2 = new Set(g2.connections.map(c => c.innovation));

    const matching = [];
    let weightDiffSum = 0;
    for (const c1 of g1.connections) {
      const c2 = g2.connections.find(c => c.innovation === c1.innovation);
      if (c2) {
        matching.push(c1);
        weightDiffSum += Math.abs(c1.weight - c2.weight);
      }
    }

    const N = Math.max(genes1.size, genes2.size) || 1;
    const excess = Math.abs(genes1.size - genes2.size) / N;
    const disjoint = (genes1.size + genes2.size - 2 * matching.length) / N;
    const weightDiff = matching.length > 0 ? weightDiffSum / matching.length : 0;

    return this.c1 * excess + this.c2 * disjoint + this.c3 * weightDiff;
  }

  adjustFitness() {
    let totalAdjusted = 0;
    for (const species of this.species) {
      species.adjustedFitnessTotal = 0;
      species.genomes.forEach(genome => {
        genome.adjustedFitness = genome.fitness / species.genomes.length;
        species.adjustedFitnessTotal += genome.adjustedFitness;
      });
      species.maxFitness = Math.max(...species.genomes.map(g => g.fitness));
      totalAdjusted += species.adjustedFitnessTotal;
    }
    return totalAdjusted;
  }

  tournamentSelect(species) {
    let best = species.genomes[0];
    for (let i = 1; i < 5; i++) {
      const cand = species.genomes[Math.floor(Math.random() * species.genomes.length)];
      if (cand.adjustedFitness + cand.sti > best.adjustedFitness + best.sti) {
        best = cand;
      }
    }
    return best;
  }

  crossover(g1, g2) {
    const child = new Genome();
    const maxInn = Math.max(...g1.connections.map(c => c.innovation), ...g2.connections.map(c => c.innovation));
    for (let inn = 0; inn <= maxInn; inn++) {
      const c1 = g1.connections.find(c => c.innovation === inn);
      const c2 = g2.connections.find(c => c.innovation === inn);
      if (c1 && c2) {
        child.connections.push(Math.random() < 0.5 ? c1 : c2);
      } else if (c1) {
        child.connections.push(c1);
      } else if (c2) {
        child.connections.push(c2);
      }
    }
    return child;
  }
}

// AlphaProMega Air flight simulator — multi-aircraft swarm evaluation
class FlightSimulator {
  constructor(numAircraft = 5) {
    this.aircraft = Array(numAircraft).fill().map(() => ({
      altitude: 0,
      velocity: 0,
      energy: 100,
      integrity: 1.0,
      targetAltitude: 500 + Math.random() * 500,
      targetVelocity: 200 + Math.random() * 100
    }));
  }

  evaluate(genome) {
    let totalFitness = 0;
    let valencePenalty = 0;

    for (const ac of this.aircraft) {
      const inputs = [
        ac.altitude / 1000,
        ac.velocity / 100,
        ac.energy / 100,
        ac.integrity,
        (ac.targetAltitude - ac.altitude) / 1000,
        (ac.targetVelocity - ac.velocity) / 100
      ];

      const [thrust, pitch] = genome.evaluate(inputs);

      // Simple physics update
      ac.velocity += thrust * 0.01 + pitch * 0.005;
      ac.altitude += ac.velocity * 0.01;
      ac.energy -= Math.abs(thrust) * 0.001 + Math.abs(pitch) * 0.0005;
      ac.integrity = Math.max(0, ac.integrity - 0.0001 * Math.random());

      // Fitness: reach targets, preserve energy/integrity
      const altError = Math.abs(ac.targetAltitude - ac.altitude) / 1000;
      const velError = Math.abs(ac.targetVelocity - ac.velocity) / 100;
      const energyLeft = ac.energy / 100;
      const integrityLeft = ac.integrity;

      const fitness = -altError - velError + energyLeft * 0.3 + integrityLeft * 0.5;
      totalFitness += fitness;

      // Valence penalty if harm detected
      if (ac.integrity < 0.7 || ac.energy < 20) valencePenalty += 0.2;
    }

    return Math.max(0, totalFitness / this.aircraft.length - valencePenalty);
  }
}

// Export for Ruskode integration
export { NEAT, FlightSimulator };

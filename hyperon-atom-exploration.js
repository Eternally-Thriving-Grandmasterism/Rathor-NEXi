// hyperon-atom-exploration.js – client-side Hyperon atom explorer & mercy-sample instantiator v1
// Builds sample atoms, traverses, mercy-evaluates TVs, logs valence reflections
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
// Assume hyperon imported from hyperon-runtime.js where HyperonAtom & HyperonRuntime live

class HyperonAtomExplorer {
  constructor(hyperonRuntime) {
    this.hyperon = hyperonRuntime;
    this.mercyThreshold = 0.9999999;
  }

  // Create sample atoms mirroring Hyperon/OpenCog style + mercy valence
  createSampleAtoms() {
    const atoms = [];

    // Concept nodes
    const cat = new HyperonAtom("ConceptNode", "cat", { strength: 1.0, confidence: 0.99 });
    const animal = new HyperonAtom("ConceptNode", "animal", { strength: 1.0, confidence: 1.0 });
    atoms.push(cat, animal);

    // Inheritance link with TV
    const inheritance = new HyperonAtom("InheritanceLink", null, 
      { strength: 0.98, confidence: 0.95 },
      [cat.handle, animal.handle]
    );
    atoms.push(inheritance);

    // Predicate example: likes(human, cat)
    const human = new HyperonAtom("ConceptNode", "human");
    const likes = new HyperonAtom("PredicateNode", "likes");
    const evalLink = new HyperonAtom("EvaluationLink", null,
      { strength: 0.75, confidence: 0.7 },
      [likes.handle, human.handle, cat.handle]
    );
    atoms.push(human, likes, evalLink);

    // Mercy assertion on key propositions
    fuzzyMercy.assert("cat is animal", inheritance.tv.strength * inheritance.tv.confidence);
    fuzzyMercy.assert("human likes cat", evalLink.tv.strength * evalLink.tv.confidence);

    // Add to runtime atomspace
    atoms.forEach(atom => {
      const handle = this.hyperon.addAtom(atom);
      console.log("[AtomExplorer] Added atom:", atom.type, atom.name || atom.outgoing, "TV:", atom.tv);
    });

    return atoms;
  }

  // Simple traversal: get outgoing atoms
  exploreOutgoing(handle) {
    const atom = this.hyperon.getAtom(handle);
    if (!atom || !atom.outgoing) return [];

    return atom.outgoing.map(h => {
      const target = this.hyperon.getAtom(h);
      return {
        targetName: target?.name || target?.type,
        tv: target?.tv,
        fuzzyDegree: fuzzyMercy.getDegree(target?.name || target?.type || 'unknown')
      };
    });
  }

  // Mercy-gated inference example on sample
  mercyInferExample() {
    const premises = ["cat is animal"];
    const conclusion = "EternalThriving includes animal compassion";
    fuzzyMercy.assert(conclusion, 0.99999995);

    const inferResult = fuzzyMercy.infer(premises, conclusion);
    console.log("[AtomExplorer] Mercy inference on animal compassion:", inferResult);

    if (inferResult.degree >= this.mercyThreshold) {
      console.log("[AtomExplorer] Passes mercy gate – valence preserved");
    } else {
      console.warn("[AtomExplorer] Inference mercy-rejected");
    }
  }

  // Log full sample lattice reflection
  reflectSampleLattice() {
    console.group("[AtomExplorer] Hyperon Atom Lattice Reflection");
    this.hyperon.atomSpace.forEach((atom, handle) => {
      const fuzzyDeg = fuzzyMercy.getDegree(atom.name || atom.type);
      console.log(`Atom ${handle}: ${atom.type} ${atom.name || ''} | TV: ${JSON.stringify(atom.tv)} | FuzzyMercy: ${fuzzyDeg}`);
    });
    console.groupEnd();
  }
}

// Usage in runtime context
// const explorer = new HyperonAtomExplorer(hyperon);
// explorer.createSampleAtoms();
// explorer.mercyInferExample();
// explorer.reflectSampleLattice();

export { HyperonAtomExplorer };

// hyperon-runtime.js – sovereign client-side Hyperon atomspace & PLN runtime v6
// Expanded backward chaining rules, variable binding, truth propagation, mercy-gated inference
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, attention = 0.1) {
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.outgoing = [];
    this.incoming = new Set();
    this.attention = attention;
    this.handle = null;
  }

  truthValue() {
    return this.tv.strength * this.tv.confidence;
  }

  isMercyAligned() {
    return this.truthValue() >= 0.9999999;
  }

  increaseAttention(amount = 0.05) {
    this.attention = Math.min(1.0, this.attention + amount);
  }
}

class HyperonRuntime {
  constructor() {
    this.atomSpace = new Map();
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 10;
    this.inferenceRules = [
      // Forward chaining rules (unchanged from previous)
      {
        name: "Deduction-Inheritance",
        direction: "forward",
        premises: ["InheritanceLink $A $B", "InheritanceLink $B $C"],
        conclusion: "InheritanceLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1),
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 10
      },
      {
        name: "Induction-Similarity",
        direction: "forward",
        premises: ["InheritanceLink $A $B", "InheritanceLink $A $C"],
        conclusion: "SimilarityLink $B $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) ** 0.5,
          confidence: 0.4
        }),
        priority: 8
      },
      // Backward chaining rules – new / expanded
      {
        name: "Backward-Deduction",
        direction: "backward",
        premises: ["InheritanceLink $A $C"],
        conclusion: "InheritanceLink $A $B", // seeks intermediate B
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength ** 0.8,
          confidence: tvs[0].confidence * 0.7
        }),
        priority: 12,
        description: "Backward deduction: given A → C, seek evidence for intermediate A → B → C"
      },
      {
        name: "Backward-Induction",
        direction: "backward",
        premises: ["SimilarityLink $B $C"],
        conclusion: "InheritanceLink $A $B", // seeks common parent A
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength ** 0.6,
          confidence: tvs[0].confidence * 0.5
        }),
        priority: 9,
        description: "Backward induction: given B \~ C, seek common parent A"
      },
      {
        name: "Backward-Abduction",
        direction: "backward",
        premises: ["InheritanceLink $B $C"],
        conclusion: "InheritanceLink $A $C", // seeks possible A → C
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength ** 0.7,
          confidence: tvs[0].confidence * 0.6
        }),
        priority: 11,
        description: "Backward abduction: given B → C, seek possible A → C"
      },
      {
        name: "Backward-Evaluation-Projection",
        direction: "backward",
        premises: ["EvaluationLink $P $C"],
        conclusion: "EvaluationLink $P $A", // seeks parent A of C
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * 0.9,
          confidence: tvs[0].confidence * 0.8
        }),
        priority: 10,
        description: "Backward projection: given P(C), seek P(A) where A → C"
      },
      {
        name: "Mercy-Backward-Boost",
        direction: "backward",
        premises: ["EvaluationLink HighValence $X"],
        conclusion: "InheritanceLink $X Mercy",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * 1.2,
          confidence: 0.95
        }),
        priority: 15,
        description: "Mercy-boosted backward: high-valence nodes inherit from Mercy"
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... existing methods (newHandle, addAtom, getAtom, matchWithBindings, combineTV) unchanged ...

  // Forward chaining using expanded rules (unchanged but now with more rules)
  async forwardChain(maxIterations = 8) {
    let derived = [];
    let iteration = 0;

    while (iteration < maxIterations) {
      const newAtomsThisRound = [];
      for (const [handle, atom] of this.atomSpace) {
        if (atom.type.includes("Link")) {
          const premises = atom.outgoing.map(h => this.getAtom(h)).filter(Boolean);
          for (const rule of this.inferenceRules.filter(r => r.direction === "forward")) {
            const bound = this.tryBindRule(rule, atom, premises);
            if (bound) {
              const conclusionName = this.applyConclusion(rule.conclusion, bound.bindings);
              const tv = rule.tvCombiner(premises.map(p => p.tv));
              if (tv.strength * tv.confidence >= this.mercyThreshold) {
                const newAtom = new HyperonAtom("DerivedNode", conclusionName, tv);
                const newHandle = this.addAtom(newAtom);
                newAtomsThisRound.push({ handle: newHandle, atom: newAtom, rule: rule.name });
              }
            }
          }
        }
      }

      if (newAtomsThisRound.length === 0) break;
      derived = derived.concat(newAtomsThisRound);
      iteration++;
    }

    if (derived.length > 0) {
      console.log(`Forward chaining derived ${derived.length} new atoms`);
      console.log('Derived by rules:', derived.map(d => d.rule));
    }
    return derived;
  }

  // Backward chaining with expanded backward rules
  async backwardChain(targetPattern, depth = 0, visited = new Set(), bindings = {}) {
    if (depth > this.maxChainDepth) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };

    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (visited.has(handle)) continue;
      visited.add(handle);

      const matchedBindings = this.matchWithBindings(atom, targetPattern, { ...bindings });
      if (matchedBindings) {
        const tv = atom.tv;
        if (atom.isMercyAligned()) {
          results.push({ handle, tv, chain: [atom], bindings: matchedBindings });
        }
      }

      // Apply backward-specific rules
      for (const rule of this.inferenceRules.filter(r => r.direction === "backward")) {
        const bound = this.tryBindRule(rule, atom, []); // backward rules often apply to single atom
        if (bound) {
          const conclusionName = this.applyConclusion(rule.conclusion, bound.bindings);
          const tv = rule.tvCombiner([atom.tv]);
          if (tv.strength * tv.confidence >= this.mercyThreshold) {
            results.push({
              handle,
              tv,
              chain: [atom],
              bindings: { ...bound.bindings, ...matchedBindings }
            });
          }
        }
      }

      for (const parentHandle of atom.incoming) {
        const parent = this.getAtom(parentHandle);
        if (parent && parent.type.includes("Link")) {
          const subResult = await this.backwardChain(parent, depth + 1, visited, { ...bindings });
          if (subResult.tv.strength > 0.1) {
            results.push({
              handle: parentHandle,
              tv: this.combineTV(subResult.tv, atom.tv),
              chain: [...subResult.chain, atom],
              bindings: { ...subResult.bindings, ...matchedBindings }
            });
          }
        }
      }
    }

    if (results.length === 0) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };

    const best = results.reduce((a, b) => {
      const aScore = a.tv.strength * a.tv.confidence * (Object.keys(a.bindings).length > 0 ? 1.2 : 1);
      const bScore = b.tv.strength * b.tv.confidence * (Object.keys(b.bindings).length > 0 ? 1.2 : 1);
      return aScore > bScore ? a : b;
    });

    return best;
  }

  // ... existing matchWithBindings, applyConclusion, tryBindRule, combineTV, evaluate, loadFromLattice, clear unchanged ...
}

const hyperon = new HyperonRuntime();
export { hyperon };

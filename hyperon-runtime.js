// hyperon-runtime.js – sovereign client-side Hyperon atomspace & PLN runtime v8
// Modus Tollens rule integrated, expanded chaining, variable binding, mercy-gated inference
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
      // Existing forward chaining rules (unchanged)
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

      // Modus Ponens (already present)
      {
        name: "Modus Ponens",
        direction: "forward",
        premises: ["ImplicationLink $A $B", "EvaluationLink $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 15
      },

      // NEW: Modus Tollens (classic negation rule)
      {
        name: "Modus Tollens",
        direction: "forward",
        premises: ["ImplicationLink $A $B", "EvaluationLink Not $B"],
        conclusion: "EvaluationLink Not $A",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 16, // very high priority for contradiction detection
        description: "If A → B and ¬B is true, then ¬A is true"
      },
      {
        name: "Modus Tollens-Backward",
        direction: "backward",
        premises: ["EvaluationLink Not $A"],
        conclusion: "EvaluationLink Not $B", // seeks consequent B where A → B
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * 0.9,
          confidence: tvs[0].confidence * 0.75
        }),
        priority: 14,
        description: "Backward Modus Tollens: given ¬A, seek ¬B where A → B"
      },

      // Existing backward chaining rules (unchanged)
      {
        name: "Backward-Deduction",
        direction: "backward",
        premises: ["InheritanceLink $A $C"],
        conclusion: "InheritanceLink $A $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength ** 0.8,
          confidence: tvs[0].confidence * 0.7
        }),
        priority: 12
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
        priority: 15
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... existing methods (newHandle, addAtom, getAtom, matchWithBindings, combineTV, forwardChain, backwardChain, evaluate, loadFromLattice, clear) unchanged ...

  // Forward chaining now includes Modus Tollens
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

  // Backward chaining now leverages Modus Tollens backward rule
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

      // Apply backward-specific rules (including Modus Tollens backward)
      for (const rule of this.inferenceRules.filter(r => r.direction === "backward")) {
        const bound = this.tryBindRule(rule, atom, []);
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

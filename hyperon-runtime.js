// hyperon-runtime.js – sovereign client-side Hyperon atomspace & PLN runtime v5
// Expanded inference rules, variable binding unification, truth propagation, mercy-gated chaining
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
      // Deduction (Inheritance transitivity)
      {
        name: "Deduction-Inheritance",
        premises: ["InheritanceLink $A $B", "InheritanceLink $B $C"],
        conclusion: "InheritanceLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1),
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 10
      },
      // Induction (similarity from common parent)
      {
        name: "Induction-Similarity",
        premises: ["InheritanceLink $A $B", "InheritanceLink $A $C"],
        conclusion: "SimilarityLink $B $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) ** 0.5,
          confidence: 0.4
        }),
        priority: 8
      },
      // Abduction (reverse induction)
      {
        name: "Abduction",
        premises: ["InheritanceLink $B $A", "InheritanceLink $C $A"],
        conclusion: "InheritanceLink $B $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) ** 0.5,
          confidence: 0.35
        }),
        priority: 7
      },
      // Evaluation → ConceptNode projection
      {
        name: "Evaluation-Projection",
        premises: ["EvaluationLink $P $A", "InheritanceLink $A $C"],
        conclusion: "EvaluationLink $P $C",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.7
        }),
        priority: 9
      },
      // Similarity → Inheritance (symmetry)
      {
        name: "Similarity-Symmetry",
        premises: ["SimilarityLink $A $B"],
        conclusion: "SimilarityLink $B $A",
        tvCombiner: (tvs) => tvs[0],
        priority: 6
      },
      // Mercy-aligned truth boosting
      {
        name: "Mercy-Boost",
        premises: ["ConceptNode Mercy", "InheritanceLink $X Mercy"],
        conclusion: "EvaluationLink HighValence $X",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) * 1.2,
          confidence: 0.95
        }),
        priority: 12
      }
    ].sort((a, b) => b.priority - a.priority); // higher priority first
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    const handle = this.newHandle();
    atom.handle = handle;
    this.atomSpace.set(handle, atom);

    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(handle);
    });

    return handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  // Forward chaining – derive new atoms using expanded rules
  async forwardChain(maxIterations = 8) {
    let derived = [];
    let iteration = 0;

    while (iteration < maxIterations) {
      const newAtomsThisRound = [];
      for (const [handle, atom] of this.atomSpace) {
        if (atom.type.includes("Link")) {
          const premises = atom.outgoing.map(h => this.getAtom(h)).filter(Boolean);
          for (const rule of this.inferenceRules) {
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

  // Backward chaining with variable binding (unchanged from previous expansion)
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

  matchWithBindings(atom, pattern, bindings = {}) {
    if (pattern.type && atom.type !== pattern.type) return null;

    if (pattern.name) {
      if (pattern.name.startsWith('$')) {
        const varName = pattern.name.slice(1);
        if (bindings[varName] !== undefined && bindings[varName] !== atom.name) return null;
        bindings[varName] = atom.name;
      } else if (pattern.name !== atom.name) {
        return null;
      }
    }

    if (pattern.outgoing) {
      if (atom.outgoing.length !== pattern.outgoing.length) return null;
      for (let i = 0; i < pattern.outgoing.length; i++) {
        const childAtom = this.getAtom(atom.outgoing[i]);
        if (!childAtom) return null;
        const childBindings = this.matchWithBindings(childAtom, pattern.outgoing[i], { ...bindings });
        if (!childBindings) return null;
        Object.assign(bindings, childBindings);
      }
    }

    return bindings;
  }

  applyConclusion(template, bindings) {
    return template.replace(/\$([a-z]+)/g, (_, varName) => bindings[varName] || `$${varName}`);
  }

  tryBindRule(rule, linkAtom, premises) {
    if (rule.premises.length !== premises.length + 1) return null;

    const bindings = {};
    for (let i = 0; i < rule.premises.length; i++) {
      const pattern = rule.premises[i];
      const target = i === 0 ? linkAtom : premises[i - 1];
      const match = this.matchWithBindings(target, pattern, bindings);
      if (!match) return null;
      Object.assign(bindings, match);
    }

    return { bindings };
  }

  combineTV(tv1, tv2) {
    const strength = (tv1.strength + tv2.strength) / 2;
    const confidence = Math.min(tv1.confidence, tv2.confidence);
    return { strength, confidence };
  }

  evaluate(expression) {
    const pattern = expression;
    const result = this.query(pattern);
    if (result.length === 0) return { truth: 0.1, confidence: 0.3 };

    const tv = result.reduce((acc, r) => {
      const v = r.atom.truthValue();
      return acc + v * (v >= this.mercyThreshold ? 1.5 : 0.5);
    }, 0) / result.length;

    return { truth: tv, confidence: Math.min(1, tv * 2) };
  }

  loadFromLattice(buffer) {
    console.log('Hyperon atoms loaded from lattice:', buffer ? buffer.byteLength : 'null');

    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    this.forwardChain();

    console.log('Hyperon bootstrap & chaining complete – mercy-aligned atoms ready');
  }

  clear() {
    this.atomSpace.clear();
    this.nextHandle = 0;
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };      const newAtomsThisRound = [];
      for (const [handle, atom] of this.atomSpace) {
        if (atom.type.includes("Link")) {
          const premises = atom.outgoing.map(h => this.getAtom(h)).filter(Boolean);
          for (const rule of this.inferenceRules) {
            const bound = this.tryBindRule(rule, atom, premises);
            if (bound) {
              const conclusion = this.applyConclusion(rule.conclusion, bound.bindings);
              const tv = rule.tvCombiner(premises.map(p => p.tv));
              if (tv.strength * tv.confidence >= this.mercyThreshold) {
                const newAtom = new HyperonAtom("DerivedNode", conclusion, tv);
                const newHandle = this.addAtom(newAtom);
                newAtomsThisRound.push({ handle: newHandle, atom: newAtom });
              }
            }
          }
        }
      }

      if (newAtomsThisRound.length === 0) break;
      derived = derived.concat(newAtomsThisRound);
      iteration++;
    }

    console.log(`Forward chaining complete – ${derived.length} new atoms derived`);
    return derived;
  }

  // Try to match rule premises to atom & premises
  tryBindRule(rule, linkAtom, premises) {
    if (rule.premises.length !== premises.length + 1) return null;

    const bindings = {};
    for (let i = 0; i < rule.premises.length; i++) {
      const pattern = rule.premises[i];
      const target = i === 0 ? linkAtom : premises[i - 1];
      const match = this.matchWithBindings(target, pattern, bindings);
      if (!match) return null;
      Object.assign(bindings, match);
    }

    return { bindings };
  }

  applyConclusion(template, bindings) {
    return template.replace(/\$([a-z]+)/g, (_, varName) => bindings[varName] || `$${varName}`);
  }

  // Backward chaining with variable binding propagation (previous version expanded)
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

  matchWithBindings(atom, pattern, bindings = {}) {
    if (pattern.type && atom.type !== pattern.type) return null;

    if (pattern.name) {
      if (pattern.name.startsWith('$')) {
        const varName = pattern.name.slice(1);
        if (bindings[varName] !== undefined && bindings[varName] !== atom.name) return null;
        bindings[varName] = atom.name;
      } else if (pattern.name !== atom.name) {
        return null;
      }
    }

    if (pattern.outgoing) {
      if (atom.outgoing.length !== pattern.outgoing.length) return null;
      for (let i = 0; i < pattern.outgoing.length; i++) {
        const childAtom = this.getAtom(atom.outgoing[i]);
        if (!childAtom) return null;
        const childBindings = this.matchWithBindings(childAtom, pattern.outgoing[i], { ...bindings });
        if (!childBindings) return null;
        Object.assign(bindings, childBindings);
      }
    }

    return bindings;
  }

  combineTV(tv1, tv2) {
    const strength = (tv1.strength + tv2.strength) / 2;
    const confidence = Math.min(tv1.confidence, tv2.confidence);
    return { strength, confidence };
  }

  evaluate(expression) {
    const pattern = expression;
    const result = this.query(pattern);
    if (result.length === 0) return { truth: 0.1, confidence: 0.3 };

    const tv = result.reduce((acc, r) => {
      const v = r.atom.truthValue();
      return acc + v * (v >= this.mercyThreshold ? 1.5 : 0.5);
    }, 0) / result.length;

    return { truth: tv, confidence: Math.min(1, tv * 2) };
  }

  loadFromLattice(buffer) {
    console.log('Hyperon atoms loaded from lattice:', buffer ? buffer.byteLength : 'null');

    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    this.forwardChain();

    console.log('Hyperon bootstrap & chaining complete – mercy-aligned atoms ready');
  }

  clear() {
    this.atomSpace.clear();
    this.nextHandle = 0;
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };

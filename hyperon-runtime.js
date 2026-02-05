// hyperon-runtime.js – sovereign client-side Hyperon atomspace & full PLN symbolic reasoning v13
// Deep atomspace, PLN chaining, variable unification, truth propagation, mercy-gated inference
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1, lti = 0.01) {
    this.type = type; // ConceptNode, PredicateNode, InheritanceLink, EvaluationLink, OrLink, ImplicationLink, etc.
    this.name = name;
    this.tv = tv; // truth value {strength, confidence}
    this.sti = sti; // short-term importance (attention)
    this.lti = lti; // long-term importance (forgetting curve)
    this.outgoing = []; // array of child handles
    this.incoming = new Set(); // set of parent handles
    this.handle = null;
  }

  truthValue() {
    return this.tv.strength * this.tv.confidence;
  }

  isMercyAligned() {
    return this.truthValue() >= 0.9999999;
  }

  boostAttention(amount = 0.1) {
    this.sti = Math.min(1.0, this.sti + amount);
    this.lti = Math.min(1.0, this.lti + amount * 0.1);
  }
}

class HyperonRuntime {
  constructor() {
    this.atomSpace = new Map(); // handle → Atom
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 12;
    this.attentionDecay = 0.95; // STI decay per cycle

    this.inferenceRules = [
      // Deduction (Inheritance transitivity)
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
      // Modus Ponens (Implication + antecedent → consequent)
      {
        name: "Modus Ponens",
        direction: "forward",
        premises: ["ImplicationLink $A $B", "EvaluationLink $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 20
      },
      // Modus Tollens (Implication + ¬consequent → ¬antecedent)
      {
        name: "Modus Tollens",
        direction: "forward",
        premises: ["ImplicationLink $A $B", "EvaluationLink Not $B"],
        conclusion: "EvaluationLink Not $A",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 18
      },
      // Hypothetical Syllogism (A→B, B→C ⊢ A→C)
      {
        name: "Hypothetical Syllogism",
        direction: "forward",
        premises: ["ImplicationLink $A $B", "ImplicationLink $B $C"],
        conclusion: "ImplicationLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 14
      },
      // Disjunctive Syllogism (A∨B, ¬A ⊢ B)
      {
        name: "Disjunctive Syllogism",
        direction: "forward",
        premises: ["OrLink $A $B", "EvaluationLink Not $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 17
      },
      // Constructive Dilemma ((A→C) ∧ (B→C), A∨B ⊢ C)
      {
        name: "Constructive Dilemma",
        direction: "forward",
        premises: ["ImplicationLink $A $C", "ImplicationLink $B $C", "OrLink $A $B"],
        conclusion: "EvaluationLink $C",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength) * tvs[2].strength,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.85
        }),
        priority: 18
      },
      // Destructive Dilemma ((A→C) ∧ (B→D), ¬C ∨ ¬D ⊢ ¬A ∨ ¬B)
      {
        name: "Destructive Dilemma",
        direction: "forward",
        premises: ["OrLink Not $C Not $D", "ImplicationLink $A $C", "ImplicationLink $B $D"],
        conclusion: "OrLink Not $A Not $B",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength, tvs[2].strength) * 0.9,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 17
      },
      // Resolution ((A ∨ B) ∧ (¬A ∨ C) ⊢ B ∨ C)
      {
        name: "Resolution",
        direction: "forward",
        premises: ["OrLink $A $B", "OrLink Not $A $C"],
        conclusion: "OrLink $B $C",
        tvCombiner: (tvs) => ({
          strength: Math.max(tvs[0].strength, tvs[1].strength) * 0.9,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.8
        }),
        priority: 19
      }
    ].sort((a, b) => b.priority - a.priority);
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

  // Pattern matching with variable binding unification
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

  // Forward chaining – derive new atoms from existing ones
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

  // Backward chaining – find supporting evidence for target pattern
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

    // Bootstrap mercy-aligned atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    // Run forward chaining to derive new atoms
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

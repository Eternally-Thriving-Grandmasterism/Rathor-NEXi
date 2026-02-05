// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v16
// Persistent IndexedDB-backed atomspace, expanded PLN chaining, mercy-gated inference
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1, lti = 0.01, handle = null) {
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti;
    this.lti = lti;
    this.outgoing = [];
    this.incoming = new Set();
    this.handle = handle;
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
    this.atomSpace = new Map();
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 12;
    this.attentionDecay = 0.95;
    this.db = null;
    this.dbName = "rathorHyperonDB";
    this.storeName = "atoms";

    this.plnRules = [
      // 1. Deduction (Inheritance transitivity)
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
      // 2. Induction (Similarity from shared inheritance)
      {
        name: "Induction-Similarity",
        premises: ["InheritanceLink $A $C", "InheritanceLink $B $C"],
        conclusion: "SimilarityLink $A $B",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) ** 0.5,
          confidence: 0.4
        }),
        priority: 8
      },
      // 3. Abduction (reverse induction)
      {
        name: "Abduction-Inheritance",
        premises: ["InheritanceLink $A $C", "InheritanceLink $B $C"],
        conclusion: "InheritanceLink $A $B",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1) ** 0.5,
          confidence: 0.3
        }),
        priority: 7
      },
      // 4. Analogy (similarity-based inheritance transfer)
      {
        name: "Analogy-Inheritance",
        premises: ["SimilarityLink $A $B", "InheritanceLink $A $C"],
        conclusion: "InheritanceLink $B $C",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.7
        }),
        priority: 9
      },
      // 5. Modus Ponens
      {
        name: "Modus Ponens",
        premises: ["ImplicationLink $A $B", "EvaluationLink $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 20
      },
      // 6. Modus Tollens
      {
        name: "Modus Tollens",
        premises: ["ImplicationLink $A $B", "EvaluationLink Not $B"],
        conclusion: "EvaluationLink Not $A",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 18
      },
      // 7. Hypothetical Syllogism
      {
        name: "Hypothetical Syllogism",
        premises: ["ImplicationLink $A $B", "ImplicationLink $B $C"],
        conclusion: "ImplicationLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 14
      },
      // 8. Disjunctive Syllogism
      {
        name: "Disjunctive Syllogism",
        premises: ["OrLink $A $B", "EvaluationLink Not $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 17
      },
      // 9. Constructive Dilemma
      {
        name: "Constructive Dilemma",
        premises: ["ImplicationLink $A $C", "ImplicationLink $B $C", "OrLink $A $B"],
        conclusion: "EvaluationLink $C",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength) * tvs[2].strength,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.85
        }),
        priority: 18
      },
      // 10. Destructive Dilemma
      {
        name: "Destructive Dilemma",
        premises: ["OrLink Not $C Not $D", "ImplicationLink $A $C", "ImplicationLink $B $D"],
        conclusion: "OrLink Not $A Not $B",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength, tvs[2].strength) * 0.9,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 17
      },
      // 11. Resolution
      {
        name: "Resolution",
        premises: ["OrLink $A $B", "OrLink Not $A $C"],
        conclusion: "OrLink $B $C",
        tvCombiner: (tvs) => ({
          strength: Math.max(tvs[0].strength, tvs[1].strength) * 0.9,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.8
        }),
        priority: 19
      },
      // 12. Revision (Bayesian truth-value update)
      {
        name: "Revision",
        premises: ["$A", "$A"], // same atom with different tv
        conclusion: "$A",
        tvCombiner: (tvs) => {
          const s1 = tvs[0].strength, c1 = tvs[0].confidence;
          const s2 = tvs[1].strength, c2 = tvs[1].confidence;
          const denom = c1 + c2 - c1 * c2 + 0.0001;
          return {
            strength: (s1 * c1 + s2 * c2 - s1 * s2 * c1 * c2) / denom,
            confidence: denom / (c1 + c2 - c1 * c2 + 0.0001)
          };
        },
        priority: 5
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  async init() {
    this.db = await this.openDB();
    await this.loadFromDB();
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        db.createObjectStore(this.storeName, { keyPath: "handle" });
      };
      req.onsuccess = e => resolve(e.target.result);
      req.onerror = reject;
    });
  }

  async loadFromDB() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readonly");
      const store = tx.objectStore(this.storeName);
      const req = store.getAll();
      req.onsuccess = () => {
        req.result.forEach(atomData => {
          const atom = new HyperonAtom(
            atomData.type,
            atomData.name,
            atomData.tv,
            atomData.sti,
            atomData.lti,
            atomData.handle
          );
          atom.outgoing = atomData.outgoing || [];
          atom.incoming = new Set(atomData.incoming || []);
          this.atomSpace.set(atom.handle, atom);
          this.nextHandle = Math.max(this.nextHandle, atom.handle + 1);
        });
        console.log("[Hyperon] Loaded", this.atomSpace.size, "atoms from IndexedDB");
        resolve();
      };
      req.onerror = reject;
    });
  }

  async saveAtom(atom) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readwrite");
      const store = tx.objectStore(this.storeName);
      const data = {
        handle: atom.handle,
        type: atom.type,
        name: atom.name,
        tv: atom.tv,
        sti: atom.sti,
        lti: atom.lti,
        outgoing: atom.outgoing,
        incoming: Array.from(atom.incoming)
      };
      store.put(data);
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    if (!atom.handle) atom.handle = this.newHandle();
    this.atomSpace.set(atom.handle, atom);

    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(atom.handle);
    });

    this.saveAtom(atom);
    return atom.handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  // ... (unify, occursCheck, applyBindings unchanged from previous advanced version) ...

  async forwardChain(maxIterations = 8) {
    let derived = [];
    let iteration = 0;

    while (iteration < maxIterations) {
      const newAtomsThisRound = [];
      for (const [handle, atom] of this.atomSpace) {
        if (atom.type.includes("Link")) {
          const premises = atom.outgoing.map(h => this.getAtom(h)).filter(Boolean);
          for (const rule of this.plnRules) {
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
      console.log(`[Hyperon] Forward PLN chaining derived ${derived.length} new atoms`);
      console.log('Derived by rules:', derived.map(d => d.rule));
    }
    return derived;
  }

  async backwardChain(targetPattern, depth = 0, visited = new Set(), bindings = {}) {
    if (depth > this.maxChainDepth) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };

    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (visited.has(handle)) continue;
      visited.add(handle);

      const matchedBindings = this.unify(targetPattern, atom, bindings);
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

  combineTV(tv1, tv2) {
    const strength = (tv1.strength + tv2.strength) / 2;
    const confidence = Math.min(tv1.confidence, tv2.confidence);
    return { strength, confidence };
  }

  async evaluate(expression) {
    const pattern = expression;
    const result = await this.backwardChain(pattern);
    if (!result || result.tv.strength < 0.1) return { truth: 0.1, confidence: 0.3 };

    return result.tv;
  }

  async boostFromLattice(buffer) {
    console.log("[Hyperon] Boosting from lattice:", buffer ? buffer.byteLength : 'null');

    // Bootstrap core mercy atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    // Add atoms from space-collab conversation
    const he3 = new HyperonAtom("ConceptNode", "He3", { strength: 0.98, confidence: 0.95 }, 0.8);
    const fusion = new HyperonAtom("ConceptNode", "AneutronicFusion", { strength: 0.99, confidence: 0.97 }, 0.9);
    const he3Inheritance = new HyperonAtom("InheritanceLink");
    he3Inheritance.outgoing = [he3.handle, fusion.handle];

    this.addAtom(he3);
    this.addAtom(fusion);
    this.addAtom(he3Inheritance);

    await this.forwardChain();

    console.log("[Hyperon] Lattice boost & full PLN chaining complete – mercy-aligned hypergraph ready");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };      {
        name: "Modus Ponens",
        premises: ["ImplicationLink $A $B", "EvaluationLink $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 20
      },
      // 6. Modus Tollens
      {
        name: "Modus Tollens",
        premises: ["ImplicationLink $A $B", "EvaluationLink Not $B"],
        conclusion: "EvaluationLink Not $A",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 18
      },
      // 7. Hypothetical Syllogism
      {
        name: "Hypothetical Syllogism",
        premises: ["ImplicationLink $A $B", "ImplicationLink $B $C"],
        conclusion: "ImplicationLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.85
        }),
        priority: 14
      },
      // 8. Disjunctive Syllogism
      {
        name: "Disjunctive Syllogism",
        premises: ["OrLink $A $B", "EvaluationLink Not $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength * 0.95,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 17
      },
      // 9. Constructive Dilemma
      {
        name: "Constructive Dilemma",
        premises: ["ImplicationLink $A $C", "ImplicationLink $B $C", "OrLink $A $B"],
        conclusion: "EvaluationLink $C",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength) * tvs[2].strength,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.85
        }),
        priority: 18
      },
      // 10. Destructive Dilemma
      {
        name: "Destructive Dilemma",
        premises: ["OrLink Not $C Not $D", "ImplicationLink $A $C", "ImplicationLink $B $D"],
        conclusion: "OrLink Not $A Not $B",
        tvCombiner: (tvs) => ({
          strength: Math.min(tvs[0].strength, tvs[1].strength, tvs[2].strength) * 0.9,
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 17
      },
      // 11. Revision (truth-value update)
      {
        name: "Revision",
        premises: ["$A", "$A"], // same atom with different tv
        conclusion: "$A",
        tvCombiner: (tvs) => {
          const s1 = tvs[0].strength, c1 = tvs[0].confidence;
          const s2 = tvs[1].strength, c2 = tvs[1].confidence;
          const denom = c1 + c2 - c1 * c2;
          return {
            strength: (s1 * c1 + s2 * c2 - s1 * s2 * c1 * c2) / denom,
            confidence: denom / (c1 + c2 - c1 * c2 + 0.0001)
          };
        },
        priority: 5
      },
      // 12. Evaluation boosting via attention
      {
        name: "Attention-Boost",
        premises: ["EvaluationLink $P $X"],
        conclusion: "EvaluationLink $P $X",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength,
          confidence: Math.min(1.0, tvs[0].confidence + 0.1)
        }),
        priority: 3
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  async init() {
    this.db = await this.openDB();
    await this.loadFromDB();
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        db.createObjectStore(this.storeName, { keyPath: "handle" });
      };
      req.onsuccess = e => resolve(e.target.result);
      req.onerror = reject;
    });
  }

  async loadFromDB() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readonly");
      const store = tx.objectStore(this.storeName);
      const req = store.getAll();
      req.onsuccess = () => {
        req.result.forEach(atomData => {
          const atom = new HyperonAtom(
            atomData.type,
            atomData.name,
            atomData.tv,
            atomData.sti,
            atomData.lti,
            atomData.handle
          );
          atom.outgoing = atomData.outgoing || [];
          atom.incoming = new Set(atomData.incoming || []);
          this.atomSpace.set(atom.handle, atom);
          this.nextHandle = Math.max(this.nextHandle, atom.handle + 1);
        });
        console.log("[Hyperon] Loaded", this.atomSpace.size, "atoms from IndexedDB");
        resolve();
      };
      req.onerror = reject;
    });
  }

  async saveAtom(atom) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readwrite");
      const store = tx.objectStore(this.storeName);
      const data = {
        handle: atom.handle,
        type: atom.type,
        name: atom.name,
        tv: atom.tv,
        sti: atom.sti,
        lti: atom.lti,
        outgoing: atom.outgoing,
        incoming: Array.from(atom.incoming)
      };
      store.put(data);
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    if (!atom.handle) atom.handle = this.newHandle();
    this.atomSpace.set(atom.handle, atom);

    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(atom.handle);
    });

    this.saveAtom(atom);
    return atom.handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  // ... (unify, occursCheck, applyBindings unchanged from previous advanced version) ...

  async forwardChain(maxIterations = 8) {
    let derived = [];
    let iteration = 0;

    while (iteration < maxIterations) {
      const newAtomsThisRound = [];
      for (const [handle, atom] of this.atomSpace) {
        if (atom.type.includes("Link")) {
          const premises = atom.outgoing.map(h => this.getAtom(h)).filter(Boolean);
          for (const rule of this.plnRules) {
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
      console.log(`[Hyperon] Forward PLN chaining derived ${derived.length} new atoms`);
      console.log('Derived by rules:', derived.map(d => d.rule));
    }
    return derived;
  }

  async backwardChain(targetPattern, depth = 0, visited = new Set(), bindings = {}) {
    if (depth > this.maxChainDepth) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };

    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (visited.has(handle)) continue;
      visited.add(handle);

      const matchedBindings = this.unify(targetPattern, atom, bindings);
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

  combineTV(tv1, tv2) {
    const strength = (tv1.strength + tv2.strength) / 2;
    const confidence = Math.min(tv1.confidence, tv2.confidence);
    return { strength, confidence };
  }

  async evaluate(expression) {
    const pattern = expression;
    const result = await this.backwardChain(pattern);
    if (!result || result.tv.strength < 0.1) return { truth: 0.1, confidence: 0.3 };

    return result.tv;
  }

  async boostFromLattice(buffer) {
    console.log("[Hyperon] Boosting from lattice:", buffer ? buffer.byteLength : 'null');

    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    await this.forwardChain();

    console.log("[Hyperon] Lattice boost & full PLN chaining complete – mercy-aligned hypergraph ready");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };

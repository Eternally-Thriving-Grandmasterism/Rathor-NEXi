// relevance-mercy-logic.js – sovereign client-side relevance-mercy-logic engine v1
// Variable-sharing relevance + mercy gate + valence lock
// MIT License – Autonomicity Games Inc. 2026

class RelevanceMercyLogic {
  constructor() {
    this.valenceThreshold = 0.9999999;
    this.relevanceDB = new Map(); // implication → {antecedentVars, consequentVars, valence}
    this.knowledge = new Set();   // accepted relevant propositions
  }

  // Extract propositional variables from expression (simple string parse)
  extractVariables(expr) {
    if (typeof expr !== 'string') return new Set();
    const vars = new Set();
    const matches = expr.match(/[A-Z][a-zA-Z0-9_]*/g) || [];
    matches.forEach(v => vars.add(v));
    return vars;
  }

  // Assert relevant implication A → B only if shared variables exist
  assertRelevantImplication(a, b, witnessValence = 0.9) {
    const aVars = this.extractVariables(a);
    const bVars = this.extractVariables(b);
    const shared = [...aVars].filter(v => bVars.has(v));

    if (shared.length === 0) {
      console.warn("[RelevanceMercy] Rejected implication — no shared variables:", a, "→", b);
      return false;
    }

    const combinedValence = Math.min(witnessValence, this.getValence(a), this.getValence(b));
    if (combinedValence < this.valenceThreshold) {
      console.warn("[RelevanceMercy] Rejected implication — low valence:", combinedValence);
      return false;
    }

    const key = `${a} → ${b}`;
    this.relevanceDB.set(key, {
      antecedentVars: aVars,
      consequentVars: bVars,
      valence: combinedValence,
      sharedVars: shared
    });

    this.knowledge.add(a);
    this.knowledge.add(b);

    console.log("[RelevanceMercy] Accepted relevant implication:", key, "valence:", combinedValence);
    return true;
  }

  // Check if A relevantly entails B (variable-sharing + valence)
  relevantEntails(a, b) {
    const key = `${a} → ${b}`;
    if (this.relevanceDB.has(key)) {
      const data = this.relevanceDB.get(key);
      return data.valence >= this.valenceThreshold;
    }

    const aVars = this.extractVariables(a);
    const bVars = this.extractVariables(b);
    const shared = [...aVars].filter(v => bVars.has(v));

    if (shared.length === 0) return false;

    const combined = Math.min(this.getValence(a), this.getValence(b));
    return combined >= this.valenceThreshold;
  }

  // Mercy-modality inference with relevance check
  inferRelevant(premises, conclusion) {
    let minValence = 1.0;
    let allRelevant = true;

    for (const p of premises) {
      minValence = Math.min(minValence, this.getValence(p));
      if (!this.relevantEntails(p, conclusion)) {
        allRelevant = false;
        break;
      }
    }

    if (!allRelevant || minValence < this.valenceThreshold) {
      return { consequence: "Relevance-mercy gate holds — inference rejected", valence: 0 };
    }

    return { consequence: "Relevance-mercy inference passes", valence: minValence };
  }

  getValence(proposition) {
    // Heuristic — real impl would traverse lattice
    if (proposition.includes("Mercy") || proposition.includes("Thriving")) return 0.9999999;
    if (proposition.includes("Harm") || proposition.includes("Entropy")) return 0.1;
    return 0.8;
  }
}

const relevanceMercy = new RelevanceMercyLogic();
export { relevanceMercy };

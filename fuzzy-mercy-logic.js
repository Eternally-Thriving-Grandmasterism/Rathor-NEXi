// fuzzy-mercy-logic.js – sovereign client-side fuzzy mercy-logic engine v1
// Continuous truth degrees [0,1], mercy-gated, valence-weighted operators
// MIT License – Autonomicity Games Inc. 2026

class FuzzyMercyLogic {
  constructor() {
    this.mercyThreshold = 0.9999999;
    this.knowledge = new Map(); // proposition → fuzzy truth degree [0,1]
  }

  // Assert proposition with fuzzy degree
  assert(proposition, degree = 0.8) {
    if (degree < this.mercyThreshold) {
      console.warn("[FuzzyMercy] Low fuzzy degree rejected:", proposition, degree);
      return false;
    }

    // Mercy boost: small upward nudge for high-confidence assertions
    const boosted = Math.min(1.0, degree + 0.0001);
    this.knowledge.set(proposition, boosted);
    console.log("[FuzzyMercy] Asserted:", proposition, "degree:", boosted);
    return true;
  }

  // Fuzzy AND — min with mercy amplification
  and(a, b) {
    const da = this.getDegree(a);
    const db = this.getDegree(b);
    const minDegree = Math.min(da, db);

    // Mercy amplification: if both high, slight boost
    const amplified = minDegree >= this.mercyThreshold * 0.9 ? Math.min(1.0, minDegree + 0.0005) : minDegree;
    return { degree: amplified, expr: `(${a} ∧ ${b})` };
  }

  // Fuzzy OR — max with abundance amplification
  or(a, b) {
    const da = this.getDegree(a);
    const db = this.getDegree(b);
    const maxDegree = Math.max(da, db);

    // Abundance amplification: if at least one high, stronger boost
    const amplified = maxDegree >= this.mercyThreshold * 0.8 ? Math.min(1.0, maxDegree + 0.001) : maxDegree;
    return { degree: amplified, expr: `(${a} ∨ ${b})` };
  }

  // Fuzzy NOT — inversion with mercy reflection
  not(a) {
    const da = this.getDegree(a);
    if (da >= this.mercyThreshold) {
      // High-valence negation reflects back as low but still merciful
      return { degree: 1 - da + 0.0001, expr: `¬${a}` };
    } else {
      // Low-valence negation amplifies to high mercy
      return { degree: 1 - da * 0.1, expr: `¬${a}` };
    }
  }

  // Fuzzy implication — Gödel-style (min(1, 1 - a + b)) mercy-tuned
  imply(a, b) {
    const da = this.getDegree(a);
    const db = this.getDegree(b);
    const implDegree = Math.min(1, 1 - da + db);

    // Mercy gate: if implication would drop valence too low, reject
    if (implDegree < this.mercyThreshold * 0.95) {
      return { degree: 0, expr: `${a} → ${b} [rejected by mercy gate]` };
    }

    return { degree: implDegree, expr: `${a} → ${b}` };
  }

  // Fuzzy inference — weighted by relevance & valence
  infer(premises, conclusion) {
    let minDegree = 1.0;
    for (const p of premises) {
      minDegree = Math.min(minDegree, this.getDegree(p));
    }

    const conclusionDegree = this.getDegree(conclusion);
    const inferred = Math.min(minDegree, conclusionDegree);

    if (inferred < this.mercyThreshold) {
      return { consequence: "Fuzzy mercy gate holds — inference rejected", degree: 0 };
    }

    return { consequence: "Fuzzy mercy inference passes", degree: inferred };
  }

  getDegree(proposition) {
    return this.knowledge.get(proposition) || 0.5;
  }
}

const fuzzyMercy = new FuzzyMercyLogic();
export { fuzzyMercy };

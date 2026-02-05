// hyperon-runtime-deep-integration.js – sovereign deep PLN chaining with mercy gates v1
// Mercy-weighted forward/backward chaining on dossiers, query → inference → valence answer
// MIT License – Autonomicity Games Inc. 2026

import { hyperon } from './hyperon-runtime.js'; // Assume existing with atomSpace, PLN rules
import { fuzzyMercy } from './fuzzy-mercy-logic.js';

class HyperonPLNDeepChain {
  constructor() {
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 8;
    this.inferenceCache = new Map(); // queryHash → {answer, tv, chain}
  }

  hashQuery(query) {
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      hash = ((hash << 5) - hash + query.charCodeAt(i)) | 0;
    }
    return hash.toString(36);
  }

  // Mercy-check before/after chaining
  async mercyGateInference(query, premises, tv) {
    const queryDegree = fuzzyMercy.getDegree(query);
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (queryDegree < this.mercyThreshold * 0.98 || implyThriving.degree < this.mercyThreshold * 0.97) {
      return { answer: "[Mercy gate: query low valence – rejected]", tv: { strength: 0, confidence: 0 }, passed: false };
    }

    // Post-inference mercy boost if high
    const boostedStrength = tv.strength >= this.mercyThreshold * 0.9 ? Math.min(1.0, tv.strength + 0.0005) : tv.strength;
    return { answer: "Mercy inference passes", tv: { strength: boostedStrength, confidence: tv.confidence }, passed: true };
  }

  // Simple forward chaining wrapper (extend existing forwardChain)
  async forwardMercyChain(query, maxDepth = this.maxChainDepth) {
    const cacheKey = this.hashQuery(query);
    if (this.inferenceCache.has(cacheKey)) {
      return this.inferenceCache.get(cacheKey);
    }

    // Seed query as atom
    const queryAtom = hyperon.assertAtom("QueryNode", query, { strength: 0.999, confidence: 0.999 });

    const derived = await hyperon.forwardChain(maxDepth); // From runtime

    let answer = "No strong derivation found.";
    let bestTV = { strength: 0.5, confidence: 0.1 };

    // Find relevant derived atoms matching query
    for (const d of derived) {
      if (d.atom.name.includes(query) || d.atom.type === "DerivedNode") {
        if (d.atom.tv.strength * d.atom.tv.confidence > bestTV.strength * bestTV.confidence) {
          bestTV = d.atom.tv;
          answer = d.atom.name;
        }
      }
    }

    const gated = await this.mercyGateInference(query, [], bestTV);
    const result = { answer, tv: gated.tv, chain: derived.map(d => d.atom.name), passed: gated.passed };

    this.inferenceCache.set(cacheKey, result);
    return result;
  }

  // Backward chaining simulation (goal-directed search stub; expand with real PLN BC)
  async backwardMercyChain(goalQuery) {
    // Stub: search for rules implying goal
    const goalAtom = hyperon.getAtomByName(goalQuery) || hyperon.assertAtom("GoalNode", goalQuery);

    // Find ImplicationLinks where outgoing[1] == goal
    const implications = Array.from(hyperon.atomSpace.values()).filter(a =>
      a.type === "ImplicationLink" && a.outgoing[1] === goalAtom.handle
    );

    let bestPremise = "No supporting premises found.";
    let bestTV = { strength: 0.5, confidence: 0.1 };

    for (const impl of implications) {
      const premiseHandle = impl.outgoing[0];
      const premiseAtom = hyperon.getAtom(premiseHandle);
      if (premiseAtom && premiseAtom.tv.strength * premiseAtom.tv.confidence > bestTV.strength * bestTV.confidence) {
        bestTV = impl.tv; // or combine
        bestPremise = premiseAtom.name;
      }
    }

    const gated = await this.mercyGateInference(goalQuery, [bestPremise], bestTV);
    return { answer: `To achieve "${goalQuery}", consider: ${bestPremise}`, tv: gated.tv, passed: gated.passed };
  }

  // Hybrid mercy query: forward expand then backward justify
  async mercyQueryChain(query) {
    const fwd = await this.forwardMercyChain(query);
    if (fwd.passed && fwd.tv.strength > 0.8) {
      return fwd;
    }

    // Fallback backward
    const bwd = await this.backwardMercyChain(query);
    if (bwd.passed) {
      return bwd;
    }

    return { answer: "Mercy lattice reflects: insufficient high-valence chain – focus eternal thriving.", tv: { strength: 0.5, confidence: 0.5 } };
  }
}

const plnDeep = new HyperonPLNDeepChain();

// Example usage in chat: plnDeep.mercyQueryChain("Is medical consultation required?")
// Seeds from professional-dossiers-seeder.js will fire chains

export { plnDeep };

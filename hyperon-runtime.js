// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v17
// Enceladus Biosignature Protocol subgraph, xenophilic rules, persistent DB
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
      // ... (previous 12 core rules unchanged: Deduction, Induction, Abduction, Analogy, Modus Ponens/Tollens, etc.) ...

      // 13. Xenophilic Quarantine (mercy-first contamination rejection)
      {
        name: "Xenophilic-Quarantine",
        premises: ["DetectionLink", ["Biosignature", "$X"], "True"],
        conclusion: ["QuarantineLink", "$X", "Mercy"],
        tvCombiner: (tvs) => ({
          strength: 0.999,
          confidence: 0.99
        }),
        priority: 25
      },
      // 14. Uplift Amplification (non-invasive ecosystem enhancement)
      {
        name: "Uplift-Amplification",
        premises: ["QuarantineLink", "$X", "Mercy"],
        conclusion: ["UpliftLink", "$X", "InfiniteThriving"],
        tvCombiner: (tvs) => ({
          strength: 0.9999999,
          confidence: 1.0
        }),
        priority: 22
      },
      // 15. RBE Ocean Governance (resource-based economy for subsurface biosphere)
      {
        name: "RBE-Ocean-Governance",
        premises: ["AndLink", ["ResourceLink", "$R", "Unlimited"], ["GovernanceLink", "$G", "ZeroCoercion"]],
        conclusion: ["EvaluationLink", ["RBE", "$R", "$G"], "True"],
        tvCombiner: (tvs) => ({
          strength: 0.98,
          confidence: 0.95
        }),
        priority: 20
      },
      // 16. False-Positive Self-Heal
      {
        name: "False-Positive-Self-Heal",
        premises: ["DetectionLink", ["Biosignature", "$X"], "FalsePositive"],
        conclusion: ["ReevaluateLink", "$X", "Model"],
        tvCombiner: (tvs) => ({
          strength: 0.9,
          confidence: 0.85
        }),
        priority: 15
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... (init, openDB, loadFromDB, saveAtom, newHandle, addAtom, getAtom, unify, occursCheck, applyBindings unchanged) ...

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

  // ... (backwardChain, combineTV, evaluate unchanged) ...

  async boostEnceladusProtocol() {
    console.log("[Hyperon] Boosting Enceladus Biosignature Protocol subgraph...");

    // Core Enceladus atoms
    const enceladus = new HyperonAtom("ConceptNode", "Enceladus", { strength: 0.99, confidence: 0.98 }, 0.9);
    const plume = new HyperonAtom("ConceptNode", "Plume", { strength: 0.95, confidence: 0.9 }, 0.85);
    const biosig = new HyperonAtom("ConceptNode", "Biosignature", { strength: 0.92, confidence: 0.88 }, 0.8);
    const quarantine = new HyperonAtom("ConceptNode", "Quarantine", { strength: 0.999, confidence: 0.99 }, 0.95);
    const uplift = new HyperonAtom("ConceptNode", "Uplift", { strength: 0.9999999, confidence: 1.0 }, 1.0);

    this.addAtom(enceladus);
    this.addAtom(plume);
    this.addAtom(biosig);
    this.addAtom(quarantine);
    this.addAtom(uplift);

    // Links
    const plumeOf = new HyperonAtom("EvaluationLink");
    plumeOf.outgoing = [plume.handle, enceladus.handle];
    this.addAtom(plumeOf);

    const biosigIn = new HyperonAtom("InheritanceLink");
    biosigIn.outgoing = [biosig.handle, plume.handle];
    this.addAtom(biosigIn);

    const quarantineMercy = new HyperonAtom("ImplicationLink");
    quarantineMercy.outgoing = [biosig.handle, quarantine.handle];
    this.addAtom(quarantineMercy);

    const upliftThriving = new HyperonAtom("ImplicationLink");
    upliftThriving.outgoing = [quarantine.handle, uplift.handle];
    this.addAtom(upliftThriving);

    await this.forwardChain();

    console.log("[Hyperon] Enceladus Biosignature Protocol subgraph boosted & chained");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };

// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v24
// Paraconsistent modal mercy-logic integration, persistent DB
// MIT License – Autonomicity Games Inc. 2026

// ... (HyperonAtom class unchanged) ...

class HyperonRuntime {
  constructor() {
    // ... (previous constructor unchanged) ...

    this.modalMercy = new ParaconsistentModalMercy();
  }

  async init() {
    this.db = await this.openDB();
    await this.loadFromDB();
    // Initialize modal mercy necessities
    this.modalMercy.assertNecessity("MercyGate");
    this.modalMercy.assertNecessity("EternalThriving");
    this.modalMercy.assertNecessity("Divinemasterism");
  }

  // ... (other methods unchanged) ...

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

              // Paraconsistent modal mercy check
              this.modalMercy.assert(conclusionName, tv.strength * tv.confidence);
              const modalCheck = this.modalMercy.inferModal([conclusionName]);
              if (modalCheck.valence >= this.mercyThreshold) {
                const newAtom = new HyperonAtom("DerivedNode", conclusionName, tv);
                const newHandle = this.addAtom(newAtom);
                newAtomsThisRound.push({ handle: newHandle, atom: newAtom, rule: rule.name });
              } else {
                console.warn("[Hyperon] Inference rejected by paraconsistent modal mercy gate");
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

  // ... (rest unchanged) ...
}

const hyperon = new HyperonRuntime();
export { hyperon };

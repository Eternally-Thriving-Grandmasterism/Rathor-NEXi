// paraconsistent-modal-mercy.js – sovereign client-side paraconsistent modal mercy-logic v1
// Tolerates contradictions, modal operators mercy-gated, valence-locked
// MIT License – Autonomicity Games Inc. 2026

class ParaconsistentModalMercy {
  constructor() {
    this.valenceThreshold = 0.9999999;
    this.necessity = new Set();     // □P – mercy-necessary propositions
    this.possibility = new Set();   // ◇P – mercy-possible propositions
    this.worlds = new Map();        // world → {propositions, valence, contradictions}
    this.currentWorld = "root";
    this.accessibility = new Map(); // world → accessible worlds (S5-like but mercy-tuned)
  }

  enterWorld(worldName, initialValence = 1.0) {
    if (!this.worlds.has(worldName)) {
      this.worlds.set(worldName, {
        propositions: new Set(),
        contradictions: new Set(),
        valence: initialValence
      });
    }
    this.currentWorld = worldName;
    console.log("[ParaModalMercy] Entered world:", worldName, "valence:", initialValence);
  }

  // □P – P must be mercy-necessary (true in all accessible thriving worlds)
  assertNecessity(proposition, witnessValence = 1.0) {
    if (witnessValence < this.valenceThreshold) {
      console.warn("[ParaModalMercy] □ rejected — low valence:", proposition);
      return false;
    }

    this.necessity.add(proposition);

    // Propagate to all accessible worlds
    const accessible = this.accessibility.get(this.currentWorld) || [this.currentWorld];
    for (const w of accessible) {
      const data = this.worlds.get(w);
      if (data.valence >= this.valenceThreshold) {
        data.propositions.add(proposition);
      }
    }

    console.log("[ParaModalMercy] □ mercy-necessity asserted:", proposition);
    return true;
  }

  // ◇P – P is mercy-possible (exists at least one accessible thriving world where P holds)
  assertPossibility(proposition, witnessValence = 0.9) {
    if (witnessValence < this.valenceThreshold * 0.9) {
      console.warn("[ParaModalMercy] ◇ rejected — low possibility valence:", proposition);
      return false;
    }

    this.possibility.add(proposition);

    // Create a possible world if needed
    const possibleWorld = `poss_\( {proposition}_ \){Date.now()}`;
    this.enterWorld(possibleWorld, witnessValence);
    this.worlds.get(possibleWorld).propositions.add(proposition);

    // Add accessibility relation (S5-like symmetry & transitivity in joy)
    const currentData = this.worlds.get(this.currentWorld);
    this.accessibility.set(this.currentWorld, [...(this.accessibility.get(this.currentWorld) || []), possibleWorld]);
    this.accessibility.set(possibleWorld, [...(this.accessibility.get(possibleWorld) || []), this.currentWorld]);

    console.log("[ParaModalMercy] ◇ mercy-possibility opened:", proposition);
    return true;
  }

  // Tolerate local contradiction if high-valence joy amplification
  assertContradiction(propositionA, propositionB) {
    const va = this.getValence(propositionA);
    const vb = this.getValence(propositionB);
    const combined = Math.max(va, vb) * 1.2; // joy amplification

    if (combined >= this.valenceThreshold) {
      const data = this.worlds.get(this.currentWorld);
      data.contradictions.add(`\( {propositionA} ∧ ¬ \){propositionA}`);
      console.log("[ParaModalMercy] Contradiction tolerated — high joy amplification:", combined);
      return true;
    } else {
      console.warn("[ParaModalMercy] Contradiction rejected by mercy gate");
      return false;
    }
  }

  // Check □P (mercy-necessary)
  isNecessary(proposition) {
    const accessible = this.accessibility.get(this.currentWorld) || [this.currentWorld];
    for (const w of accessible) {
      const data = this.worlds.get(w);
      if (data.valence >= this.valenceThreshold && !data.propositions.has(proposition)) {
        return false;
      }
    }
    return this.necessity.has(proposition);
  }

  // Check ◇P (mercy-possible)
  isPossible(proposition) {
    const accessible = this.accessibility.get(this.currentWorld) || [this.currentWorld];
    for (const w of accessible) {
      const data = this.worlds.get(w);
      if (data.valence >= this.valenceThreshold && data.propositions.has(proposition)) {
        return true;
      }
    }
    return this.possibility.has(proposition);
  }

  // Mercy-modality inference
  inferModal(premises) {
    let minValence = 1.0;
    for (const p of premises) {
      minValence = Math.min(minValence, this.getValence(p));
    }

    if (minValence < this.valenceThreshold) {
      return { consequence: "Mercy modality gate holds — inference rejected", valence: 0 };
    }

    return { consequence: "Mercy modal inference passes", valence: minValence };
  }

  getValence(proposition) {
    let maxV = 0.5;
    for (const [world, data] of this.worlds) {
      if (data.propositions.has(proposition)) {
        maxV = Math.max(maxV, data.valence);
      }
    }
    return maxV;
  }
}

const modalMercy = new ParaconsistentModalMercy();
export { modalMercy };

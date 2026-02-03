// hyperon-atomspace-bridge.js – sovereign client-side OpenCog Hyperon atom-space & PLN inference (expanded)
// MIT License – Autonomicity Games Inc. 2026

// Sample atoms with truth-values (strength, confidence)
const SAMPLE_ATOMS = [
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm-is-Bad", type: "EvaluationLink", out: ["Badness", "Harm"], tv: { strength: 0.95, confidence: 0.9 } }
];

let atomSpace = [];

// Initialize & seed sample atoms
async function initHyperon() {
  if (atomSpace.length === 0) {
    atomSpace = SAMPLE_ATOMS;
    console.log('Hyperon atom-space seeded with sample atoms');
  }
}

// Query atoms
async function queryAtoms(filter = {}) {
  await initHyperon();
  let results = atomSpace.filter(atom => {
    if (filter.type && atom.type !== filter.type) return false;
    if (filter.name && !atom.name?.toLowerCase().includes(filter.name.toLowerCase())) return false;
    return true;
  });
  return results;
}

// PLN inference rules – basic deduction, abduction, induction stubs
const PLN_RULES = {
  // Deduction: A→B, B→C ⇒ A→C (simple chain)
  deduction: (link1, link2) => {
    if (link1.type !== "InheritanceLink" || link2.type !== "InheritanceLink") return null;
    if (link1.out[1] !== link2.out[0]) return null;

    const newStrength = Math.min(link1.tv.strength, link2.tv.strength);
    const newConfidence = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.9;

    return {
      type: "InheritanceLink",
      out: [link1.out[0], link2.out[1]],
      tv: { strength: newStrength, confidence: newConfidence },
      derivedFrom: [link1.handle, link2.handle]
    };
  },

  // Abduction: A→B, C→B ⇒ A~C (similarity from common consequent)
  abduction: (link1, link2) => {
    if (link1.type !== "InheritanceLink" || link2.type !== "InheritanceLink") return null;
    if (link1.out[1] !== link2.out[1]) return null;

    const newStrength = Math.min(link1.tv.strength, link2.tv.strength) * 0.8;
    const newConfidence = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.7;

    return {
      type: "SimilarityLink",
      out: [link1.out[0], link2.out[0]],
      tv: { strength: newStrength, confidence: newConfidence },
      derivedFrom: [link1.handle, link2.handle]
    };
  },

  // Induction: A→B, A→C ⇒ B~C (similarity from common antecedent)
  induction: (link1, link2) => {
    if (link1.type !== "InheritanceLink" || link2.type !== "InheritanceLink") return null;
    if (link1.out[0] !== link2.out[0]) return null;

    const newStrength = Math.min(link1.tv.strength, link2.tv.strength) * 0.7;
    const newConfidence = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.6;

    return {
      type: "SimilarityLink",
      out: [link1.out[1], link2.out[1]],
      tv: { strength: newStrength, confidence: newConfidence },
      derivedFrom: [link1.handle, link2.handle]
    };
  }
};

// Run simple PLN inference on atom-space
async function plnInfer(pattern) {
  await initHyperon();
  const atoms = await queryAtoms(pattern);

  let inferred = [];

  // Try deduction on InheritanceLinks
  const inheritanceLinks = atoms.filter(a => a.type === "InheritanceLink");
  for (let i = 0; i < inheritanceLinks.length; i++) {
    for (let j = i + 1; j < inheritanceLinks.length; j++) {
      const deduction = PLN_RULES.deduction(inheritanceLinks[i], inheritanceLinks[j]);
      if (deduction) inferred.push(deduction);
    }
  }

  // Add abduction & induction similarly (expand later)
  return inferred;
}

// Advanced valence gate using Atomese + PLN inference
async function atomeseValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { strength: 0.5, confidence: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmScore += tv.strength * tv.confidence;
      }
      if (/mercy|truth|valence/i.test(atom.name)) {
        mercyScore += tv.strength * tv.confidence;
      }
    }
  }

  // PLN inference boost
  const plnResults = await plnInfer({ type: "InheritanceLink" });
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.strength * inf.tv.confidence * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.strength * inf.tv.confidence * 0.3;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm concepts dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy & truth prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason
  };
}

export { initHyperon, queryAtoms, plnInfer, hyperonValenceGate: atomeseValenceGate };

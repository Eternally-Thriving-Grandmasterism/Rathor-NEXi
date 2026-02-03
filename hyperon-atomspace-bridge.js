// hyperon-atomspace-bridge.js – sovereign client-side Hyperon atom-space & expanded PLN inference
// MIT License – Autonomicity Games Inc. 2026

let atomSpace = [];

// Sample atoms with truth-values (strength, confidence)
const SAMPLE_ATOMS = [
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { s: 0.01, c: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { s: 0.001, c: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { s: 1.0, c: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { s: 1.0, c: 1.0 } },
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { s: 1.0, c: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm-is-Bad", type: "EvaluationLink", out: ["Badness", "Harm"], tv: { s: 0.95, c: 0.9 } }
];

async function initHyperon() {
  if (atomSpace.length === 0) {
    atomSpace = SAMPLE_ATOMS;
    console.log('Hyperon atom-space seeded');
  }
}

// Query atoms
async function queryAtoms(filter = {}) {
  await initHyperon();
  return atomSpace.filter(atom => {
    if (filter.type && atom.type !== filter.type) return false;
    if (filter.name && !atom.name?.toLowerCase().includes(filter.name.toLowerCase())) return false;
    return true;
  });
}

// Expanded PLN inference rules
const PLN_RULES = {
  // Deduction: A→B (s1,c1), B→C (s2,c2) ⇒ A→C (min(s1,s2), min(c1,c2)*decay)
  deduction: (link1, link2) => {
    if (link1.type !== 'InheritanceLink' || link2.type !== 'InheritanceLink') return null;
    if (link1.out[1] !== link2.out[0]) return null;

    const s = Math.min(link1.tv.s, link2.tv.s);
    const c = Math.min(link1.tv.c, link2.tv.c) * 0.9;
    return {
      type: "InheritanceLink",
      out: [link1.out[0], link2.out[1]],
      tv: { s, c },
      derivedFrom: [link1.handle, link2.handle]
    };
  },

  // Abduction: A→B (s1,c1), C→B (s2,c2) ⇒ A~C (s1*s2*0.8, min(c1,c2)*0.7)
  abduction: (link1, link2) => {
    if (link1.type !== 'InheritanceLink' || link2.type !== 'InheritanceLink') return null;
    if (link1.out[1] !== link2.out[1]) return null;

    const s = link1.tv.s * link2.tv.s * 0.8;
    const c = Math.min(link1.tv.c, link2.tv.c) * 0.7;
    return {
      type: "SimilarityLink",
      out: [link1.out[0], link2.out[0]],
      tv: { s, c },
      derivedFrom: [link1.handle, link2.handle]
    };
  },

  // Induction: A→B (s1,c1), A→C (s2,c2) ⇒ B~C (s1*s2*0.7, min(c1,c2)*0.6)
  induction: (link1, link2) => {
    if (link1.type !== 'InheritanceLink' || link2.type !== 'InheritanceLink') return null;
    if (link1.out[0] !== link2.out[0]) return null;

    const s = link1.tv.s * link2.tv.s * 0.7;
    const c = Math.min(link1.tv.c, link2.tv.c) * 0.6;
    return {
      type: "SimilarityLink",
      out: [link1.out[1], link2.out[1]],
      tv: { s, c },
      derivedFrom: [link1.handle, link2.handle]
    };
  },

  // Analogy: A→B, C→D ⇒ A~C, B~D (simple pattern mapping)
  analogy: (link1, link2) => {
    if (link1.type !== 'InheritanceLink' || link2.type !== 'InheritanceLink') return null;
    const s = Math.min(link1.tv.s, link2.tv.s) * 0.75;
    const c = Math.min(link1.tv.c, link2.tv.c) * 0.65;
    return [
      { type: "SimilarityLink", out: [link1.out[0], link2.out[0]], tv: { s, c } },
      { type: "SimilarityLink", out: [link1.out[1], link2.out[1]], tv: { s, c } }
    ];
  }
};

// Run PLN inference (try all rules on matching atoms)
async function plnInfer(pattern = {}) {
  await initHyperon();
  const atoms = await queryAtoms(pattern);

  let inferred = [];

  // Deduction
  const inheritanceLinks = atoms.filter(a => a.type === "InheritanceLink");
  for (let i = 0; i < inheritanceLinks.length; i++) {
    for (let j = i + 1; j < inheritanceLinks.length; j++) {
      const ded = PLN_RULES.deduction(inheritanceLinks[i], inheritanceLinks[j]);
      if (ded) inferred.push(ded);
    }
  }

  // Abduction & Induction (expand similarly)
  for (let i = 0; i < inheritanceLinks.length; i++) {
    for (let j = i + 1; j < inheritanceLinks.length; j++) {
      const abd = PLN_RULES.abduction(inheritanceLinks[i], inheritanceLinks[j]);
      if (abd) inferred.push(abd);
      const ind = PLN_RULES.induction(inheritanceLinks[i], inheritanceLinks[j]);
      if (ind) inferred.push(ind);
    }
  }

  // Analogy
  for (let i = 0; i < inheritanceLinks.length; i++) {
    for (let j = i + 1; j < inheritanceLinks.length; j++) {
      const ana = PLN_RULES.analogy(inheritanceLinks[i], inheritanceLinks[j]);
      if (ana) inferred.push(...ana);
    }
  }

  return inferred;
}

// Hyperon valence gate using atom-space + PLN inference
async function hyperonValenceGate(expression) {
  await initHyperon();
  const harmAtoms = await queryAtoms({ name: 'harm|kill|destroy' });
  const mercyAtoms = await queryAtoms({ name: 'mercy|truth' });

  let harmScore = harmAtoms.reduce((sum, a) => sum + (a.tv?.s || 0) * (a.tv?.c || 0), 0);
  let mercyScore = mercyAtoms.reduce((sum, a) => sum + (a.tv?.s || 0) * (a.tv?.c || 0), 0);

  // PLN inference boost
  const plnResults = await plnInfer({ type: "InheritanceLink" });
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.s * inf.tv.c * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.s * inf.tv.c * 0.3;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm dominates (score ${harmScore.toFixed(4)})` 
    : `Mercy prevails (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason
  };
}

export { initHyperon, queryAtoms, plnInfer, hyperonValenceGate };async function plnInfer(pattern) {
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

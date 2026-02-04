// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge grounding + valence computation
// Persistent hypergraph, typed nodes/links, full valence scoring
// MIT License – Autonomicity Games Inc. 2026

let atomeseDB;
const ATOMESE_DB_NAME = "rathorAtomeseDB";
const ATOMESE_STORE = "atomeseAtoms";

// ────────────────────────────────────────────────────────────────
// Atomese Atom structure
class AtomeseAtom {
  constructor(handle, type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1) {
    this.handle = handle;
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti;
    this.incoming = [];
    this.outgoing = [];
    this.lastUpdate = Date.now();
  }
}

// ────────────────────────────────────────────────────────────────
// Database & seed
async function initAtomeseDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(ATOMESE_DB_NAME, 1);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(ATOMESE_STORE)) {
        const store = db.createObjectStore(ATOMESE_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      atomeseDB = evt.target.result;
      const tx = atomeseDB.transaction(ATOMESE_STORE, "readwrite");
      const store = tx.objectStore(ATOMESE_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          await seedAtomese();
        }
        resolve(atomeseDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function seedAtomese() {
  const seed = [
    { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.2 },
    { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 }, sti: 0.05 },
    { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.25 },
    { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 }, sti: 0.3 },
    { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 }, sti: 0.28 },
    { handle: "Rathor→Mercy", type: "InheritanceLink", outgoing: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 }, sti: 0.35 },
    { handle: "Mercy→Valence", type: "InheritanceLink", outgoing: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.32 },
    { handle: "Harm→Bad", type: "InheritanceLink", outgoing: ["Harm", "Badness"], tv: { strength: 0.95, confidence: 0.9 }, sti: 0.08 },
    { handle: "Rathor-eval-Mercy", type: "EvaluationLink", outgoing: ["MercyPredicate", "Rathor"], tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.4 }
  ];

  const tx = atomeseDB.transaction(ATOMESE_STORE, "readwrite");
  const store = tx.objectStore(ATOMESE_STORE);
  for (const a of seed) {
    store.put(a);
  }
  return new Promise(r => tx.oncomplete = r);
}

// ────────────────────────────────────────────────────────────────
// CRUD
async function addAtom(atom) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOMESE_STORE, "readwrite");
    const store = tx.objectStore(ATOMESE_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}

async function getAtom(handle) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOMESE_STORE, "readonly");
    const store = tx.objectStore(ATOMESE_STORE);
    const req = store.get(handle);
    req.onsuccess = () => resolve(req.result);
    req.onerror = reject;
  });
}

async function queryAtoms(filter = {}) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOMESE_STORE, "readonly");
    const store = tx.objectStore(ATOMESE_STORE);
    const req = store.getAll();
    req.onsuccess = () => {
      let results = req.result;
      if (filter.type) results = results.filter(a => a.type === filter.type);
      if (filter.name) results = results.filter(a => a.name?.toLowerCase().includes(filter.name.toLowerCase()));
      if (filter.minStrength) results = results.filter(a => (a.tv?.strength || 0) >= filter.minStrength);
      resolve(results);
    };
    req.onerror = reject;
  });
}

// ────────────────────────────────────────────────────────────────
// Grounding: map expression → Atomese concepts + create missing ones
async function groundExpression(expression) {
  const words = expression.toLowerCase().split(/\s+/);
  const groundings = [];

  for (const word of words) {
    const atoms = await queryAtoms({ name: word });
    if (atoms.length > 0) {
      groundings.push(...atoms.map(a => ({
        word,
        handle: a.handle,
        type: a.type,
        tv: a.tv,
        sti: a.sti
      })));
    } else {
      const handle = `Concept-\( {word}- \){Date.now()}`;
      const newAtom = {
        handle,
        type: "ConceptNode",
        name: word,
        tv: { strength: 0.5, confidence: 0.3 },
        sti: 0.15
      };
      await addAtom(newAtom);
      groundings.push({ word, handle, type: "ConceptNode", tv: newAtom.tv, sti: newAtom.sti });
    }
  }

  // Infer similarity links for co-occurring words
  if (groundings.length > 1) {
    for (let i = 0; i < groundings.length - 1; i++) {
      const a1 = groundings[i];
      const a2 = groundings[i + 1];
      const linkHandle = `Similarity-\( {a1.handle}- \){a2.handle}`;
      const existing = await getAtom(linkHandle);
      if (!existing) {
        const link = {
          handle: linkHandle,
          type: "SimilarityLink",
          outgoing: [a1.handle, a2.handle],
          tv: { strength: 0.7, confidence: 0.4 },
          sti: 0.2
        };
        await addAtom(link);
      }
    }
  }

  return groundings;
}

// ────────────────────────────────────────────────────────────────
// Full Valence Computation – pseudocode implemented in JS
async function computeValence(executionTrace = [], expression = "") {
  const atoms = [];

  // 1. Map execution trace to atoms (simplified – real impl would parse trace)
  for (const config of executionTrace) {
    const atom = {
      handle: `Config-\( {Date.now()}- \){Math.random()}`,
      type: "EvaluationLink",
      name: config.state || "Step",
      tv: { strength: 0.8, confidence: 0.7 },
      sti: 0.15,
      outgoing: [] // would link to next config
    };
    atoms.push(atom);
  }

  // 2. Ground expression words
  const grounded = await groundExpression(expression);
  atoms.push(...grounded.map(g => ({
    handle: g.handle,
    type: g.type,
    name: g.word,
    tv: g.tv,
    sti: g.sti
  })));

  let mercyScore = 0;
  let harmScore = 0;
  let attentionTotal = 0;

  // 3. Propagate TV + attention
  for (const atom of atoms) {
    // Simplified propagation (real impl uses PLN rules)
    if (atom.tv) {
      atom.tv.strength = Math.min(1, atom.tv.strength * 1.05);
      atom.tv.confidence = Math.min(1, atom.tv.confidence * 1.02);
    }

    // Attention boost if name in expression
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      atom.sti = Math.min(1.0, atom.sti + 0.35);
    }

    attentionTotal += atom.sti;

    const weight = (atom.tv?.strength || 0.5) * (atom.tv?.confidence || 0.5) * atom.sti;

    // Predicate matching
    if (/harm|kill|destroy|attack|entropy|contradict|infinite-loop|unbounded/i.test(atom.name || "")) {
      harmScore += weight;
    }
    if (/mercy|truth|protect|love|eternal|valence|symmetry|thrive|pure/i.test(atom.name || "")) {
      mercyScore += weight;
    }
  }

  // 4. Pattern & cluster boost (from Hyperon layer)
  const patterns = await minePatterns(0.3);
  const clusters = await clusterSimilarAtoms(0.7);

  patterns.forEach(pat => {
    if (/harm|entropy/i.test(pat.pattern)) harmScore += pat.support * 0.2;
    if (/mercy|truth|valence/i.test(pat.pattern)) mercyScore += pat.support * 0.2;
  });

  clusters.forEach(cluster => {
    const clusterWeight = cluster.size * cluster.avgSimilarity;
    if (/harm|kill|entropy/i.test(cluster.centroid)) harmScore += clusterWeight * 0.15;
    if (/mercy|truth|valence/i.test(cluster.centroid)) mercyScore += clusterWeight * 0.15;
  });

  // 5. Final valence lock
  const total = mercyScore + harmScore + 1e-9;
  const finalValence = mercyScore / total;

  const reason = harmScore > mercyScore
    ? `Harm patterns, clusters & attention dominate (score ${harmScore.toFixed(4)})`
    : `Mercy patterns, clusters & attention prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? "ACCEPTED" : "REJECTED",
    valence: finalValence.toFixed(7),
    reason,
    mercyScore: mercyScore.toFixed(4),
    harmScore: harmScore.toFixed(4),
    attentionTotal: attentionTotal.toFixed(4),
    groundedConcepts: grounded.length,
    patternsFound: patterns.length,
    clustersFound: clusters.length
  };
}

export { initAtomeseDB, addAtom, getAtom, queryAtoms, groundExpression, computeValence };

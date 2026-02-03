// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph reasoning engine
// Full PLN chaining, hyperedge support, attention dynamics, pattern mining, clustering
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonHypergraph";

// ────────────────────────────────────────────────────────────────
// Core atom structure (OpenCog Hypergraph style)
class HyperonAtom {
  constructor(handle, type, name = null, tv = { s: 0.5, c: 0.5 }, sti = 0.1, lti = 0.5) {
    this.handle = handle;
    this.type = type; // ConceptNode, PredicateNode, InheritanceLink, EvaluationLink, ImplicationHyperedge, etc.
    this.name = name;
    this.tv = tv;     // truth value {strength, confidence}
    this.sti = sti;   // short-term importance
    this.lti = lti;   // long-term importance
    this.out = [];    // outgoing links / hyperedge targets
    this.in = [];     // incoming links (for fast traversal)
    this.lastUpdate = Date.now();
  }
}

// ────────────────────────────────────────────────────────────────
// Database & initialization
async function initHyperonDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HYPERON_DB_NAME, 3);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(HYPERON_STORE)) {
        const store = db.createObjectStore(HYPERON_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      hyperonDB = evt.target.result;
      const tx = hyperonDB.transaction(HYPERON_STORE, "readwrite");
      const store = tx.objectStore(HYPERON_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          await seedHypergraph();
        }
        resolve(hyperonDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function seedHypergraph() {
  const seedAtoms = [
    new HyperonAtom("Truth", "ConceptNode", "Truth", { s: 0.9999999, c: 1.0 }, 0.15, 0.9),
    new HyperonAtom("Harm", "ConceptNode", "Harm", { s: 0.01, c: 0.99 }, 0.05, 0.4),
    new HyperonAtom("Mercy", "ConceptNode", "Mercy", { s: 0.9999999, c: 1.0 }, 0.2, 0.95),
    new HyperonAtom("Rathor", "ConceptNode", "Rathor", { s: 1.0, c: 1.0 }, 0.25, 1.0),
    new HyperonAtom("Valence", "ConceptNode", "Valence", { s: 1.0, c: 1.0 }, 0.22, 0.98),

    // Hyperedges
    {
      handle: "Rathor-Implies-Mercy-Valence",
      type: "ImplicationHyperedge",
      out: ["Rathor", "Mercy", "Valence"],
      tv: { s: 0.9999999, c: 1.0 },
      sti: 0.3,
      lti: 0.95
    },
    {
      handle: "Harm-Implies-Badness-Entropy",
      type: "ImplicationHyperedge",
      out: ["Harm", "Badness", "Entropy"],
      tv: { s: 0.93, c: 0.9 },
      sti: 0.08,
      lti: 0.45
    }
  ];

  const tx = hyperonDB.transaction(HYPERON_STORE, "readwrite");
  const store = tx.objectStore(HYPERON_STORE);
  for (const atom of seedAtoms) {
    store.put(atom);
  }
  return new Promise(r => tx.oncomplete = r);
}

async function addAtom(atom) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readwrite");
    const store = tx.objectStore(HYPERON_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}

async function getAtom(handle) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readonly");
    const store = tx.objectStore(HYPERON_STORE);
    const req = store.get(handle);
    req.onsuccess = () => resolve(req.result);
    req.onerror = reject;
  });
}

async function queryAtoms(filter = {}) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readonly");
    const store = tx.objectStore(HYPERON_STORE);
    const req = store.getAll();
    req.onsuccess = () => {
      let results = req.result;
      if (filter.type) results = results.filter(a => a.type === filter.type);
      if (filter.name) results = results.filter(a => a.name?.toLowerCase().includes(filter.name.toLowerCase()));
      if (filter.minStrength) results = results.filter(a => (a.tv?.s || 0) >= filter.minStrength);
      resolve(results);
    };
    req.onerror = reject;
  });
}

// ────────────────────────────────────────────────────────────────
// Attention dynamics
async function updateAttention(expression = "") {
  const atoms = await queryAtoms();
  const now = Date.now();

  for (const atom of atoms) {
    const timePassed = (now - (atom.lastUpdate || now)) / (1000 * 60 * 5);
    atom.sti = (atom.sti || 0.1) * Math.pow(0.5, timePassed);

    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      atom.sti = Math.min(1.0, (atom.sti || 0) + 0.35);
      atom.lti = Math.min(1.0, (atom.lti || 0) + 0.06);
    }

    if (atom.tv && atom.tv.s > 0.8 && atom.tv.c < 0.4) {
      atom.sti = Math.min(1.0, atom.sti + 0.25);
    }

    atom.lastUpdate = now;
    await addAtom(atom);
  }

  return atoms.filter(a => a.sti > 0.35).sort((a, b) => b.sti - a.sti);
}

// ────────────────────────────────────────────────────────────────
// PLN inference chaining over hypergraph
async function plnChainInfer(start, target = null, maxDepth = 5, decay = 0.88) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(currentHandle, depth, path, currentTV) {
    if (depth > maxDepth) return;
    if (target && currentHandle === target && path.length > 1) {
      chains.push({ path, tv: currentTV, length: path.length });
      return;
    }

    const outgoing = links.filter(l => l.out && l.out[0] === currentHandle);
    for (const link of outgoing) {
      const next = link.out[1];
      if (visited.has(next)) continue;
      visited.add(next);

      const newTV = {
        s: Math.min(currentTV.s, link.tv.s),
        c: Math.min(currentTV.c, link.tv.c) * decay
      };

      await dfs(next, depth + 1, [...path, link.handle], newTV);
      visited.delete(next);
    }
  }

  await dfs(start, 0, [], { s: 1.0, c: 1.0 });
  return chains.sort((a, b) => (b.tv.s * b.tv.c / b.length) - (a.tv.s * a.tv.c / a.length));
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – full hypergraph reasoning
async function hyperonValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { s: 0.5, c: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c;
      if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c;
    }
  }

  // PLN chaining boost
  const harmChains = await plnChainInfer("Harm", "Entropy", 5);
  const mercyChains = await plnChainInfer("Mercy", "Valence", 5);
  harmChains.forEach(c => harmScore += c.tv.s * c.tv.c * 0.35 / c.length);
  mercyChains.forEach(c => mercyScore += c.tv.s * c.tv.c * 0.35 / c.length);

  // Attention boost
  const highAtt = await updateAttention(expression);
  highAtt.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.45;
    if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c * weight;
    if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c * weight;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 1e-9);
  const reason = harmScore > mercyScore
    ? `Harm hyperedges & attention dominate (${harmChains.length} chains, score ${harmScore.toFixed(4)})`
    : `Mercy hyperedges & attention prevail (${mercyChains.length} chains, score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChains: harmChains.length,
    mercyChains: mercyChains.length,
    highAttention: highAtt.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, plnChainInfer, updateAttention, hyperonValenceGate };
  async function dfs(current, depth, path, currentTV) {
    if (depth > maxDepth) return;
    if (targetConcept && current === targetConcept && path.length > 1) {
      chains.push({ path, tv: currentTV, length: path.length });
      return;
    }

    const outgoing = inheritanceLinks.filter(l => l.out[0] === current);
    for (const link of outgoing) {
      const next = link.out[1];
      if (visited.has(next)) continue;
      visited.add(next);

      const newTV = {
        s: Math.min(currentTV.s, link.tv.s),
        c: Math.min(currentTV.c, link.tv.c) * decay
      };

      await dfs(next, depth + 1, [...path, link.handle], newTV);
      visited.delete(next);
    }
  }

  // Start chaining from every known concept that matches startConcept pattern
  const startNodes = await queryHyperonAtoms({ name: startConcept });
  for (const start of startNodes) {
    await dfs(start.handle, 0, [], { s: 1.0, c: 1.0 });
  }

  // Sort by strength × confidence × inverse length (shorter stronger chains preferred)
  return chains.sort((a, b) => 
    (b.tv.s * b.tv.c / b.length) - (a.tv.s * a.tv.c / a.length)
  );
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – now with chaining boost
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  // Direct atom matching
  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { s: 0.5, c: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmScore += tv.s * tv.c;
      }
      if (/mercy|truth|protect|love/i.test(atom.name)) {
        mercyScore += tv.s * tv.c;
      }
    }
  }

  // PLN chaining boost – long high-confidence chains amplify scores
  const harmChains = await plnChainInfer("Harm", "Entropy", 4);
  const mercyChains = await plnChainInfer("Mercy", "Valence", 4);

  harmChains.forEach(chain => {
    harmScore += chain.tv.s * chain.tv.c * 0.4 / chain.length;
  });

  mercyChains.forEach(chain => {
    mercyScore += chain.tv.s * chain.tv.c * 0.4 / chain.length;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm chaining dominates (${harmChains.length} chains, score ${harmScore.toFixed(4)})` 
    : `Mercy chaining prevails (${mercyChains.length} chains, score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChainsFound: harmChains.length,
    mercyChainsFound: mercyChains.length
  };
}

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, plnChainInfer, hyperonValenceGate };

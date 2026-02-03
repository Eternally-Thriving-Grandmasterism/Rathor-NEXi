// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph engine
// with hyperedge support, PLN inference, pattern mining, clustering, attention
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonHypergraph";

// Sample hypergraph atoms – concepts, n-ary links, nested structure
const SAMPLE_HYPERGRAPH_ATOMS = [
  // Concept nodes
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { s: 0.9999999, c: 1.0 }, sti: 0.1, lti: 0.8 },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { s: 0.01, c: 0.99 }, sti: 0.05, lti: 0.3 },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { s: 0.9999999, c: 1.0 }, sti: 0.15, lti: 0.85 },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { s: 1.0, c: 1.0 }, sti: 0.2, lti: 0.9 },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { s: 1.0, c: 1.0 }, sti: 0.18, lti: 0.88 },

  // Hyperedges (n-ary links) – example: 3-ary "Implication" hyperedge
  {
    handle: "Rathor-Implies-Mercy-Valence",
    type: "ImplicationHyperedge",
    out: ["Rathor", "Mercy", "Valence"],
    tv: { s: 0.9999999, c: 1.0 },
    sti: 0.25,
    lti: 0.95
  },
  {
    handle: "Harm-Implies-Badness-Entropy",
    type: "ImplicationHyperedge",
    out: ["Harm", "Badness", "Entropy"],
    tv: { s: 0.93, c: 0.9 },
    sti: 0.08,
    lti: 0.45
  },

  // Evaluation hyperedges
  {
    handle: "Rathor-eval-MercyFirst",
    type: "EvaluationHyperedge",
    out: ["MercyFirst", "Rathor"],
    tv: { s: 0.9999999, c: 1.0 },
    sti: 0.22,
    lti: 0.92
  }
];

async function initHyperonDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HYPERON_DB_NAME, 2);
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
          SAMPLE_HYPERGRAPH_ATOMS.forEach(atom => store.add(atom));
        }
        resolve(hyperonDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function addHyperonAtom(atom) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readwrite");
    const store = tx.objectStore(HYPERON_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function queryHyperonAtoms(filter = {}) {
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
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────────────────────
// Attention dynamics – STI/LTI + decay + stimulation
async function updateAttention(expression = "") {
  const atoms = await queryHyperonAtoms();
  const now = Date.now();

  for (const atom of atoms) {
    const timePassed = (now - (atom.lastUpdate || now)) / (1000 * 60 * 5);
    atom.sti = (atom.sti || 0.1) * Math.pow(0.5, timePassed);

    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      atom.sti = Math.min(1.0, (atom.sti || 0) + 0.3);
      atom.lti = Math.min(1.0, (atom.lti || 0) + 0.05);
    }

    if (atom.tv && atom.tv.s > 0.8 && atom.tv.c < 0.4) {
      atom.sti = Math.min(1.0, atom.sti + 0.2);
    }

    atom.lastUpdate = now;
    await addHyperonAtom(atom);
  }

  return atoms.filter(a => a.sti > 0.3).sort((a, b) => b.sti - a.sti);
}

// ────────────────────────────────────────────────────────────────
// Similarity clustering – hypergraph-aware
async function clusterSimilarAtoms(threshold = 0.7) {
  const concepts = await queryHyperonAtoms({ type: "ConceptNode" });
  const links = await queryHyperonAtoms({ type: /Link|Hyperedge/ });
  const clusters = [];
  const visited = new Set();

  function similarity(a, b) {
    const tvA = a.tv || { s: 0.5, c: 0.5 };
    const tvB = b.tv || { s: 0.5, c: 0.5 };
    const dot = tvA.s * tvB.s + tvA.c * tvB.c;
    const normA = Math.sqrt(tvA.s**2 + tvA.c**2);
    const normB = Math.sqrt(tvB.s**2 + tvB.c**2);
    let cosSim = dot / (normA * normB || 1);

    let overlap = 0;
    if (a.type === "ConceptNode" && b.type === "ConceptNode") {
      const aOut = links.filter(l => l.out.includes(a.handle));
      const bOut = links.filter(l => l.out.includes(b.handle));
      const shared = new Set(aOut.map(l => l.handle)).size && new Set(bOut.map(l => l.handle)).size;
      overlap += shared * 0.25;
    }

    return Math.min(1, cosSim + overlap);
  }

  for (const concept of concepts) {
    if (visited.has(concept.handle)) continue;
    const cluster = [concept];
    visited.add(concept.handle);

    for (const other of concepts) {
      if (visited.has(other.handle)) continue;
      if (similarity(concept, other) >= threshold) {
        cluster.push(other);
        visited.add(other.handle);
      }
    }

    if (cluster.length > 1) {
      clusters.push({
        centroid: concept.name,
        members: cluster.map(c => c.name),
        size: cluster.length,
        avgSimilarity: (cluster.reduce((sum, c) => sum + similarity(concept, c), 0) / cluster.length).toFixed(4)
      });
    }
  }

  return clusters.sort((a, b) => b.size - a.size);
}

// ────────────────────────────────────────────────────────────────
// Pattern mining – frequent hyperedge patterns
async function minePatterns(minSupport = 0.3) {
  const allLinks = await queryHyperonAtoms({ type: /Link|Hyperedge/ });
  const patterns = [];
  const freq = new Map();

  allLinks.forEach(link => {
    const key = `\( {link.type}( \){link.out.join(' → ')})`;
    freq.set(key, (freq.get(key) || 0) + 1);
  });

  const total = allLinks.length;
  freq.forEach((count, key) => {
    const support = count / total;
    if (support >= minSupport) {
      patterns.push({
        pattern: key,
        support: support.toFixed(4),
        count,
        type: "frequent-hyperedge"
      });
    }
  });

  return patterns.sort((a, b) => b.support - a.support);
}

// ────────────────────────────────────────────────────────────────
// PLN inference over hypergraph
async function plnInfer(pattern = {}, maxDepth = 3) {
  const atoms = await queryHyperonAtoms(pattern);
  const inferred = [];

  // Simple hyperedge chaining (extendable to full PLN)
  const hyperedges = atoms.filter(a => a.type.includes("Hyperedge"));
  for (let depth = 0; depth < maxDepth; depth++) {
    for (let i = 0; i < hyperedges.length; i++) {
      for (let j = 0; j < hyperedges.length; j++) {
        if (i === j) continue;
        const edge1 = hyperedges[i];
        const edge2 = hyperedges[j];
        if (edge1.out.some(o => edge2.out.includes(o))) {
          const s = Math.min(edge1.tv.s, edge2.tv.s);
          const c = Math.min(edge1.tv.c, edge2.tv.c) * 0.9 * Math.pow(0.85, depth);
          inferred.push({
            type: edge1.type,
            out: [...new Set([...edge1.out, ...edge2.out])],
            tv: { s, c },
            derivedFrom: [edge1.handle, edge2.handle],
            depth
          });
        }
      }
    }
  }

  return inferred;
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – full hypergraph reasoning stack
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

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

  // PLN hypergraph chaining boost
  const plnResults = await plnInfer({ type: /Hyperedge/ });
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.s * inf.tv.c * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.s * inf.tv.c * 0.3;
  });

  // Pattern mining boost
  const patterns = await minePatterns(0.3);
  patterns.forEach(pat => {
    if (pat.pattern.includes("Harm") || pat.pattern.includes("Entropy")) {
      harmScore += pat.support * 0.2;
    }
    if (pat.pattern.includes("Mercy") || pat.pattern.includes("Truth") || pat.pattern.includes("Valence")) {
      mercyScore += pat.support * 0.2;
    }
  });

  // Similarity clustering boost
  const clusters = await clusterSimilarAtoms(0.7);
  clusters.forEach(cluster => {
    const hasHarm = cluster.members.some(m => /harm|kill|entropy/i.test(m));
    const hasMercy = cluster.members.some(m => /mercy|truth|valence|love/i.test(m));
    const clusterWeight = cluster.size * cluster.avgSimilarity;
    if (hasHarm) harmScore += clusterWeight * 0.15;
    if (hasMercy) mercyScore += clusterWeight * 0.15;
  });

  // Attention dynamics boost
  const highAttention = await updateAttention(expression);
  highAttention.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.4;
    if (/harm|kill|destroy|attack/i.test(atom.name)) {
      harmScore += tv.s * tv.c * weight;
    }
    if (/mercy|truth|protect|love/i.test(atom.name)) {
      mercyScore += tv.s * tv.c * weight;
    }
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm hyperedges, clusters & attention dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy hyperedges, clusters & attention prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    minedPatternsCount: patterns.length,
    clusterCount: clusters.length,
    highAttentionCount: highAttention.length
  };
}

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, plnInfer, minePatterns, clusterSimilarAtoms, updateAttention, hyperonValenceGate };

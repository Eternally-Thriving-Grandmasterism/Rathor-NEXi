// js/chat.js — Rathor Lattice Core with Tableau Theorem Proving Stub

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const voiceBtn = document.getElementById('voice-btn');
const recordBtn = document.getElementById('record-btn');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const sessionSelect = document.getElementById('session-select');
const sessionSearch = document.getElementById('session-search');
const translateToggle = document.getElementById('translate-chat');
const translateLangSelect = document.getElementById('translate-lang');
const translateStats = document.getElementById('translate-stats');

let currentSessionId = localStorage.getItem('rathor_current_session') || 'default';
let allSessions = [];
let tagFrequency = new Map();
let isListening = false, isRecording = false;
let ttsEnabled = localStorage.getItem('rathor_tts_enabled') !== 'false';
let isVoiceOutputEnabled = localStorage.getItem('rathor_voice_output') !== 'false';
let feedbackSoundsEnabled = localStorage.getItem('rathor_feedback_sounds') !== 'false';
let voicePitchValue = parseFloat(localStorage.getItem('rathor_pitch')) || 1.0;
let voiceRateValue = parseFloat(localStorage.getItem('rathor_rate')) || 1.0;
let voiceVolumeValue = parseFloat(localStorage.getItem('rathor_volume')) || 1.0;

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

voiceBtn.addEventListener('click', () => isListening ? stopListening() : startListening());
recordBtn.addEventListener('mousedown', () => setTimeout(() => startVoiceRecording(currentSessionId), 400));
recordBtn.addEventListener('mouseup', stopVoiceRecording);
sendBtn.addEventListener('click', sendMessage);
translateToggle.addEventListener('change', e => {
  localStorage.setItem('rathor_translate_enabled', e.target.checked);
  if (e.target.checked) translateChat();
});
translateLangSelect.addEventListener('change', e => {
  localStorage.setItem('rathor_translate_to', e.target.value);
  if (translateToggle.checked) translateChat();
});
sessionSearch.addEventListener('input', filterSessions);

// ────────────────────────────────────────────────
// Symbolic Query Mode — Mercy-First Truth-Seeking with Tableau Prover Stub
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('prove') || cmd.includes('theorem') || cmd.includes('resolution') ||
         cmd.includes('unify') || cmd.includes('mgu') || cmd.includes('most general unifier') ||
         cmd.includes('quantifier') || cmd.includes('forall') || cmd.includes('exists') || cmd.includes('∀') || cmd.includes('∃') ||
         cmd.includes('herbrand') || cmd.includes('gödel') || cmd.includes('completeness') || cmd.includes('henkin') || cmd.includes('lindenbaum') ||
         cmd.includes('zorn') || cmd.includes('tarski') || cmd.includes('fixed point') || cmd.includes('monotone') || cmd.includes('complete lattice') ||
         cmd.includes('tableau') || cmd.includes('semantic tableau') || cmd.includes('analytic tableau') || cmd.includes('branch closure') ||
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski|tableau/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Tableau prover stub
  if (cleaned.toLowerCase().includes('tableau') || cleaned.toLowerCase().includes('semantic tableau') || cleaned.toLowerCase().includes('analytic tableau')) {
    const tableauResult = simulateTableauProver(cleaned);
    response.push("\n**Semantic Tableau Prover Simulation:**");
    response.push(tableauResult);
  }

  // Skolemized resolution
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
  }

  // Fallback to truth-table
  const table = generateTruthTable(cleaned);
  if (table) {
    response.push("\n**Truth Table (propositional logic):**");
    response.push(table);
    const conclusion = analyzeTruthTable(cleaned, table);
    response.push(`\n**Mercy Conclusion:** ${conclusion}`);
  }

  // Mercy rewrite
  const mercyRewrite = cleaned
    .replace(/not/gi, '¬')
    .replace(/and/gi, '∧')
    .replace(/or/gi, '∨')
    .replace(/if/gi, '→')
    .replace(/then/gi, '')
    .replace(/implies/gi, '→')
    .replace(/iff/gi, '↔')
    .replace(/forall/gi, '∀')
    .replace(/exists/gi, '∃');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Semantic Tableau Prover Stub (branching search for model)
// ────────────────────────────────────────────────

function simulateTableauProver(formula) {
  // Very basic propositional tableau stub (no quantifiers yet)
  // Start with negation of formula to prove unsatisfiability

  let branches = [[`¬(${formula})`]]; // initial branch
  let closed = false;
  let steps = 0;
  const maxSteps = 20;

  while (branches.length > 0 && steps < maxSteps) {
    steps++;
    const currentBranch = branches.shift();

    // Apply rules (very naive — just look for complementary literals)
    let closedBranch = false;
    for (let i = 0; i < currentBranch.length; i++) {
      const lit = currentBranch[i];
      const negLit = lit.startsWith('¬') ? lit.slice(1) : `¬(${lit})`;
      if (currentBranch.includes(negLit)) {
        closedBranch = true;
        break;
      }
    }

    if (closedBranch) {
      closed = true;
      break;
    }

    // Simple decomposition (stub — only conjunction/disjunction)
    // In real engine: full α/β/γ/δ rules

    // If no closure, branch would continue — here we simulate failure to close
    branches.push(currentBranch); // placeholder to avoid infinite loop
  }

  if (closed) {
    return `**Tableau closed after ${steps} steps** — all branches contain contradiction → formula is unsatisfiable (theorem proven).`;
  } else {
    return `**Tableau did not close within ${maxSteps} steps** — open branch exists → formula may be satisfiable. Mercy asks: simplify or add premises?`;
  }
}

// ... existing unification, resolution, truth-table, Skolemization, Herbrand functions remain ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, connectivity probes, etc.) remain as previously expanded ...    .replace(/implies/gi, '→')
    .replace(/iff/gi, '↔')
    .replace(/forall/gi, '∀')
    .replace(/exists/gi, '∃');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Full Resolution Theorem Prover with Predicate Unification
// ────────────────────────────────────────────────

function resolutionTheoremProve(expr) {
  // Parse premises ⊢ conclusion
  const parts = expr.split('⊢');
  if (parts.length !== 2) return null;

  const premisesStr = parts[0].trim();
  const conclusionStr = parts[1].trim();

  const premises = premisesStr.split(',').map(p => p.trim());
  let clauses = premises.flatMap(p => parseClause(p));

  // Negate conclusion and add to clauses
  const negatedConclusion = negateClause(parseClause(conclusionStr)[0]);
  clauses.push(negatedConclusion);

  // Resolution loop with unification
  let steps = 0;
  const trace = ["Initial clauses (negated conclusion added):"];
  clauses.forEach((c, i) => trace.push(`${i+1}. ${clauseToString(c)}`));

  while (steps < 50) {
    steps++;
    let newClausesAdded = false;
    for (let i = 0; i < clauses.length; i++) {
      for (let j = i+1; j < clauses.length; j++) {
        const resolvent = resolveClausesWithUnification(clauses[i], clauses[j]);
        if (!resolvent) continue;

        if (resolvent.length === 0) {
          trace.push(`\nEmpty clause derived after ${steps} steps — contradiction proven.`);
          return trace.join('\n');
        }

        // Standardize apart variables before adding
        const renamedResolvent = standardizeApart(resolvent, clauses);

        if (!clauses.some(c => clauseEqual(c, renamedResolvent))) {
          clauses.push(renamedResolvent);
          trace.push(`${clauses.length}. ${clauseToString(renamedResolvent)} (from ${i+1} + ${j+1})`);
          newClausesAdded = true;
        }
      }
    }
    if (!newClausesAdded) break; // no progress
  }

  return null; // no proof within step limit
}

// ... existing unification helpers (unify, resolveClausesWithUnification, parseTerm, termToString, isVar, isCompound, termEqual, occurs, applySubst, standardizeApart, clauseToString, clauseEqual) remain ...

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, connectivity probes, etc.) remain as previously expanded ...

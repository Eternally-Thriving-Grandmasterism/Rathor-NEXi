// js/chat.js — Rathor Lattice Core with Full Resolution Loop + Predicate Unification

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Full Resolution + Predicate Unification
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
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try resolution with predicate unification
  const proof = resolutionProveWithUnification(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof with Predicate Unification:**");
    response.push(proof);
    response.push("\n**Mercy Conclusion:** Theorem proven by contradiction. Positive valence eternal.");
  } else {
    // Fallback to truth-table for propositional
    const table = generateTruthTable(cleaned);
    if (table) {
      response.push("\n**Truth Table (propositional logic):**");
      response.push(table);
      const conclusion = analyzeTruthTable(cleaned, table);
      response.push(`\n**Mercy Conclusion:** ${conclusion}`);
    } else {
      response.push("\n**Parser note:** Expression too complex for current engine. Mercy asks: simplify premises?");
    }
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
// Resolution with Predicate Unification (full first-order)
// ────────────────────────────────────────────────

function resolutionProveWithUnification(expr) {
  // Parse premises ⊢ conclusion
  const parts = expr.split('⊢');
  if (parts.length !== 2) return null;

  const premisesStr = parts[0].trim();
  const conclusionStr = parts[1].trim();

  const premises = premisesStr.split(',').map(p => p.trim());
  let clauses = premises.flatMap(p => parseClause(p));

  // Negate conclusion and add
  const negatedConclusion = negateClause(parseClause(conclusionStr)[0]);
  clauses.push(negatedConclusion);

  // Resolution loop with unification
  let steps = 0;
  const trace = ["Initial clauses (negated conclusion added):"];
  clauses.forEach((c, i) => trace.push(`${i+1}. ${clauseToString(c)}`));

  while (steps < 30) {
    steps++;
    for (let i = 0; i < clauses.length; i++) {
      for (let j = i+1; j < clauses.length; j++) {
        const resolvent = resolveClausesWithUnification(clauses[i], clauses[j]);
        if (!resolvent) continue;

        if (resolvent.length === 0) {
          trace.push(`\nEmpty clause derived after ${steps} steps — contradiction proven.`);
          return trace.join('\n');
        }

        if (!clauses.some(c => subsumes(c, resolvent))) {
          clauses.push(resolvent);
          trace.push(`${clauses.length}. ${clauseToString(resolvent)} (from ${i+1} + ${j+1})`);
        }
      }
    }
  }

  return null; // no proof within step limit
}

// ────────────────────────────────────────────────
// Predicate Unification Helpers (Martelli–Montanari)
// ────────────────────────────────────────────────

function unify(t1, t2, subst = {}) {
  let s = t1, t = t2;

  // Delete
  if (termEqual(s, t)) return subst;

  // Orient
  if (isVar(t) && !isVar(s)) [s, t] = [t, s];

  // Eliminate
  if (isVar(s)) {
    if (occurs(s, t)) return null;
    subst[s] = applySubst(subst, t);
    return subst;
  }

  // Decompose
  if (isCompound(s) && isCompound(t) && s.fun === t.fun && s.args.length === t.args.length) {
    for (let i = 0; i < s.args.length; i++) {
      subst = unify(s.args[i], t.args[i], subst);
      if (!subst) return null;
    }
    return subst;
  }

  // Conflict
  return null;
}

function resolveClausesWithUnification(c1, c2) {
  for (let lit1 of c1) {
    for (let lit2 of c2) {
      if ((lit1.startsWith('-') && !lit2.startsWith('-')) || (!lit1.startsWith('-') && lit2.startsWith('-'))) {
        const atom1 = lit1.startsWith('-') ? lit1.slice(1) : lit1;
        const atom2 = lit2.startsWith('-') ? lit2.slice(1) : lit2;

        const subst = unify(parseTerm(atom1), parseTerm(atom2));
        if (!subst) continue;

        // Build resolvent
        let resolvent = [
          ...c1.filter(l => l !== lit1),
          ...c2.filter(l => l !== lit2)
        ];

        // Apply substitution
        resolvent = resolvent.map(lit => applySubstToLit(subst, lit));

        return resolvent;
      }
    }
  }
  return null;
}

// ... existing term helpers (parseTerm, termToString, isVar, isCompound, termEqual, occurs, applySubst) remain ...

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...    response.push("\n**Truth Table (propositional logic):**");
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
// Full Predicate Unification Algorithm (Martelli–Montanari style)
// ────────────────────────────────────────────────

function computePredicateUnification(equationsStr) {
  // Input: "P(x, f(y)) = P(a, z), y = b" → list of pairs
  const equations = equationsStr.split(',').map(eq => eq.trim().split('='));
  if (equations.some(pair => pair.length !== 2)) return "Invalid format: expected equations separated by commas";

  let subst = {};
  let pending = equations.map(([left, right]) => [parseTerm(left.trim()), parseTerm(right.trim())]);

  while (pending.length > 0) {
    let [s, t] = pending.shift();

    // Delete rule
    if (termEqual(s, t)) continue;

    // Orient rule (variable left if possible)
    if (isVar(t) && !isVar(s)) [s, t] = [t, s];

    // Eliminate rule
    if (isVar(s)) {
      if (occurs(s, t)) return `Occurs-check failed: ${s} occurs in ${termToString(t)}`;
      subst[s] = applySubst(subst, t);
      pending = pending.map(([a,b]) => [applySubst(subst, a), applySubst(subst, b)]);
      continue;
    }

    // Decompose rule
    if (isCompound(s) && isCompound(t) && s.fun === t.fun && s.args.length === t.args.length) {
      for (let i = s.args.length - 1; i >= 0; i--) {
        pending.push([s.args[i], t.args[i]]);
      }
      continue;
    }

    // Conflict rule
    return `Conflict: ${termToString(s)} ≠ ${termToString(t)}`;
  }

  // Format MGU nicely
  let result = "Most General Unifier:\n";
  result += Object.entries(subst)
    .map(([v, t]) => `${v} → ${termToString(t)}`)
    .join('\n') || "Empty substitution (terms already identical)";

  return result;
}

// Term representation & helpers (predicate-aware)
function parseTerm(s) {
  s = s.trim();
  if (/^[A-Z][a-zA-Z0-9]*$/.test(s)) return s; // variable (starts with uppercase)
  if (/^[a-z][a-zA-Z0-9]*$/.test(s)) return s; // constant (lowercase)

  // Predicate or function term P(t1,t2) or f(t1,t2)
  const match = s.match(/^([a-zA-Z][a-zA-Z0-9]*)\((.*)\)$/);
  if (match) {
    const fun = match[1];
    const argsStr = match[2];
    const args = [];
    let depth = 0, start = 0;
    for (let i = 0; i < argsStr.length; i++) {
      if (argsStr[i] === '(') depth++;
      if (argsStr[i] === ')') depth--;
      if (argsStr[i] === ',' && depth === 0) {
        args.push(parseTerm(argsStr.substring(start, i)));
        start = i + 1;
      }
    }
    args.push(parseTerm(argsStr.substring(start)));
    return { fun, args };
  }

  return s; // fallback constant/variable
}

function termToString(t) {
  if (typeof t === 'string') return t;
  if (t.fun) return `\( {t.fun}( \){t.args.map(termToString).join(',')})`;
  return t;
}

function isVar(t) {
  return typeof t === 'string' && /^[A-Z][a-zA-Z0-9]*$/.test(t);
}

function isCompound(t) {
  return typeof t === 'object' && t.fun && t.args;
}

function termEqual(t1, t2) {
  if (typeof t1 === 'string' && typeof t2 === 'string') return t1 === t2;
  if (isCompound(t1) && isCompound(t2)) {
    return t1.fun === t2.fun && t1.args.length === t2.args.length &&
           t1.args.every((a, i) => termEqual(a, t2.args[i]));
  }
  return false;
}

function occurs(varName, term) {
  if (typeof term === 'string') return term === varName;
  if (term.fun) return term.args.some(a => occurs(varName, a));
  return false;
}

function applySubst(subst, term) {
  if (typeof term === 'string') return subst[term] || term;
  if (term.fun) {
    return {
      fun: term.fun,
      args: term.args.map(a => applySubst(subst, a))
    };
  }
  return term;
}

// ... existing resolution, truth-table, Skolemization, Herbrand functions remain as previously implemented ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, connectivity probes, etc.) remain as previously expanded ...

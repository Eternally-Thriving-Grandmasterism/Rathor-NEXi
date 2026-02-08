// js/chat.js — Rathor Lattice Core with Full Unification Algorithm

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Full Unification
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('unify') || cmd.includes('mgu') || cmd.includes('most general unifier') ||
         cmd.includes('substitution') || cmd.includes('unification') ||
         cmd.includes('⊢') || cmd.includes('prove') || cmd.includes('theorem') || cmd.includes('resolution');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|unify|mgu|most general unifier|substitution|unification/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try unification first
  const mgu = computeMGU(cleaned);
  if (mgu) {
    response.push("\n**Most General Unifier (MGU) Found:**");
    response.push(mgu);
    response.push("\n**Mercy Conclusion:** Unification succeeded — terms are compatible under this substitution. Positive valence eternal.");
  } else {
    response.push("\n**Unification failed** — terms are not unifiable under current engine. Mercy asks: check occurs-check or variable sharing?");
  }

  // Fallback to truth-table for propositional
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
    .replace(/iff/gi, '↔');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Full Unification Algorithm (Martelli–Montanari style)
// ────────────────────────────────────────────────

function computeMGU(equationsStr) {
  // Parse equations: "f(x,y) = g(a,b), x = z" → list of pairs
  const equations = equationsStr.split(',').map(eq => eq.trim().split('='));
  if (equations.some(pair => pair.length !== 2)) return null;

  let subst = {};
  let pending = equations.map(([t1, t2]) => [parseTerm(t1.trim()), parseTerm(t2.trim())]);

  while (pending.length > 0) {
    let [t1, t2] = pending.shift();

    // Delete rule
    if (t1 === t2) continue;

    // Orient rule (put variable on left if possible)
    if (isVar(t2) && !isVar(t1)) [t1, t2] = [t2, t1];

    // Eliminate rule
    if (isVar(t1)) {
      if (occurs(t1, t2)) return null; // occurs-check
      subst[t1] = applySubst(subst, t2);
      pending = pending.map(([a,b]) => [applySubst(subst, a), applySubst(subst, b)]);
      continue;
    }

    // Decompose rule
    if (isCompound(t1) && isCompound(t2) && t1.fun === t2.fun && t1.args.length === t2.args.length) {
      for (let i = 0; i < t1.args.length; i++) {
        pending.push([t1.args[i], t2.args[i]]);
      }
      continue;
    }

    // Conflict rule
    return null;
  }

  // Format substitution nicely
  let result = Object.entries(subst)
    .map(([v, t]) => `${v} → ${termToString(t)}`)
    .join('\n');

  return result || "Empty substitution (terms already identical)";
}

// Term representation: { fun: string, args: array } or string (variable/constant)
function parseTerm(s) {
  s = s.trim();
  if (/^[A-Z]$/.test(s)) return s; // variable
  if (/^[a-z0-9]+$/.test(s)) return s; // constant

  // Function term f(t1,t2)
  const match = s.match(/^([a-z][a-z0-9]*)\((.*)\)$/);
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

  return s; // fallback
}

function termToString(t) {
  if (typeof t === 'string') return t;
  return `\( {t.fun}( \){t.args.map(termToString).join(',')})`;
}

function isVar(t) {
  return typeof t === 'string' && /^[A-Z]$/.test(t);
}

function isCompound(t) {
  return typeof t === 'object' && t.fun && t.args;
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

// ... existing truth-table functions (generateTruthTable, evaluateExpression, analyzeTruthTable) remain ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, import, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

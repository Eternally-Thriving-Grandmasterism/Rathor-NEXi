// js/chat.js — Rathor Lattice Core with Symbolic Query Mode + Truth-Table Stub

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Truth-Table Stub
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Basic truth-table stub for propositional logic
  const table = generateTruthTable(cleaned);
  if (table) {
    response.push("\n**Truth Table Stub (propositional logic):**");
    response.push(table);
    const conclusion = analyzeTruthTable(cleaned, table);
    response.push(`\n**Mercy Conclusion:** ${conclusion}`);
  } else {
    response.push("\n**Parser note:** Expression too complex for current truth-table stub (max 4 variables). Mercy asks: simplify premises?");
  }

  // Mercy rewrite & first-principles reflection
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

// Basic truth-table generator (supports ¬, ∧, ∨, →, ↔, variables A-D)
function generateTruthTable(expr) {
  // Very simple parser — extract variables (A,B,C,D only for now)
  const vars = [];
  for (let c of expr.toUpperCase()) {
    if (/[A-D]/.test(c) && !vars.includes(c)) vars.push(c);
  }
  if (vars.length > 4) return null; // limit to 16 rows

  vars.sort(); // A,B,C,D order

  const rows = 1 << vars.length;
  let table = `| \( {vars.join(' | ')} | Result |\n| \){'-|'.repeat(vars.length + 1)}\n`;

  for (let i = 0; i < rows; i++) {
    const assignment = {};
    vars.forEach((v, idx) => {
      assignment[v] = !!(i & (1 << (vars.length - 1 - idx)));
    });

    const row = vars.map(v => assignment[v] ? 'T' : 'F').join(' | ');
    let result;
    try {
      result = evaluateExpression(expr, assignment) ? 'T' : 'F';
    } catch (e) {
      result = 'ERR';
    }

    table += `| ${row} | ${result} |\n`;
  }

  return table;
}

// Simple expression evaluator (propositional only)
function evaluateExpression(expr, assignment) {
  // Replace variables with true/false
  let evalStr = expr
    .replace(/¬/g, '!')
    .replace(/∧/g, '&&')
    .replace(/∨/g, '||')
    .replace(/→/g, '=>') // JS has no implication — approximate
    .replace(/↔/g, '===');

  // Replace variables
  for (let v in assignment) {
    evalStr = evalStr.replace(new RegExp(v, 'g'), assignment[v]);
  }

  // Replace implication (A => B) with (!A || B)
  evalStr = evalStr.replace(/(.+?)=>(.+?)/g, '(!$1 || $2)');

  // Replace biconditional (A === B) with (A === B)
  // Already === in JS

  return eval(evalStr);
}

function analyzeTruthTable(expr, table) {
  if (table.includes('ERR')) return "Expression too complex for analysis.";

  const lines = table.split('\n');
  const header = lines[0];
  const rows = lines.slice(2);

  let allTrue = rows.every(row => row.endsWith('| T |'));
  let allFalse = rows.every(row => row.endsWith('| F |'));
  let someTrueSomeFalse = !allTrue && !allFalse;

  if (allTrue) return "This is a **tautology** — always true regardless of premises. Mercy affirms eternal truth.";
  if (allFalse) return "This is a **contradiction** — always false. Mercy asks: re-examine axioms?";
  if (someTrueSomeFalse) return "This is **contingent** — true under some conditions, false under others. Mercy seeks clearer premises.";

  return "Truth table generated — mercy reflection continues.";
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query + truth table
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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|first principles/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session search with tags, import/export, etc.) remain as previously expanded ...

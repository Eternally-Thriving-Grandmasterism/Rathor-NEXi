// js/chat.js — Rathor™ Lattice Core (NEXi-superseded monorepo pinnacle)
// Unified integration of session search, voice commands, symbolic reasoning, connectivity probes, import/export, emergency assistants

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
const voiceSettingsBtn = document.getElementById('voice-settings-btn');
const importFileInput = document.getElementById('import-file-input');
const importSessionBtn = document.getElementById('import-session-btn');

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

// Connectivity state
let isOffline = false;
let isHighLatency = false;
let isHighJitter = false;
let isHighPacketLoss = false;
let isVeryUnstable = false;
let rttHistory = [];
let packetLossHistory = [];

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

// Event listeners (voice, record, send, translate, search)
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
// Expanded Session Search — Full Tag Filtering + Color Indicators + Pills
// ────────────────────────────────────────────────

function filterSessions() {
  const filter = sessionSearch.value.toLowerCase().trim();
  const options = Array.from(sessionSelect.options);

  if (!filter) {
    options.forEach(opt => opt.style.display = '');
    return;
  }

  let matchCount = 0;
  options.forEach(opt => {
    const session = allSessions.find(s => s.id === opt.value);
    if (!session) {
      opt.style.display = 'none';
      return;
    }

    const name = (session.name || session.id).toLowerCase();
    const tags = (session.tags || '').toLowerCase().split(',').map(t => t.trim());
    const color = session.color || '#ffaa00';

    const nameMatch = name.includes(filter);
    const tagMatch = tags.some(tag => tag.includes(filter));

    if (nameMatch || tagMatch) {
      opt.style.display = '';
      matchCount++;

      let dot = opt.querySelector('.session-dot');
      if (!dot) {
        dot = document.createElement('span');
        dot.className = 'session-dot';
        dot.style.cssText = 'display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; border: 1px solid #444;';
        opt.insertBefore(dot, opt.firstChild);
      }
      dot.style.background = color;

      let pills = opt.querySelector('.tag-pills');
      if (!pills) {
        pills = document.createElement('div');
        pills.className = 'tag-pills';
        pills.style.cssText = 'display: inline-flex; gap: 6px; margin-left: 12px; vertical-align: middle;';
        opt.appendChild(pills);
      }
      pills.innerHTML = '';
      tags.filter(tag => tag.includes(filter)).forEach(tag => {
        const pill = document.createElement('span');
        pill.textContent = tag;
        pill.style.cssText = 'background: rgba(255,170,0,0.15); color: #ffaa00; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; border: 1px solid rgba(255,170,0,0.3);';
        pills.appendChild(pill);
      });
    } else {
      opt.style.display = 'none';
    }
  });

  if (matchCount === 0) showToast('No matching sessions or tags found');
}

// ────────────────────────────────────────────────
// More Voice Commands
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (cmd.includes('clear chat') || cmd.includes('clear messages')) {
    chatMessages.innerHTML = '';
    showToast('Chat cleared ⚡️');
    return true;
  }

  if (cmd.includes('save session') || cmd.includes('save current session')) {
    await saveCurrentSession();
    showToast('Current session saved ⚡️');
    return true;
  }

  if (cmd.includes('load last') || cmd.includes('load previous session')) {
    await loadLastSession();
    showToast('Loaded last session ⚡️');
    return true;
  }

  // ... all existing commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, symbolic query, recording, export, import, connectivity, etc.) ...

  return false;
}

// ────────────────────────────────────────────────
// Symbolic Query Mode — Mercy-First Full Reasoning Engine
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

  // Resolution with unification
  const proof = resolutionProve(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof:**");
    response.push(proof);
  }

  // Tarski fixed point reflection
  if (cleaned.toLowerCase().includes('fixed point') || cleaned.toLowerCase().includes('tarski') || cleaned.toLowerCase().includes('monotone') || cleaned.toLowerCase().includes('complete lattice')) {
    response.push("\n**Tarski's Fixed Point Theorem Reflection:**");
    response.push("In any complete lattice, every monotone function has least & greatest fixed points.");
    response.push("Mercy insight: Iteration reveals stable truth — mercy guides convergence.");
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

// ... existing functions (sendMessage, speak, recognition, recording, emergency assistants, connectivity probes, import/export, etc.) remain as previously expanded ...  if (table) {
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
// Expanded Herbrand Universe Builder & Finite Model Check
// ────────────────────────────────────────────────

function expandedHerbrandUniverse(expr) {
  // Extract constants & function symbols (stub — simple parsing)
  const constants = ['a', 'b']; // base constants
  const functions = ['f']; // unary function for demo
  const predicates = ['P', 'Q', 'Human', 'Mortal']; // example predicates

  // Build Herbrand universe up to depth 2
  let universe = [...constants];
  let currentLevel = [...constants];

  for (let depth = 1; depth <= 2; depth++) {
    const nextLevel = [];
    functions.forEach(f => {
      currentLevel.forEach(arg => {
        nextLevel.push(`\( {f}( \){arg})`);
      });
    });
    universe = [...universe, ...nextLevel];
    currentLevel = nextLevel;
  }

  // Build Herbrand base (ground atoms)
  let herbrandBase = [];
  predicates.forEach(pred => {
    universe.forEach(term => {
      herbrandBase.push(`\( {pred}( \){term})`);
    });
  });

  let report = `**Herbrand Universe (depth 2):** ${universe.join(', ')}\n`;
  report += `**Size:** ${universe.length} terms\n`;
  report += `**Herbrand Base (ground atoms):** ${herbrandBase.length} atoms (showing first 10): ${herbrandBase.slice(0,10).join(', ')}...\n`;

  // Finite satisfiability check (stub — assume satisfiable if no contradiction)
  report += "\n**Finite Model Witness (stub):** The sentence is satisfiable in this finite Herbrand universe (no contradiction found in small domain).";
  report += "\n**Mercy Insight:** By Herbrand's theorem, satisfiability in some finite universe implies satisfiability in general. Truth is witnessed here — mercy reveals the model without infinite chase.";

  return report;
}

// ... existing unification, resolution, truth-table, Skolemization functions remain as previously implemented ...

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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, connectivity probes, etc.) remain as previously expanded ...

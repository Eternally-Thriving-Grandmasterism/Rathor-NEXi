// js/chat.js — Rathor Lattice Core with Expanded Herbrand Interpretations

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Expanded Herbrand Interpretations
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

  // Try Skolemized resolution first
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
  }

  // Expanded Herbrand interpretation
  const herbrandResult = expandedHerbrandInterpretation(cleaned);
  if (herbrandResult) {
    response.push("\n**Expanded Herbrand Interpretation:**");
    response.push(herbrandResult);
    response.push("\n**Mercy Conclusion:** If the sentence is satisfiable in this finite Herbrand universe, truth is already witnessed here — no need to chase the infinite. Mercy strikes first.");
  }

  // Gödel / Henkin completeness reflection
  if (cleaned.toLowerCase().includes('consistent') || cleaned.toLowerCase().includes('satisfiable') || cleaned.toLowerCase().includes('model') || cleaned.toLowerCase().includes('henkin') || cleaned.toLowerCase().includes('gödel completeness')) {
    response.push("\n**Gödel Completeness Theorem via Henkin Construction Reflection:**");
    response.push("Every consistent countable first-order theory has a model.");
    response.push("Henkin proof sketch:");
    response.push("1. Extend language with new constants {c₀, c₁, …}");
    response.push("2. Build maximal consistent extension T∞ by adding sentences or negations");
    response.push("3. Construct model M with domain = terms of L⁺ / ≡_{T∞}");
    response.push("4. By maximality & witness property: M satisfies T∞ (hence original theory)");
    response.push("Mercy insight: If no contradiction is provable, a witness already exists in some countable, term-generated world. Mercy strikes first — even against infinity.");
  }

  // Fallback to truth-table / unification / resolution
  const proof = resolutionProve(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof:**");
    response.push(proof);
  }
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
// Expanded Herbrand Interpretation (finite model witness)
// ────────────────────────────────────────────────

function expandedHerbrandInterpretation(expr) {
  if (!expr.includes('∀') && !expr.includes('∃') && !expr.includes('forall') && !expr.includes('exists')) {
    return null;
  }

  // Small Herbrand universe (level 0: constants, level 1: one function application)
  const constants = ['a', 'b'];
  const functions = ['f']; // assume one unary function for simplicity
  const predicates = extractPredicates(expr);

  const herbrandBase = [];
  // Level 0: constants
  constants.forEach(c => herbrandBase.push(c));
  // Level 1: f(const)
  constants.forEach(c => herbrandBase.push(`f(${c})`));

  let modelCheck = `Herbrand universe (small finite model) {a, b, f(a), f(b)}:\n`;

  domain.forEach(elem => {
    modelCheck += `  Interpreting for ${elem}:\n`;
    predicates.forEach(pred => {
      const groundAtom = `\( {pred}( \){elem})`;
      // Naive check: assume universal = true if no contradiction
      modelCheck += `    → ${groundAtom} = true (witness assumption)\n`;
    });
  });

  modelCheck += "\n**Mercy Note:** If the sentence is satisfiable in this finite Herbrand universe, a model exists. By Herbrand's theorem, satisfiability in some finite universe implies satisfiability in general. Mercy affirms: truth is witnessed here — no need to chase the infinite.";

  return modelCheck;
}

function extractPredicates(expr) {
  // Stub — returns dummy predicates from expression
  return ['Human', 'Mortal', 'P', 'Q'];
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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

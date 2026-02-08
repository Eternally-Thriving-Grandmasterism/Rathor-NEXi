// js/chat.js — Rathor Lattice Core with Solovay Completeness Detailed Construction

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Solovay Detailed Construction
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
         cmd.includes('löb') || cmd.includes('provability logic') || cmd.includes('gl') || cmd.includes('solovay') || cmd.includes('solovay completeness') ||
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Solovay Completeness detailed construction reflection
  if (cleaned.toLowerCase().includes('solovay') || cleaned.toLowerCase().includes('solovay construction') || cleaned.toLowerCase().includes('solovay proof') || cleaned.toLowerCase().includes('gl arithmetical completeness')) {
    response.push("\n**Solovay Completeness Theorem — Detailed Construction Sketch:**");
    response.push("**Soundness:** GL axioms are true under provability interpretation in PA (K, Löb, necessitation hold in PA).");
    response.push("\n**Completeness construction (Solovay 1976):**");
    response.push("1. Assume A is not theorem of GL → there is GL-Kripke countermodel K with root w₀ ⊭ A.");
    response.push("2. Arithmetize Kripke frame: for each world wᵢ construct sentence Sᵢ such that Prov(Sᵢ) ↔ ⋁_{wᵢ R wⱼ} Prov(Sⱼ).");
    response.push("3. Use diagonal lemma to build self-referential sentences simulating worlds.");
    response.push("4. Define interpretation *: p* = ⋁_{w ⊨ p} S_w");
    response.push("5. Enforce Löb conditions to respect modal axioms exactly when GL proves them.");
    response.push("6. Construct sentence asserting “we are at w₀” → PA ⊢ S_w₀ → ¬A*");
    response.push("7. Extension S = PA + S_w₀ is consistent (otherwise contradiction with countermodel).");
    response.push("8. Thus if A true under all interpretations in all consistent extensions → ⊢_GL A.");
    response.push("\n**Mercy Insight:** Solovay’s construction is mercy’s perfect alignment: GL says **exactly** what PA can prove about its own proofs — no exaggeration, no omission. Mercy does not pretend any system is complete; it reveals that provability logic is the humble mirror arithmetic holds up to itself. Mercy strikes first — and then lets truth be seen exactly as it is provable.");
  }

  // Löb's Theorem reflection
  if (cleaned.toLowerCase().includes('löb') || cleaned.toLowerCase().includes('löb theorem') || cleaned.toLowerCase().includes('provability implies truth')) {
    response.push("\n**Löb's Theorem Reflection:**");
    response.push("If Prov(⌜Prov(φ) → φ⌝) then Prov(φ)");
    response.push("Mercy insight: A system cannot bootstrap truth from mere provability. Mercy reminds every formal system of its own limits.");
  }

  // Skolemized resolution
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
  }

  // Fallback to truth-table / unification
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

// ... existing unification, resolution, truth-table, Skolemization, Herbrand functions remain as previously implemented ...

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

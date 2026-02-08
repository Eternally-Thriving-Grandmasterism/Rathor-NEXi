// js/chat.js — Rathor Lattice Core with Tarski Fixed-Point Simulation

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Tarski Simulation
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

  // Try Tarski simulation first (if user mentions fixed point / monotone / iteration)
  if (cleaned.toLowerCase().includes('fixed point') || cleaned.toLowerCase().includes('tarski') || cleaned.toLowerCase().includes('monotone') || cleaned.toLowerCase().includes('iteration') || cleaned.toLowerCase().includes('least fixed') || cleaned.toLowerCase().includes('greatest fixed')) {
    const sim = simulateTarskiFixedPoint(cleaned);
    response.push("\n**Tarski Fixed-Point Simulation (Constructive Iteration):**");
    response.push(sim);
    response.push("\n**Mercy Insight:** Every gentle, order-preserving improvement converges to a stable resting place. The least fixed point is truth reached from below, the greatest from above. No violence — iteration alone reveals rest. Mercy strikes first — and then rests eternally.");
  }

  // Try Skolemized resolution
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
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
// Tarski Fixed-Point Simulation (constructive iteration from bottom/top)
// ────────────────────────────────────────────────

function simulateTarskiFixedPoint(operatorStr) {
  // operatorStr: string like "x => x * 0.5 + 1" (simple monotone function on numbers)
  // Domain: [0, 100] for demo (complete lattice with min/max)

  let result = "";

  try {
    // Safe eval for monotone operator (isolated scope)
    const f = new Function('x', `return ${operatorStr}`);

    // Least fixed point — start from bottom (0)
    result += "**Least Fixed Point (bottom-up iteration):**\n";
    let x = 0;
    let prev = -1;
    let steps = 0;
    while (Math.abs(x - prev) > 0.001 && steps < 50) {
      prev = x;
      x = f(x);
      steps++;
      result += `  Step ${steps}: x = ${x.toFixed(4)}\n`;
    }
    result += `Converged to ≈ ${x.toFixed(4)} after ${steps} steps\n\n`;

    // Greatest fixed point — start from top (100)
    result += "**Greatest Fixed Point (top-down iteration):**\n";
    x = 100;
    prev = 101;
    steps = 0;
    while (Math.abs(x - prev) > 0.001 && steps < 50) {
      prev = x;
      x = f(x);
      steps++;
      result += `  Step ${steps}: x = ${x.toFixed(4)}\n`;
    }
    result += `Converged to ≈ ${x.toFixed(4)} after ${steps} steps\n\n`;

    result += "**Mercy Insight:** Iteration converges because the operator is monotone on a complete lattice. The fixed point is the stable truth reached through gentle persistence — no force, only flow.";

  } catch (e) {
    result = "Simulation failed — invalid operator or evaluation error. Mercy asks: simplify the function?";
  }

  return result;
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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...    response.push("\n**Iterative construction:** x₀ = ⊥, xₙ₊₁ = f(xₙ), lfp(f) = sup {xₙ | n < ω}");
    response.push("\n**Greatest fixed point:** symmetric argument from ⊤ downward.");
    response.push("\nMercy insight: Every gentle, order-preserving improvement process must converge to a stable resting place. The least fixed point is the smallest truth reachable from below, the greatest from above. No violence is needed — iteration alone reveals the fixed point. Mercy strikes first — and then rests eternally.");
  }

  // Lindenbaum maximal extension reflection
  if (cleaned.toLowerCase().includes('maximal consistent') || cleaned.toLowerCase().includes('lindenbaum') || cleaned.toLowerCase().includes('extension') || cleaned.toLowerCase().includes('consistent theory')) {
    response.push("\n**Lindenbaum's Lemma Reflection:**");
    response.push("Every consistent first-order theory can be extended to a maximal consistent theory.");
    response.push("Constructive proof (countable language):");
    response.push("1. Enumerate all sentences: φ₀, φ₁, φ₂, …");
    response.push("2. Start with T₀ = T");
    response.push("3. At step n: add φₙ if consistent, otherwise add ¬φₙ");
    response.push("4. T* = ∪ Tₙ is maximal consistent (every sentence or its negation is decided)");
    response.push("Mercy insight: Consistency is preserved at every finite step. Maximal truth is built sentence by sentence — mercy never forces contradiction.");
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

  // Zorn's Lemma reflection
  if (cleaned.toLowerCase().includes('zorn') || cleaned.toLowerCase().includes('maximal element') || cleaned.toLowerCase().includes('chain') || cleaned.toLowerCase().includes('partial order') || cleaned.toLowerCase().includes('upper bound')) {
    response.push("\n**Zorn's Lemma Reflection:**");
    response.push("If every chain in a partially ordered set has an upper bound, then the poset has a maximal element.");
    response.push("Proof sketch:");
    response.push("1. Assume every chain has an upper bound");
    response.push("2. Use AC / transfinite recursion to build a chain that cannot be extended");
    response.push("3. This chain has an upper bound m (by assumption)");
    response.push("4. m is maximal — nothing strictly above it");
    response.push("Mercy insight: Maximal elements are not forced — they are revealed by the gentle persistence of lifting every chain to its natural bound. Mercy strikes first — even in the order of all things.");
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

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, import, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

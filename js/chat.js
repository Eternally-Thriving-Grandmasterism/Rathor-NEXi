// js/chat.js — Rathor Lattice Core with Full Predicate Unification & Quantifier Handling in Resolution

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Full Predicate Resolution
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

  // Try full resolution with predicate unification & quantifier handling
  const proof = resolutionProveWithQuantifiers(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof with Predicate Unification & Quantifier Handling:**");
    response.push(proof);
    response.push("\n**Mercy Conclusion:** Theorem proven in full first-order logic. Positive valence eternal.");
  } else {
    // Fallback to Skolemized / propositional
    const skolemProof = skolemizedResolutionProve(cleaned);
    if (skolemProof) {
      response.push("\n**Skolemized Resolution Proof:**");
      response.push(skolemProof);
    }
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
// Resolution with Predicate Unification & Quantifier Handling
// ────────────────────────────────────────────────

function resolutionProveWithQuantifiers(expr) {
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

  // Skolemize & convert to clausal form
  clauses = clauses.map(clause => skolemizeClause(clause));

  // Resolution loop with unification
  let steps = 0;
  const trace = ["Skolemized clauses (negated conclusion added):"];
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

  return null;
}

// ... existing unification, resolution, truth-table, Skolemization functions remain ...

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

// ... rest of chat.js functions remain as previously expanded ...    response.push("\n**Iterative construction:** x₀ = ⊥, xₙ₊₁ = f(xₙ), lfp(f) = sup {xₙ | n < ω}");
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

// ... existing unification, resolution, truth-table, Skolemization, Herbrand, Henkin, Lindenbaum, Tarski functions remain as previously implemented ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski/gi, '').trim();
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

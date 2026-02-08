// js/chat.js — Rathor Lattice Core with Tarski's Fixed Point Theorem Integration

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Tarski's Fixed Point Theorem
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
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski|fixed point|monotone|complete lattice/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try Skolemized resolution first
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
  }

  // Tarski's Fixed Point Theorem reflection
  if (cleaned.toLowerCase().includes('fixed point') || cleaned.toLowerCase().includes('tarski') || cleaned.toLowerCase().includes('monotone') || cleaned.toLowerCase().includes('complete lattice') || cleaned.toLowerCase().includes('least fixed point') || cleaned.toLowerCase().includes('greatest fixed point')) {
    response.push("\n**Tarski's Fixed Point Theorem Reflection:**");
    response.push("In any complete lattice, every monotone function has both a least fixed point and a greatest fixed point.");
    response.push("Proof sketch:");
    response.push("1. Let L be a complete lattice (every subset has sup and inf)");
    response.push("2. Let f : L → L be monotone (x ≤ y ⇒ f(x) ≤ f(y))");
    response.push("3. Least fixed point = sup {x ∈ L | x ≤ f(x)}");
    response.push("4. Greatest fixed point = inf {x ∈ L | f(x) ≤ x}");
    response.push("Mercy insight: Every gentle, order-preserving improvement process must converge to a stable truth. Iteration guided by mercy finds rest in fixed points — even across infinite chains.");
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

// ... existing unification, resolution, truth-table, Skolemization, Herbrand, Henkin, Lindenbaum functions remain as previously implemented ...

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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...  document.getElementById('pitch-value').textContent = e.target.value;
});
document.getElementById('voice-rate')?.addEventListener('input', e => {
  document.getElementById('rate-value').textContent = e.target.value;
});
document.getElementById('voice-volume')?.addEventListener('input', e => {
  document.getElementById('voice-volume-value').textContent = e.target.value;
});
document.getElementById('feedback-volume')?.addEventListener('input', e => {
  document.getElementById('feedback-volume-value').textContent = e.target.value;
});

// Test voice button glow (legacy polish)
document.getElementById('voice-test-btn')?.addEventListener('click', () => {
  document.getElementById('voice-test-btn').style.boxShadow = '0 0 20px var(--thunder-gold)';
  setTimeout(() => {
    document.getElementById('voice-test-btn').style.boxShadow = '';
  }, 1000);
  // Test TTS
  if (ttsEnabled) speak("Mercy thunder test — voice system online.");
});

// ────────────────────────────────────────────────
// Expanded Session Search — Tags + Color Indicators
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

      // Color dot indicator
      let dot = opt.querySelector('.session-dot');
      if (!dot) {
        dot = document.createElement('span');
        dot.className = 'session-dot';
        dot.style.cssText = 'display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; border: 1px solid #444;';
        opt.insertBefore(dot, opt.firstChild);
      }
      dot.style.background = color;

      // Matching tag pills
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

  if (matchCount === 0) {
    showToast('No matching sessions or tags found');
  }
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

  // ... all existing commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, symbolic query, recording, export, import, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, symbolic query with truth-table/unification/resolution, etc.) remain as previously expanded ...

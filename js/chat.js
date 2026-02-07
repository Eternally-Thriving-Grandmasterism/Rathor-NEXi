// js/chat.js — Rathor Lattice Core with Expanded Session Search, Voice Commands & Symbolic Mode

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

      // Color dot
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

  // ... existing commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, symbolic, recording, export, import, etc.) ...

  return false;
}

// ────────────────────────────────────────────────
// Symbolic Query Mode — Expanded with Predicate Logic Stub
// ────────────────────────────────────────────────

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Truth-table for propositional
  const table = generateTruthTable(cleaned);
  if (table) {
    response.push("\n**Truth Table (propositional logic):**");
    response.push(table);
    const conclusion = analyzeTruthTable(cleaned, table);
    response.push(`\n**Mercy Conclusion:** ${conclusion}`);
  }

  // Predicate logic stub (very basic)
  if (cleaned.includes('for all') || cleaned.includes('exists') || cleaned.includes('∀') || cleaned.includes('∃')) {
    response.push("\n**Predicate Logic Stub:**");
    response.push("Current engine supports basic quantifiers. Example:");
    response.push("∀x (Human(x) → Mortal(x))");
    response.push("Premise: Socrates is human → Conclusion: Socrates is mortal");
    response.push("Mercy asks: What domain are we quantifying over? What predicates are atomic?");
  }

  const mercyRewrite = cleaned
    .replace(/not/gi, '¬')
    .replace(/and/gi, '∧')
    .replace(/or/gi, '∨')
    .replace(/if/gi, '→')
    .replace(/then/gi, '')
    .replace(/implies/gi, '→')
    .replace(/iff/gi, '↔')
    .replace(/for all/gi, '∀')
    .replace(/exists/gi, '∃');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ... existing truth-table functions (generateTruthTable, evaluateExpression, analyzeTruthTable) remain ...

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

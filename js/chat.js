// js/chat.js — Rathor Lattice Core with MercyOS Symbolic Crates WASM Integration

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

// MercyOS symbolic WASM state
let mercySymbolicModule = null;

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
// Load MercyOS Symbolic WASM Module
// ────────────────────────────────────────────────

async function loadMercySymbolicWASM() {
  if (mercySymbolicModule) return mercySymbolicModule;

  try {
    const response = await fetch('/wasm/mercyos-symbolic.wasm'); // self-hosted or CDN
    const buffer = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(buffer, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256 }),
        // ... other imports if needed
      }
    });
    mercySymbolicModule = module.instance.exports;
    console.log('[Rathor] MercyOS symbolic WASM loaded');
    return mercySymbolicModule;
  } catch (e) {
    console.warn('MercyOS WASM load failed — falling back to JS truth-table', e);
    return null;
  }
}

// ────────────────────────────────────────────────
// Symbolic Query Mode — Mercy-First Truth-Seeking with Truth-Table + WASM
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

async function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try WASM first
  const wasm = await loadMercySymbolicWASM();
  let table = null;
  let conclusion = '';

  if (wasm) {
    try {
      // Assume MercyOS exports evaluate_expression(expr: &str) → String
      const ptr = wasm.alloc(cleaned.length + 1);
      const mem = new Uint8Array(wasm.memory.buffer);
      const encoder = new TextEncoder();
      encoder.encodeInto(cleaned + '\0', mem.subarray(ptr, ptr + cleaned.length + 1));
      const resultPtr = wasm.evaluate_expression(ptr);
      const resultMem = new Uint8Array(wasm.memory.buffer);
      let result = '';
      for (let i = resultPtr; resultMem[i] !== 0; i++) {
        result += String.fromCharCode(resultMem[i]);
      }
      response.push(`**MercyOS Symbolic Result:** ${result}`);
    } catch (e) {
      response.push("MercyOS WASM evaluation failed — falling back to truth-table stub");
      table = generateTruthTable(cleaned);
    }
  } else {
    table = generateTruthTable(cleaned);
  }

  if (table) {
    response.push("\n**Truth Table Stub (propositional logic):**");
    response.push(table);
    conclusion = analyzeTruthTable(cleaned, table);
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

// ... existing truth-table stub functions (generateTruthTable, evaluateExpression, analyzeTruthTable) remain as previously implemented ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();
    const answer = await symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, import, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

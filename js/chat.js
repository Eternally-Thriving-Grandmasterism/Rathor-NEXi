// js/chat.js — Rathor Lattice Core with Emergency Assistants stubs

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

// ────────────────────────────────────────────────
// Emergency Assistants Stubs (offline-first)
// ────────────────────────────────────────────────

const emergencyAssistants = {
  medical: {
    title: "Medical Guidance (Offline Stub)",
    disclaimer: "This is NOT medical advice. For emergencies call your local emergency number immediately (e.g. 112, 911). Rathor is NOT a doctor.",
    content: `Basic first-aid reminders:
• Unconscious but breathing → recovery position
• No breathing → start CPR if trained
• Severe bleeding → apply direct pressure
• Suspected heart attack → sit down, chew aspirin if available
• Always seek professional help as soon as possible.`
  },
  legal: {
    title: "Legal Rights Reminder (Offline Stub)",
    disclaimer: "This is NOT legal advice. Rathor is NOT a lawyer. Consult a qualified attorney for your situation.",
    content: `Common basic rights (varies by country):
• Right to remain silent when questioned by police
• Right to an attorney / legal counsel
• Protection against unreasonable search & seizure
• Freedom of speech & expression (with limits)
• Do NOT rely on this — laws change and context matters.`
  },
  crisis: {
    title: "Crisis Grounding (Offline Stub)",
    disclaimer: "If you are in immediate danger call emergency services NOW. This is only a temporary grounding aid.",
    content: `5-4-3-2-1 grounding technique:
5 things you can see
4 things you can touch
3 things you can hear
2 things you can smell
1 thing you can taste
Breathe slowly: in for 4, hold for 4, out for 6.
You are safe in this moment. Help is available.`
  }
};

function triggerEmergencyAssistant(mode) {
  const assistant = emergencyAssistants[mode];
  if (!assistant) return;

  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content emergency-modal">
      <h2 style="color: #ff4444;">${assistant.title}</h2>
      <p style="color: #ff6666; font-weight: bold;">${assistant.disclaimer}</p>
      <p style="white-space: pre-wrap;">${assistant.content}</p>
      <div class="modal-buttons">
        <button onclick="this.closest('.modal-overlay').remove()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with emergency
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // Emergency assistants
  if (cmd.includes('medical help') || cmd.includes('medical advice') || cmd.includes('health emergency')) {
    triggerEmergencyAssistant('medical');
    return true;
  }

  if (cmd.includes('legal advice') || cmd.includes('legal help') || cmd.includes('rights')) {
    triggerEmergencyAssistant('legal');
    return true;
  }

  if (cmd.includes('crisis mode') || cmd.includes('emotional support') || cmd.includes('grounding')) {
    triggerEmergencyAssistant('crisis');
    return true;
  }

  // Existing commands (emergency recording, export, test, bridges, etc.)
  if (cmd.includes('emergency mode') || cmd.includes('crisis recording')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency recording started — saved with priority ⚠️');
    return true;
  }

  if (cmd.includes('stop emergency') || cmd.includes('end recording')) {
    stopVoiceRecording();
    showToast('Recording stopped & saved ⚡️');
    return true;
  }

  // ... other existing commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session CRUD, etc.) remain unchanged ...  modal.innerHTML = `
    <div class="modal-content emergency-modal">
      <h2 style="color: #ff4444;">${assistant.title}</h2>
      <p style="color: #ff6666; font-weight: bold;">${assistant.disclaimer}</p>
      <p style="white-space: pre-wrap;">${assistant.content}</p>
      <div class="modal-buttons">
        <button onclick="this.closest('.modal-overlay').remove()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with emergency
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // Emergency assistants
  if (cmd.includes('medical help') || cmd.includes('medical advice') || cmd.includes('health emergency')) {
    triggerEmergencyAssistant('medical');
    return true;
  }

  if (cmd.includes('legal advice') || cmd.includes('legal help') || cmd.includes('rights')) {
    triggerEmergencyAssistant('legal');
    return true;
  }

  if (cmd.includes('crisis mode') || cmd.includes('emotional support') || cmd.includes('grounding')) {
    triggerEmergencyAssistant('crisis');
    return true;
  }

  // Existing commands (emergency recording, export, test, bridges, etc.)
  if (cmd.includes('emergency mode') || cmd.includes('crisis recording')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency recording started — saved with priority ⚠️');
    return true;
  }

  if (cmd.includes('stop emergency') || cmd.includes('end recording')) {
    stopVoiceRecording();
    showToast('Recording stopped & saved ⚡️');
    return true;
  }

  // ... other existing commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session CRUD, etc.) remain unchanged ...      let indicator = opt.querySelector('.session-indicator');
      if (!indicator) {
        indicator = document.createElement('span');
        indicator.className = 'session-indicator';
        indicator.style.cssText = 'display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle;';
        opt.insertBefore(indicator, opt.firstChild);
      }
      indicator.style.background = session.color || '#ffaa00';
    }
  });
}

// ... (rest of functions: voice recognition, recording, processVoiceCommand, refreshSessionList, loadChatHistory, updateTranslationStats, bridge pings, etc. remain as previously deployed)

// Example session object shape (for reference)
async function refreshSessionList() {
  allSessions = await rathorDB.getAllSessions();
  sessionSelect.innerHTML = '';
  allSessions.forEach(session => {
    const option = document.createElement('option');
    option.value = session.id;
    option.textContent = session.name || session.id;
    if (session.id === currentSessionId) option.selected = true;
    sessionSelect.appendChild(option);
  });
}

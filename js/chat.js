// js/chat.js — Rathor Lattice Core with Modal Close Protection for Crisis Modes

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
// Modal Close Protection for Crisis Modes
// ────────────────────────────────────────────────

const PROTECTED_MODES = ['crisis', 'mental', 'ptsd', 'cptsd', 'ifs'];

function protectCrisisModalClose(modal, mode) {
  if (!PROTECTED_MODES.includes(mode)) return;

  const originalRemove = modal.remove;
  modal.remove = function() {
    if (confirm("Parts may still need attention. Close anyway?")) {
      originalRemove.call(modal);
    }
    // else do nothing — modal stays open
  };

  // Also protect close via backdrop click
  modal.addEventListener('click', e => {
    if (e.target === modal) {
      e.stopPropagation();
      if (confirm("Parts may still need attention. Close anyway?")) {
        originalRemove.call(modal);
      }
    }
  });
}

// Update trigger function to add protection
function triggerEmergencyAssistant(mode) {
  const assistant = emergencyAssistants[mode];
  if (!assistant) return;

  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content emergency-modal">
      <h2 style="color: #ff4444;">${assistant.title}</h2>
      <p style="color: #ff6666; font-weight: bold; margin-bottom: 1em;">${assistant.disclaimer}</p>
      <div style="max-height: 60vh; overflow-y: auto; padding-right: 12px;">
        ${assistant.templates.map(t => `
          <h3 style="margin: 1.5em 0 0.5em; color: #ffaa00;">${t.name}</h3>
          <p style="white-space: pre-wrap; line-height: 1.6;">${t.content}</p>
        `).join('')}
      </div>
      <div class="modal-buttons" style="margin-top: 1.5em;">
        <button onclick="this.closest('.modal-overlay').remove()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';

  // Apply protection
  protectCrisisModalClose(modal, mode);
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session search with tags, import/export, etc.) remain as previously expanded ...

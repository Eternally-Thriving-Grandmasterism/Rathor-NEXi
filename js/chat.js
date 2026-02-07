// js/chat.js — Rathor Lattice Core with Full Session Search + Tags & Colors

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
sessionSearch.addEventListener('input', filterSessions);

// ────────────────────────────────────────────────
// Session Search — Restored with Tags + Color Indicators
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

      // Visual indicators (color dot + tag pills)
      let indicator = opt.querySelector('.session-indicator');
      if (!indicator) {
        indicator = document.createElement('span');
        indicator.className = 'session-indicator';
        indicator.style.cssText = 'display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; border: 1px solid #444;';
        opt.insertBefore(indicator, opt.firstChild);
      }
      indicator.style.background = color;

      // Tag pills (show matching tags)
      let pillContainer = opt.querySelector('.tag-pills');
      if (!pillContainer) {
        pillContainer = document.createElement('div');
        pillContainer.className = 'tag-pills';
        pillContainer.style.cssText = 'display: inline-flex; gap: 6px; margin-left: 12px; vertical-align: middle;';
        opt.appendChild(pillContainer);
      }
      pillContainer.innerHTML = '';
      tags.filter(tag => tag.includes(filter)).forEach(tag => {
        const pill = document.createElement('span');
        pill.textContent = tag;
        pill.style.cssText = 'background: rgba(255,170,0,0.15); color: #ffaa00; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; border: 1px solid rgba(255,170,0,0.3);';
        pillContainer.appendChild(pill);
      });
    } else {
      opt.style.display = 'none';
    }
  });

  if (matchCount === 0) {
    showToast('No matching sessions or tags found');
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, etc.) remain as previously expanded ...

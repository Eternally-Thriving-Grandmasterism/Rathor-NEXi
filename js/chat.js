// js/chat.js — Rathor Lattice Core with Full Tag Autocomplete

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
let tagFrequency = new Map(); // global tag frequency for autocomplete
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
await updateTagFrequency(); // initial load

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
// Tag Frequency & Autocomplete
// ────────────────────────────────────────────────

async function updateTagFrequency() {
  tagFrequency.clear();
  const sessions = await rathorDB.getAllSessions();
  sessions.forEach(session => {
    if (session.tags) {
      session.tags.split(',').map(t => t.trim()).filter(t => t).forEach(tag => {
        tagFrequency.set(tag, (tagFrequency.get(tag) || 0) + 1);
      });
    }
  });
}

// Autocomplete dropdown for tags in edit modal
const editTagsInput = document.getElementById('edit-tags');
const editTagPreview = document.getElementById('edit-tag-preview');
const editTagSuggestions = document.createElement('div');
editTagSuggestions.id = 'edit-tag-suggestions';
editTagSuggestions.style.cssText = 'position: absolute; background: #110022; border: 1px solid var(--thunder-gold); border-radius: 8px; max-height: 200px; overflow-y: auto; z-index: 10; display: none; width: 100%;';
editTagsInput.parentNode.appendChild(editTagSuggestions);

editTagsInput.addEventListener('input', e => {
  const value = e.target.value.trim();
  const lastTag = value.split(',').pop().trim().toLowerCase();

  if (!lastTag) {
    editTagSuggestions.style.display = 'none';
    renderTagPills(value);
    return;
  }

  const suggestions = Array.from(tagFrequency.keys())
    .filter(tag => tag.toLowerCase().includes(lastTag))
    .sort((a, b) => tagFrequency.get(b) - tagFrequency.get(a)) // most used first
    .slice(0, 8);

  editTagSuggestions.innerHTML = '';
  if (suggestions.length === 0) {
    editTagSuggestions.style.display = 'none';
  } else {
    suggestions.forEach(tag => {
      const div = document.createElement('div');
      div.textContent = tag;
      div.style.cssText = 'padding: 8px 12px; cursor: pointer;';
      div.onmouseover = () => div.style.background = 'rgba(255,170,0,0.15)';
      div.onmouseout = () => div.style.background = '';
      div.onclick = () => {
        const parts = value.split(',');
        parts.pop();
        parts.push(' ' + tag);
        editTagsInput.value = parts.join(',');
        renderTagPills(editTagsInput.value);
        editTagSuggestions.style.display = 'none';
        editTagsInput.focus();
      };
      editTagSuggestions.appendChild(div);
    });
    editTagSuggestions.style.display = 'block';
  }
  renderTagPills(value);
});

editTagsInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && editTagSuggestions.style.display !== 'none') {
    e.preventDefault();
    const first = editTagSuggestions.querySelector('div');
    if (first) first.click();
  }
});

function renderTagPills(tagsString) {
  editTagPreview.innerHTML = '';
  if (!tagsString) return;
  const tags = tagsString.split(',').map(t => t.trim()).filter(t => t);
  tags.forEach(tag => {
    const pill = document.createElement('span');
    pill.textContent = tag;
    pill.style.cssText = 'background: rgba(255,170,0,0.2); color: #ffaa00; padding: 4px 10px; border-radius: 12px; font-size: 0.9em; margin: 4px; display: inline-flex; align-items: center; gap: 6px;';
    const remove = document.createElement('span');
    remove.textContent = '×';
    remove.style.cssText = 'cursor: pointer; font-weight: bold;';
    remove.onclick = () => {
      const newTags = tags.filter(t => t !== tag).join(', ');
      editTagsInput.value = newTags;
      renderTagPills(newTags);
    };
    pill.appendChild(remove);
    editTagPreview.appendChild(pill);
  });
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, etc.) remain as previously expanded ...

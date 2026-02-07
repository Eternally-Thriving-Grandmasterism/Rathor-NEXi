// js/chat.js — Rathor Lattice Core with Full Tag Management UI

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
// Tag Management UI — Edit Modal Integration
// ────────────────────────────────────────────────

const editTagsInput = document.getElementById('edit-tags');
const editTagPreview = document.getElementById('edit-tag-preview');
const editColorInput = document.getElementById('edit-color');

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

editTagsInput.addEventListener('input', e => renderTagPills(e.target.value));

// Edit modal save logic (add to existing modal-save listener or replace)
document.getElementById('modal-save')?.addEventListener('click', async () => {
  const name = document.getElementById('edit-name').value.trim();
  const description = document.getElementById('edit-description').value.trim();
  const tags = document.getElementById('edit-tags').value.trim();
  const color = document.getElementById('edit-color').value;

  const session = await getSession(currentSessionId);
  if (session) {
    session.name = name || session.name;
    session.description = description || session.description;
    session.tags = tags;
    session.color = color;
    await saveSession(session);
    await refreshSessionList();
    showToast('Session updated ⚡️');
  }

  document.getElementById('edit-modal-overlay').style.display = 'none';
});

// ────────────────────────────────────────────────
// Session Search — Expanded with Tag Filtering & Color Indicators
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

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, etc.) remain unchanged ...

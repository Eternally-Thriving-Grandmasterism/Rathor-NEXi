// js/chat.js — Rathor Lattice Core with Tag Length Limits & Strict Validation

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
// Tag Validation, Length Limits & Error Messaging
// ────────────────────────────────────────────────

const TAG_MAX_LENGTH = 32;
const TAG_MIN_LENGTH = 2;
const MAX_TAGS_PER_SESSION = 15;
const TAG_ALLOWED_CHARS = /^[a-z0-9\-_ ]+$/i;

function normalizeAndValidateTags(tagsString) {
  if (!tagsString) return { cleaned: '', errors: [] };

  const rawTags = tagsString.split(',').map(t => t.trim());
  const errors = [];
  const seen = new Set();
  const validTags = [];

  rawTags.forEach(tag => {
    if (!tag) {
      errors.push('Empty tags ignored');
      return;
    }

    if (tag.length < TAG_MIN_LENGTH) {
      errors.push(`Tag "${tag}" too short (min ${TAG_MIN_LENGTH} chars)`);
      return;
    }

    if (tag.length > TAG_MAX_LENGTH) {
      errors.push(`Tag "${tag}" too long (max ${TAG_MAX_LENGTH} chars)`);
      return;
    }

    if (!TAG_ALLOWED_CHARS.test(tag)) {
      errors.push(`Tag "${tag}" contains invalid characters — only letters, numbers, hyphen, underscore, space allowed`);
      return;
    }

    const canonical = tag.toLowerCase();
    if (seen.has(canonical)) {
      errors.push(`Duplicate tag "${tag}" removed`);
      return;
    }

    seen.add(canonical);
    validTags.push(tag);
  });

  if (validTags.length > MAX_TAGS_PER_SESSION) {
    errors.push(`Too many tags (\( {validTags.length}/ \){MAX_TAGS_PER_SESSION}). Using first ${MAX_TAGS_PER_SESSION}`);
    validTags.length = MAX_TAGS_PER_SESSION;
  }

  const cleaned = validTags.join(', ');
  return { cleaned, errors };
}

function renderTagPills(tagsString) {
  editTagPreview.innerHTML = '';
  const { cleaned } = normalizeAndValidateTags(tagsString);
  if (!cleaned) return;
  const tags = cleaned.split(',').map(t => t.trim());
  tags.forEach(tag => {
    const pill = document.createElement('span');
    pill.textContent = tag;
    pill.style.cssText = 'background: rgba(255,170,0,0.2); color: #ffaa00; padding: 4px 10px; border-radius: 12px; font-size: 0.9em; margin: 4px; display: inline-flex; align-items: center; gap: 6px;';
    const remove = document.createElement('span');
    remove.textContent = '×';
    remove.style.cssText = 'cursor: pointer; font-weight: bold;';
    remove.onclick = () => {
      const remaining = tags.filter(t => t !== tag).join(', ');
      editTagsInput.value = remaining;
      renderTagPills(remaining);
    };
    pill.appendChild(remove);
    editTagPreview.appendChild(pill);
  });
  // Sync input to cleaned version
  editTagsInput.value = cleaned;
}

editTagsInput.addEventListener('input', e => {
  const value = e.target.value;
  const { cleaned, errors } = normalizeAndValidateTags(value);
  renderTagPills(value);

  if (errors.length > 0) {
    showToast(errors.join(' • '), 'warning');
  }
});

// On modal save — validate & clean tags
document.getElementById('modal-save')?.addEventListener('click', async () => {
  const name = document.getElementById('edit-name').value.trim();
  const description = document.getElementById('edit-description').value.trim();
  const rawTags = document.getElementById('edit-tags').value.trim();
  const { cleaned: tags, errors } = normalizeAndValidateTags(rawTags);
  const color = document.getElementById('edit-color').value;

  if (errors.length > 0) {
    showToast('Tags cleaned automatically: ' + errors.join(' • '), 'warning');
  }

  const session = await getSession(currentSessionId);
  if (session) {
    session.name = name || session.name;
    session.description = description || session.description;
    session.tags = tags;
    session.color = color;
    await saveSession(session);
    await refreshSessionList();
    await updateTagFrequency(); // refresh global frequency
    showToast('Session updated — tags validated & saved ⚡️');
  }

  document.getElementById('edit-modal-overlay').style.display = 'none';
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...  const tags = cleaned.split(',').map(t => t.trim());
  tags.forEach(tag => {
    const pill = document.createElement('span');
    pill.textContent = tag;
    pill.style.cssText = 'background: rgba(255,170,0,0.2); color: #ffaa00; padding: 4px 10px; border-radius: 12px; font-size: 0.9em; margin: 4px; display: inline-flex; align-items: center; gap: 6px;';
    const remove = document.createElement('span');
    remove.textContent = '×';
    remove.style.cssText = 'cursor: pointer; font-weight: bold;';
    remove.onclick = () => {
      const remaining = tags.filter(t => t !== tag).join(', ');
      editTagsInput.value = remaining;
      renderTagPills(remaining);
    };
    pill.appendChild(remove);
    editTagPreview.appendChild(pill);
  });
  // Sync input to cleaned version
  editTagsInput.value = cleaned;
}

editTagsInput.addEventListener('input', e => {
  const value = e.target.value;
  const { cleaned, errors } = normalizeAndValidateTags(value);
  renderTagPills(value);

  // Show errors as toast (one at a time or combined)
  if (errors.length > 0) {
    showToast(errors.join(' • '));
  }

  // Autocomplete code remains as before, but suggestions normalized
});

// On modal save — validate & clean tags
document.getElementById('modal-save')?.addEventListener('click', async () => {
  const name = document.getElementById('edit-name').value.trim();
  const description = document.getElementById('edit-description').value.trim();
  const rawTags = document.getElementById('edit-tags').value.trim();
  const { cleaned: tags, errors } = normalizeAndValidateTags(rawTags);
  const color = document.getElementById('edit-color').value;

  if (errors.length > 0) {
    showToast('Tags cleaned automatically: ' + errors.join(' • '));
  }

  const session = await getSession(currentSessionId);
  if (session) {
    session.name = name || session.name;
    session.description = description || session.description;
    session.tags = tags;
    session.color = color;
    await saveSession(session);
    await refreshSessionList();
    await updateTagFrequency(); // refresh global frequency
    showToast('Session updated — tags validated & saved ⚡️');
  }

  document.getElementById('edit-modal-overlay').style.display = 'none';
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...  // Autocomplete code remains as before, but suggestions will be normalized too
});

// On modal save — ensure tags are normalized
document.getElementById('modal-save')?.addEventListener('click', async () => {
  const name = document.getElementById('edit-name').value.trim();
  const description = document.getElementById('edit-description').value.trim();
  const rawTags = document.getElementById('edit-tags').value.trim();
  const tags = normalizeTags(rawTags); // clean here too
  const color = document.getElementById('edit-color').value;

  const session = await getSession(currentSessionId);
  if (session) {
    session.name = name || session.name;
    session.description = description || session.description;
    session.tags = tags;
    session.color = color;
    await saveSession(session);
    await refreshSessionList();
    await updateTagFrequency(); // refresh global frequency
    showToast('Session updated — tags cleaned & saved ⚡️');
  }

  document.getElementById('edit-modal-overlay').style.display = 'none';
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...

// js/chat.js — Rathor Lattice Core with Tag-based Session Export

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

// New export button & modal refs (add to HTML if not present)
const exportByTagBtn = document.getElementById('export-by-tag-btn') || document.createElement('button'); // fallback create

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

// ... existing event listeners (voice, record, send, translate, search) ...

// ────────────────────────────────────────────────
// Tag-based Session Export
// ────────────────────────────────────────────────

exportByTagBtn.textContent = 'Export by Tag';
exportByTagBtn.style.cssText = 'background: var(--thunder-gold); color: #000; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin: 10px 0;';
exportByTagBtn.onclick = showTagExportModal;
document.getElementById('session-controls')?.appendChild(exportByTagBtn); // append to controls

function showTagExportModal() {
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content">
      <h2>Export Sessions by Tag</h2>
      <label for="export-tag-select">Select Tag:</label>
      <select id="export-tag-select">
        <option value="">All Sessions</option>
        \( {Array.from(tagFrequency.keys()).map(tag => `<option value=" \){tag}">\( {tag} ( \){tagFrequency.get(tag)})</option>`).join('')}
      </select>
      <div style="margin-top: 1em;">
        <button id="export-confirm">Export</button>
        <button id="export-cancel">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';

  document.getElementById('export-confirm').onclick = async () => {
    const selectedTag = document.getElementById('export-tag-select').value;
    await exportSessionsByTag(selectedTag);
    modal.remove();
  };

  document.getElementById('export-cancel').onclick = () => modal.remove();
}

async function exportSessionsByTag(tag = '') {
  let sessionsToExport = allSessions;
  if (tag) {
    sessionsToExport = allSessions.filter(s => (s.tags || '').split(',').map(t => t.trim()).includes(tag));
  }

  if (sessionsToExport.length === 0) {
    showToast('No sessions match this tag');
    return;
  }

  const exportData = await Promise.all(sessionsToExport.map(async session => {
    const messages = await rathorDB.getMessages(session.id);
    return {
      id: session.id,
      name: session.name || session.id,
      description: session.description || '',
      tags: session.tags || '',
      color: session.color || '#ffaa00',
      messages
    };
  }));

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = tag ? `rathor-sessions-tag-\( {tag}- \){new Date().toISOString().split('T')[0]}.json` : `rathor-all-sessions-${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast(`Exported \( {sessionsToExport.length} session \){sessionsToExport.length !== 1 ? 's' : ''} ⚡️`);
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, etc.) remain as previously expanded ...

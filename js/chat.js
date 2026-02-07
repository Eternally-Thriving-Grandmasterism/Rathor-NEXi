// js/chat.js — Rathor Lattice Core with Starlink Connectivity Detection

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
let isOffline = false;
let isHighLatency = false;

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
// Starlink / Connectivity Awareness
// ────────────────────────────────────────────────

function updateConnectivityStatus() {
  if (!navigator.connection) return;

  const conn = navigator.connection;
  isOffline = conn.type === 'none' || conn.rtt > 10000 || conn.downlinkMax < 1;
  isHighLatency = conn.rtt > 150 || conn.downlinkMax < 10;

  if (isOffline) {
    showToast('Offline mode — voice notes & messages queued ⚡️');
  } else if (isHighLatency) {
    showToast('High latency (Starlink mode?) — compressing & batching ⚡️');
  } else {
    showToast('Good connection — lattice fully online ⚡️');
  }
}

// Listen for connectivity changes
if (navigator.connection) {
  navigator.connection.addEventListener('change', updateConnectivityStatus);
}

// Initial check
updateConnectivityStatus();

// Queue voice note if offline/high-latency
async function startVoiceRecording(sessionId, isEmergency = false) {
  // ... existing recording logic ...
  if (isOffline || isHighLatency) {
    await rathorDB.saveQueuedAction('voice-note', { sessionId, blob, timestamp, isEmergency });
    showToast('Voice note queued for Starlink reconnection ⚡️');
  } else {
    await rathorDB.saveVoiceNote(sessionId, blob, timestamp, isEmergency);
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...      if (Array.isArray(importedSession.messages)) {
        await saveMessages(finalId, importedSession.messages.map(m => ({
          ...m,
          sessionId: finalId,
          timestamp: m.timestamp || Date.now()
        })));
      }

      imported++;
    }

    await refreshSessionList();
    await updateTagFrequency();

    let msg = `Imported \( {imported} session \){imported !== 1 ? 's' : ''} successfully ⚡️`;
    if (warnings.length > 0) msg += `\nWarnings: ${warnings.join('; ')}`;
    showToast(msg);

  } catch (err) {
    showToast('Import failed: ' + err.message, 'error');
    console.error(err);
  }

  importFileInput.value = ''; // reset input
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...      if (Array.isArray(importedSession.messages)) {
        await saveMessages(finalId, importedSession.messages.map(m => ({
          ...m,
          sessionId: finalId,
          timestamp: m.timestamp || Date.now()
        })));
      }

      imported++;
    }

    await refreshSessionList();
    await updateTagFrequency();

    let msg = `Imported \( {imported} session \){imported !== 1 ? 's' : ''} successfully ⚡️`;
    if (warnings.length > 0) msg += `\nWarnings: ${warnings.join('; ')}`;
    showToast(msg);

  } catch (err) {
    showToast('Import failed: ' + err.message, 'error');
    console.error(err);
  }

  importFileInput.value = ''; // reset input
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...        tags: normalizeTags(importedSession.tags || ''),
        color: importedSession.color || '#ffaa00',
        createdAt: importedSession.createdAt || Date.now()
      };

      await saveSession(session);

      // Import messages
      if (Array.isArray(importedSession.messages)) {
        await saveMessages(finalId, importedSession.messages.map(m => ({
          ...m,
          sessionId: finalId,
          timestamp: m.timestamp || Date.now()
        })));
      }

      imported++;
    }

    await refreshSessionList();
    await updateTagFrequency();

    let msg = `Imported \( {imported} session \){imported !== 1 ? 's' : ''} successfully ⚡️`;
    if (warnings.length > 0) msg += `\nWarnings: ${warnings.join('; ')}`;
    showToast(msg);

  } catch (err) {
    showToast('Import failed: ' + err.message, 'error');
    console.error(err);
  }

  importFileInput.value = ''; // reset input
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, etc.) remain as previously expanded ...    return {
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

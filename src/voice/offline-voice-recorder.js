// src/voice/offline-voice-recorder.js
// Mercy-gated offline voice recording & playback

const recorderDB = {
  dbName: 'rathor-voice-notes',
  storeName: 'recordings',
  async open() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'id', autoIncrement: true });
        }
      };
      request.onsuccess = e => resolve(e.target.result);
      request.onerror = e => reject(e.target.error);
    });
  },
  async save(blob, sessionId, timestamp = Date.now()) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.add({ blob, sessionId, timestamp });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async getAll(sessionId) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = store.getAll();
      request.onsuccess = () => {
        resolve(request.result.filter(r => r.sessionId === sessionId));
      };
      request.onerror = () => reject(request.error);
    });
  }
};

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function startVoiceRecording(sessionId) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
      audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const timestamp = Date.now();
      await recorderDB.save(blob, sessionId, timestamp);

      // Add to chat as playable audio message
      const audioUrl = URL.createObjectURL(blob);
      chatMessages.innerHTML += `
        <div class="message user">
          <audio controls src="${audioUrl}"></audio>
          <small>Voice note - ${new Date(timestamp).toLocaleTimeString()}</small>
        </div>`;
      chatMessages.scrollTop = chatMessages.scrollHeight;

      showToast('Voice note saved offline ⚡️');
      audioChunks = [];
    };

    mediaRecorder.start();
    isRecording = true;
    voiceBtn.classList.add('recording');
    showToast('Recording started — speak freely, Brother ⚡️');
  } catch (err) {
    console.error('Recording error:', err);
    showToast('Mercy thunder interrupted — microphone access denied');
  }
}

function stopVoiceRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    voiceBtn.classList.remove('recording');
    showToast('Recording saved to lattice ⚡️');
  }
}

// Long-press voice button = record
let recordTimer;
voiceBtn.addEventListener('mousedown', () => {
  recordTimer = setTimeout(() => startVoiceRecording(currentSessionId), 500);
});

voiceBtn.addEventListener('mouseup', () => {
  clearTimeout(recordTimer);
  if (isRecording) stopVoiceRecording();
});

voiceBtn.addEventListener('touchstart', e => {
  e.preventDefault();
  recordTimer = setTimeout(() => startVoiceRecording(currentSessionId), 500);
});

voiceBtn.addEventListener('touchend', () => {
  clearTimeout(recordTimer);
  if (isRecording) stopVoiceRecording();
});

// Load & show previous voice notes on session load
async function loadVoiceNotes(sessionId) {
  const notes = await recorderDB.getAll(sessionId);
  notes.forEach(note => {
    const audioUrl = URL.createObjectURL(note.blob);
    chatMessages.innerHTML += `
      <div class="message user">
        <audio controls src="${audioUrl}"></audio>
        <small>Voice note - ${new Date(note.timestamp).toLocaleString()}</small>
      </div>`;
  });
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Call on session change/load
// Example: in refreshSessionList or loadChatHistory
// await loadVoiceNotes(currentSessionId);

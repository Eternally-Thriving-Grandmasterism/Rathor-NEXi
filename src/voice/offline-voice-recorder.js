// src/voice/offline-voice-recorder.js
// Mercy-gated offline voice recording, silence detection, export & emergency mode

const recorderDB = {
  dbName: 'rathor-voice-notes',
  storeName: 'recordings',
  async open() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 2); // bump version for new indexes
      request.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id', autoIncrement: true });
          store.createIndex('sessionId', 'sessionId', { unique: false });
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
      request.onsuccess = e => resolve(e.target.result);
      request.onerror = e => reject(e.target.error);
    });
  },
  async save(blob, sessionId, timestamp = Date.now(), isEmergency = false) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.add({ blob, sessionId, timestamp, isEmergency });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async getAll(sessionId) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = store.index('sessionId').getAll(sessionId);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async exportAll(sessionId) {
    const notes = await this.getAll(sessionId);
    notes.forEach((note, index) => {
      const url = URL.createObjectURL(note.blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `rathor-voice-\( {sessionId}- \){note.timestamp}${note.isEmergency ? '-EMERGENCY' : ''}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    showToast(`Exported \( {notes.length} voice note \){notes.length !== 1 ? 's' : ''} ⚡️`);
  }
};

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let silenceTimer = null;
let lastSoundTime = Date.now();
const SILENCE_THRESHOLD = -45; // dB
const SILENCE_TIMEOUT = 3500;  // ms

async function startVoiceRecording(sessionId, isEmergency = false) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks = [];
    lastSoundTime = Date.now();

    // Silence detection via audio analyser
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function detectSilence() {
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const avg = sum / bufferLength;
      const db = 20 * Math.log10(avg / 255) || -100;

      if (db > SILENCE_THRESHOLD) {
        lastSoundTime = Date.now();
      }

      if (Date.now() - lastSoundTime > SILENCE_TIMEOUT && isRecording) {
        stopVoiceRecording();
        showToast('Silence detected — recording stopped ⚡️');
      } else {
        silenceTimer = requestAnimationFrame(detectSilence);
      }
    }

    detectSilence();

    mediaRecorder.ondataavailable = e => {
      audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      cancelAnimationFrame(silenceTimer);
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const timestamp = Date.now();
      await recorderDB.save(blob, sessionId, timestamp, isEmergency);

      const audioUrl = URL.createObjectURL(blob);
      chatMessages.innerHTML += `
        <div class="message user ${isEmergency ? 'emergency' : ''}">
          <audio controls src="${audioUrl}"></audio>
          <small>Voice note ${isEmergency ? '(EMERGENCY)' : ''} — ${new Date(timestamp).toLocaleTimeString()}</small>
        </div>`;
      chatMessages.scrollTop = chatMessages.scrollHeight;

      showToast(isEmergency ? 'Emergency voice note saved offline ⚠️' : 'Voice note saved offline ⚡️');
      audioChunks = [];
    };

    mediaRecorder.start();
    isRecording = true;
    voiceBtn.classList.add('recording');
    if (isEmergency) voiceBtn.classList.add('emergency-recording');
    showToast(isEmergency ? 'Emergency recording started — speak now ⚠️' : 'Recording started — speak freely ⚡️');
  } catch (err) {
    console.error('Recording error:', err);
    showToast('Mercy thunder interrupted — microphone access denied');
  }
}

function stopVoiceRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    voiceBtn.classList.remove('recording', 'emergency-recording');
    showToast('Recording saved to lattice ⚡️');
  }
}

// ────────────────────────────────────────────────
// Emergency Voice Commands Integration
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // Emergency triggers
  if (cmd.includes('emergency mode') || cmd.includes('crisis mode') || cmd.includes('help now')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency mode activated — recording & saving critical ⚠️');
    // Future: auto-queue for Starlink / high-priority send when online
    return true;
  }

  if (cmd.includes('stop emergency') || cmd.includes('end crisis')) {
    stopVoiceRecording();
    showToast('Emergency recording stopped — saved safely ⚡️');
    return true;
  }

  // Export all voice notes via voice
  if (cmd.includes('export voice notes') || cmd.includes('download recordings')) {
    await recorderDB.exportAll(currentSessionId);
    return true;
  }

  // ... keep all previous commands ...
}

// ────────────────────────────────────────────────
// UI Integration (add to index.html)
// ────────────────────────────────────────────────

// Add this button near voice-btn in chat-input-area
/*
<button id="record-btn" title="Long-press or click to record voice note">🎙️</button>
*/

// Bindings (add to main script)
const recordBtn = document.getElementById('record-btn');
if (recordBtn) {
  let recordTimer;
  recordBtn.addEventListener('mousedown', () => {
    recordTimer = setTimeout(() => startVoiceRecording(currentSessionId), 400);
  });
  recordBtn.addEventListener('mouseup', () => {
    clearTimeout(recordTimer);
    if (isRecording) stopVoiceRecording();
  });
  // Touch support
  recordBtn.addEventListener('touchstart', e => {
    e.preventDefault();
    recordTimer = setTimeout(() => startVoiceRecording(currentSessionId), 400);
  });
  recordBtn.addEventListener('touchend', () => {
    clearTimeout(recordTimer);
    if (isRecording) stopVoiceRecording();
  });
}

// Load voice notes when session changes
async function onSessionLoaded(sessionId) {
  currentSessionId = sessionId;
  await loadVoiceNotes(sessionId);
}

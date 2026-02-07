// js/chat.js — Rathor Lattice Core with Periodic RTT Probes + Jitter Calculation

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

// Connectivity state
let isOffline = false;
let isHighLatency = false;
let isHighJitter = false;
let rttHistory = []; // last 5 successful RTTs (ms)
const RTT_PROBE_INTERVAL = 10000; // 10 seconds
const HIGH_LATENCY_RTT_THRESHOLD = 150; // ms
const HIGH_LATENCY_DOWNLINK_THRESHOLD = 10; // Mbps
const HIGH_JITTER_THRESHOLD = 50; // ms (standard deviation)
const HIGH_VARIANCE_THRESHOLD = 40; // ms (avg abs diff between consecutive)

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

// ... existing event listeners (voice, record, send, translate, search) ...

// ────────────────────────────────────────────────
// Periodic RTT Probes with Jitter Calculation
// ────────────────────────────────────────────────

async function probeRTT() {
  if (document.hidden) return; // don't probe when tab hidden

  const start = performance.now();
  try {
    await fetch('/ping?t=' + Date.now(), { cache: 'no-store', mode: 'no-cors' });
    const end = performance.now();
    const rtt = end - start;

    rttHistory.push(rtt);
    if (rttHistory.length > 5) rttHistory.shift();

    // Median RTT (for latency)
    const sorted = [...rttHistory].sort((a,b)=>a-b);
    const median = sorted[Math.floor(sorted.length/2)];

    // Jitter = standard deviation
    const mean = rttHistory.reduce((a,b)=>a+b,0) / rttHistory.length;
    const variance = rttHistory.reduce((a,b)=>a + Math.pow(b-mean,2),0) / rttHistory.length;
    const jitter = Math.sqrt(variance);

    // Consecutive variance (alternative metric)
    let consecutiveVariance = 0;
    for (let i = 1; i < rttHistory.length; i++) {
      consecutiveVariance += Math.abs(rttHistory[i] - rttHistory[i-1]);
    }
    consecutiveVariance /= (rttHistory.length - 1);

    isHighLatency = median > HIGH_LATENCY_RTT_THRESHOLD;
    isHighJitter = jitter > HIGH_JITTER_THRESHOLD || consecutiveVariance > HIGH_VARIANCE_THRESHOLD;

    if (navigator.connection) {
      isOffline = navigator.connection.type === 'none' || navigator.connection.rtt > 10000;
      isHighLatency = isHighLatency || navigator.connection.downlinkMax < HIGH_LATENCY_DOWNLINK_THRESHOLD;
    }

    updateConnectivityUI();
  } catch (e) {
    isOffline = true;
    updateConnectivityUI();
  }
}

function updateConnectivityUI() {
  let status = '';
  if (isOffline) {
    status = 'Offline mode — queued actions will sync later ⚡️';
  } else if (isHighJitter) {
    status = 'High jitter detected (Starlink spikes?) — increasing batch size & compression ⚡️';
  } else if (isHighLatency) {
    status = 'High latency (Starlink mode?) — batching & compressing ⚡️';
  } else {
    status = 'Strong connection — lattice fully online ⚡️';
  }
  showToast(status);
}

// Start periodic probe
let probeInterval;
function startProbes() {
  probeInterval = setInterval(probeRTT, RTT_PROBE_INTERVAL);
  probeRTT(); // immediate first probe
}

function stopProbes() {
  if (probeInterval) clearInterval(probeInterval);
}

// Visibility handling
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopProbes();
  } else {
    startProbes();
  }
});

// Initial start
startProbes();

// Queue voice note if offline/high-latency/jitter
async function startVoiceRecording(sessionId, isEmergency = false) {
  // ... existing recording logic ...
  if (isOffline || isHighLatency || isHighJitter) {
    await rathorDB.saveQueuedAction('voice-note', { sessionId, blob, timestamp, isEmergency });
    showToast('Voice note queued for reconnection ⚡️');
  } else {
    await rathorDB.saveVoiceNote(sessionId, blob, timestamp, isEmergency);
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopProbes();
  } else {
    startProbes();
  }
});

// Initial start
startProbes();

// Queue voice note if offline/high-latency
async function startVoiceRecording(sessionId, isEmergency = false) {
  // ... existing recording logic ...
  if (isOffline || isHighLatency) {
    await rathorDB.saveQueuedAction('voice-note', { sessionId, blob, timestamp, isEmergency });
    showToast('Voice note queued for reconnection ⚡️');
  } else {
    await rathorDB.saveVoiceNote(sessionId, blob, timestamp, isEmergency);
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

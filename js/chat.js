// js/chat.js — Rathor Lattice Core with Periodic RTT Probes + Packet Loss Detection

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
let isHighPacketLoss = false;
let rttHistory = [];           // last 5 successful RTTs (ms)
let packetLossHistory = [];    // last 5 loss ratios (0–1)
const PROBE_COUNT = 5;         // packets per probe cycle
const PROBE_INTERVAL = 12000;  // 12 seconds
const HIGH_LATENCY_RTT_THRESHOLD = 150;
const HIGH_JITTER_THRESHOLD = 50;
const HIGH_PACKET_LOSS_THRESHOLD = 0.10; // 10%
const HIGH_VARIANCE_THRESHOLD = 40;

// Ping endpoint (lightweight, can be /ping or external like 1.1.1.1)
const PING_ENDPOINT = '/ping'; // or 'https://1.1.1.1/cdn-cgi/trace'

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

// ... existing event listeners ...

// ────────────────────────────────────────────────
// Periodic Multi-Packet RTT + Loss Probes
// ────────────────────────────────────────────────

async function probeConnectivity() {
  if (document.hidden) return;

  let successes = 0;
  let totalRtt = 0;
  const probeTimes = [];

  for (let i = 0; i < PROBE_COUNT; i++) {
    const start = performance.now();
    try {
      await fetch(`\( {PING_ENDPOINT}?t= \){Date.now() + i}`, { 
        cache: 'no-store', 
        mode: 'no-cors',
        keepalive: true
      });
      const end = performance.now();
      const rtt = end - start;
      probeTimes.push(rtt);
      totalRtt += rtt;
      successes++;
    } catch (e) {
      probeTimes.push(null);
    }
    await new Promise(r => setTimeout(r, 200)); // small spacing
  }

  const lossRatio = (PROBE_COUNT - successes) / PROBE_COUNT;

  if (successes > 0) {
    const avgRtt = totalRtt / successes;
    rttHistory.push(avgRtt);
    if (rttHistory.length > 5) rttHistory.shift();

    packetLossHistory.push(lossRatio);
    if (packetLossHistory.length > 5) packetLossHistory.shift();

    // Median RTT
    const sortedRtt = [...rttHistory].sort((a,b)=>a-b);
    const medianRtt = sortedRtt[Math.floor(sortedRtt.length/2)];

    // Jitter (std dev)
    const meanRtt = rttHistory.reduce((a,b)=>a+b,0) / rttHistory.length;
    const variance = rttHistory.reduce((a,b)=>a + Math.pow(b-meanRtt,2),0) / rttHistory.length;
    const jitter = Math.sqrt(variance);

    // Median loss
    const medianLoss = [...packetLossHistory].sort((a,b)=>a-b)[Math.floor(packetLossHistory.length/2)];

    isHighLatency = medianRtt > HIGH_LATENCY_RTT_THRESHOLD;
    isHighJitter = jitter > HIGH_JITTER_THRESHOLD;
    isHighPacketLoss = medianLoss > HIGH_PACKET_LOSS_THRESHOLD;

    if (navigator.connection) {
      isOffline = navigator.connection.type === 'none' || navigator.connection.rtt > 10000;
      isHighLatency = isHighLatency || navigator.connection.downlinkMax < 10;
    }
  } else {
    isOffline = true;
  }

  updateConnectivityUI();
}

function updateConnectivityUI() {
  let status = '';
  if (isOffline) {
    status = 'Offline — queued actions will sync later ⚡️';
  } else if (isHighPacketLoss) {
    status = 'High packet loss detected (Starlink obstruction?) — ultra-batch mode ⚠️';
  } else if (isHighJitter) {
    status = 'High jitter — increasing batch size & compression ⚡️';
  } else if (isHighLatency) {
    status = 'High latency (Starlink mode?) — batching & compressing ⚡️';
  } else {
    status = 'Strong connection — lattice fully online ⚡️';
  }
  showToast(status);
}

// Probe loop
let probeInterval;
function startProbes() {
  probeInterval = setInterval(probeConnectivity, PROBE_INTERVAL);
  probeConnectivity(); // immediate
}

function stopProbes() {
  if (probeInterval) clearInterval(probeInterval);
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopProbes();
  else startProbes();
});

// Initial start
startProbes();

// Adaptive voice recording queue
async function startVoiceRecording(sessionId, isEmergency = false) {
  // ... existing recording logic ...
  if (isOffline || isHighLatency || isHighJitter || isHighPacketLoss) {
    await rathorDB.saveQueuedAction('voice-note', { sessionId, blob, timestamp, isEmergency });
    showToast('Voice note queued — unstable connection ⚡️');
  } else {
    await rathorDB.saveVoiceNote(sessionId, blob, timestamp, isEmergency);
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...

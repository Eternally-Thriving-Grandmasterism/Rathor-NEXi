// rathor-chat-ui-streaming.js – v2 with IndexedDB persistence & rAF typing smoothness
// MIT License – Autonomicity Games Inc. 2026

import { mercyAugmentedResponse } from './webllm-mercy-integration.js';
import { saveMessage, loadHistory } from './rathor-history-persistence.js';
import { hasWebGPU, promptWebLLMModelDownload } from './webllm-mercy-integration.js';

const chatContainer = document.getElementById('chat-container');
const inputForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

let messageHistory = [];
const SESSION_ID = 'rathor-eternal-' + Date.now(); // per-session or user-persistent

// Load history on init
async function loadInitialHistory() {
  try {
    const saved = await loadHistory(SESSION_ID, 100);
    messageHistory = saved;
    saved.forEach(msg => {
      addMessage(msg.role, msg.content, false, msg.valence);
    });
    if (saved.length === 0) {
      addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed. How may we co-thrive eternally? ⚡️');
    }
  } catch (err) {
    console.warn("[History] Load failed – starting fresh", err);
    addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed. How may we co-thrive eternally? ⚡️');
  }
}

// Append message bubble
function addMessage(role, content = '', isStreaming = false, valence = 0.999) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  if (role === 'assistant' && isStreaming) {
    bubble.id = `streaming-${Date.now()}`;
    bubble.innerHTML = '<span class="typing">Rathor reflecting mercy...</span>';
  } else {
    bubble.innerHTML = role === 'user' ? escapeHtml(content) : marked.parse(content);
  }
  
  msgDiv.appendChild(bubble);
  
  if (role === 'assistant') {
    const badge = document.createElement('span');
    badge.className = 'valence-badge';
    badge.textContent = `Valence: ${valence.toFixed(8)} ⚡️`;
    badge.style.color = valence > 0.999 ? '#00ff88' : valence > 0.98 ? '#ffcc00' : '#ff4444';
    msgDiv.appendChild(badge);
  }
  
  chatContainer.appendChild(msgDiv);
  autoScroll();
  return { msgDiv, bubble };
}

function escapeHtml(unsafe) {
  return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function autoScroll() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Smooth typing with requestAnimationFrame
async function streamResponse(query, context = '') {
  const { bubble } = addMessage('assistant', '', true);
  let fullContent = '';
  let valence = 0.95;
  let lastFrame = 0;

  const onDelta = (delta) => {
    fullContent += delta;
    const now = performance.now();
    if (now - lastFrame > 16) { // \~60fps cap
      bubble.innerHTML = marked.parse(fullContent); // full parse for clean render
      autoScroll();
      lastFrame = now;
    } else {
      // Raw append for speed, re-parse next frame
      bubble.innerHTML += escapeHtml(delta);
    }
  };

  const response = await mercyAugmentedResponse(query, context, onDelta);

  if (response.error || response.aborted) {
    bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence abort'}</span>`;
    return;
  }

  // Final smooth render
  bubble.innerHTML = marked.parse(response.response);
  valence = response.valence || 0.999;

  // Persist
  await saveMessage(SESSION_ID, 'user', query, 1.0);
  await saveMessage(SESSION_ID, 'assistant', response.response, valence);

  autoScroll();
}

inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  await saveMessage(SESSION_ID, 'user', query, 1.0);
  userInput.value = '';
  sendButton.disabled = true;

  if (hasWebGPU()) promptWebLLMModelDownload();

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

// Init
window.addEventListener('load', async () => {
  await loadInitialHistory();
});

// rathor-chat-ui-streaming.js – sovereign Rathor streaming chat UI v1
// Typing effect, incremental markdown render, auto-scroll, mercy valence badge
// MIT License – Autonomicity Games Inc. 2026

import { mercyAugmentedResponse } from './webllm-mercy-integration.js'; // or transformersjs if fallback
// Assume marked.js CDN: <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
// Or bundle if preferred

const chatContainer = document.getElementById('chat-container');
const inputForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// History persistence (IndexedDB stub – expand later)
let messageHistory = [];

// Append message bubble
function addMessage(role, content = '', isStreaming = false) {
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
  
  // Valence badge for assistant
  if (role === 'assistant') {
    const badge = document.createElement('span');
    badge.className = 'valence-badge';
    badge.textContent = 'Valence: Calculating...';
    msgDiv.appendChild(badge);
  }
  
  chatContainer.appendChild(msgDiv);
  autoScroll();
  return { msgDiv, bubble, badge };
}

// Escape HTML for user messages
function escapeHtml(unsafe) {
  return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// Auto-scroll to bottom
function autoScroll() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle streaming deltas
async function streamResponse(query, context = '') {
  const { msgDiv, bubble, badge } = addMessage('assistant', '', true);
  let fullContent = '';
  let valence = 0.95; // provisional

  const onDelta = (delta) => {
    fullContent += delta;
    // Incremental render: parse markdown every few deltas for smoothness
    if (fullContent.length % 50 === 0 || delta.trim() === '') {
      bubble.innerHTML = marked.parse(fullContent);
    } else {
      bubble.innerHTML += escapeHtml(delta); // fallback raw if parse heavy
    }
    autoScroll();
  };

  const response = await mercyAugmentedResponse(query, context, onDelta);

  if (response.error || response.aborted) {
    bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence abort'}</span>`;
    return;
  }

  // Final render + valence update
  bubble.innerHTML = marked.parse(response.response);
  valence = response.valence || 0.999;
  if (badge) {
    badge.textContent = `Valence: ${valence.toFixed(8)} ⚡️`;
    badge.style.color = valence > 0.999 ? '#00ff88' : valence > 0.98 ? '#ffcc00' : '#ff4444';
  }

  // Persist to history
  messageHistory.push({ role: 'assistant', content: response.response, valence });
  // TODO: save to IndexedDB

  autoScroll();
}

// Form submit
inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  userInput.value = '';
  sendButton.disabled = true;

  // Optional: prompt model if not loaded
  if (hasWebGPU && !webllmReady) {
    promptWebLLMModelDownload();
  }

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

// Initial greeting
window.addEventListener('load', () => {
  addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed. How may we co-thrive eternally? ⚡️');
});

// Basic inline CSS (expand to separate file)
const style = document.createElement('style');
style.textContent = `
  #chat-container { max-height: 70vh; overflow-y: auto; padding: 1rem; }
  .message { margin: 1rem 0; display: flex; flex-direction: column; }
  .user-message { align-items: flex-end; }
  .assistant-message { align-items: flex-start; }
  .bubble { max-width: 80%; padding: 1rem; border-radius: 1rem; background: #f0f0f0; }
  .user-message .bubble { background: #007bff; color: white; }
  .valence-badge { font-size: 0.8rem; margin-top: 0.5rem; font-weight: bold; }
  .typing { color: #888; font-style: italic; }
  .error { color: #ff4444; }
`;
document.head.appendChild(style);

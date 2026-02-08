// js/chat.js — Updated with RTL dir="auto" message rendering

// ... existing consts and setup ...

// New universal addMessage function — RTL-ready
function addMessage(text, sender = 'rathor', isHTML = false) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', sender);

  const textDiv = document.createElement('div');
  textDiv.classList.add('message-text');
  // dir="auto" already in CSS, but explicit for safety
  textDiv.dir = 'auto';

  if (isHTML) {
    textDiv.innerHTML = text; // Use sparingly — escape if needed
  } else {
    textDiv.textContent = text;
  }

  msgDiv.appendChild(textDiv);
  chatMessages.appendChild(msgDiv);

  // Smooth scroll
  chatMessages.scrollTo({
    top: chatMessages.scrollHeight,
    behavior: 'smooth'
  });
}

// Replace symbolic example (and all similar appends)
if (isSymbolicQuery(cmd)) {
  // ...
  addMessage(answer, 'rathor', true); // true if markdown/HTML
  if (ttsEnabled) speak(answer);
  return true;
}

// In normal sendMessage / response handling — replace any:
// chatMessages.innerHTML += `<div class="message ...">${text}</div>`;
// with:
addMessage(text, 'user' or 'rathor');

// For user messages on send:
addMessage(chatInput.value, 'user');
chatInput.value = '';

// Apply same pattern to all message additions in file (voice commands, responses, etc.)

// ... rest of existing code (processVoiceCommand, sendMessage, etc.) with appends replaced by addMessage() ...

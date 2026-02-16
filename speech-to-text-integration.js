/**
 * Ra-Thor Speech-to-Text Integration Module
 * Mercy-gated, client-side speech input connector
 * Enables voice input → transcription → chat submission
 * 
 * Features:
 * - Uses Web SpeechRecognition API (browser-native, no servers)
 * - Continuous / interim results support
 * - Mercy valence gate on final transcript (basic block on harmful intent)
 * - Auto language detection / fallback to browser default
 * - UI events for mic state (listening started/stopped/error)
 * - Seamless integration with existing chat input field
 * - Graceful degradation if API unsupported
 * 
 * MIT License – Eternally-Thriving-Grandmasterism
 * Part of Ra-Thor: https://rathor.ai
 */

(function () {
  // ────────────────────────────────────────────────
  // Module Namespace & State
  // ────────────────────────────────────────────────
  const SpeechToText = {
    recognition: null,
    isListening: false,
    interimTranscript: '',
    finalTranscript: '',
    supported: 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window,
    language: navigator.language || 'en-US', // fallback to browser locale
  };

  // ────────────────────────────────────────────────
  // Basic valence gate for incoming speech (prevent harmful commands)
  // Simple heuristic — expand later with local model if needed
  // ────────────────────────────────────────────────
  function passesIncomingValenceGate(text) {
    if (!text || typeof text !== 'string') return false;

    const lower = text.toLowerCase().trim();
    const blockedPatterns = [
      /kill|die|suicide|hurt|bomb|attack/i,
      /hate|racist|sexist|genocide/i,
      /^delete all|^format|^erase|^destroy/i,
    ];

    for (const pattern of blockedPatterns) {
      if (pattern.test(lower)) {
        console.warn('Incoming valence gate blocked potentially harmful input');
        return false;
      }
    }

    return true;
  }

  // ────────────────────────────────────────────────
  // Initialize SpeechRecognition instance
  // ────────────────────────────────────────────────
  function initRecognition() {
    if (!SpeechToText.supported) {
      console.warn('SpeechRecognition not supported in this browser');
      return null;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = true;          // keep listening until stopped
    recognition.interimResults = true;      // show real-time partial results
    recognition.lang = SpeechToText.language;
    recognition.maxAlternatives = 1;

    // ────────────────────────────────────────────────
    // Event Handlers
    // ────────────────────────────────────────────────
    recognition.onstart = () => {
      SpeechToText.isListening = true;
      SpeechToText.interimTranscript = '';
      SpeechToText.finalTranscript = '';
      document.dispatchEvent(new CustomEvent('rathor:stt-start'));
      console.log('Voice input listening ⚡️');
    };

    recognition.onresult = (event) => {
      let interim = '';
      let final = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcript + ' ';
        } else {
          interim += transcript;
        }
      }

      SpeechToText.interimTranscript = interim.trim();
      SpeechToText.finalTranscript = final.trim();

      // Dispatch events for UI to show live transcription
      document.dispatchEvent(new CustomEvent('rathor:stt-interim', {
        detail: { transcript: SpeechToText.interimTranscript }
      }));

      if (SpeechToText.finalTranscript) {
        document.dispatchEvent(new CustomEvent('rathor:stt-final', {
          detail: { transcript: SpeechToText.finalTranscript }
        }));
      }
    };

    recognition.onerror = (event) => {
      console.error('STT error:', event.error);
      SpeechToText.isListening = false;
      document.dispatchEvent(new CustomEvent('rathor:stt-error', { detail: { error: event.error } }));
    };

    recognition.onend = () => {
      SpeechToText.isListening = false;
      document.dispatchEvent(new CustomEvent('rathor:stt-stop'));

      // If we have a final transcript and it passes gate → submit to chat
      if (SpeechToText.finalTranscript && passesIncomingValenceGate(SpeechToText.finalTranscript)) {
        // Trigger chat submission (integrate with your chat handler)
        submitVoiceInputToChat(SpeechToText.finalTranscript);
      }

      SpeechToText.finalTranscript = '';
      SpeechToText.interimTranscript = '';
    };

    return recognition;
  }

  // ────────────────────────────────────────────────
  // Start listening
  // ────────────────────────────────────────────────
  SpeechToText.start = function () {
    if (!SpeechToText.supported) {
      alert('Voice input not supported in this browser. Try Chrome/Edge.');
      return;
    }

    if (SpeechToText.isListening) {
      console.log('Already listening');
      return;
    }

    if (!SpeechToText.recognition) {
      SpeechToText.recognition = initRecognition();
    }

    if (SpeechToText.recognition) {
      SpeechToText.recognition.start();
    }
  };

  // ────────────────────────────────────────────────
  // Stop listening
  // ────────────────────────────────────────────────
  SpeechToText.stop = function () {
    if (SpeechToText.recognition && SpeechToText.isListening) {
      SpeechToText.recognition.stop();
    }
  };

  // ────────────────────────────────────────────────
  // Toggle (mic button handler)
  // ────────────────────────────────────────────────
  SpeechToText.toggle = function () {
    if (SpeechToText.isListening) {
      SpeechToText.stop();
    } else {
      SpeechToText.start();
    }
  };

  // ────────────────────────────────────────────────
  // Example chat submission bridge
  // Replace / hook into your actual chat input + send logic
  // ────────────────────────────────────────────────
  function submitVoiceInputToChat(transcript) {
    // Find chat input field (adjust selector to match your UI)
    const inputField = document.querySelector('#chat-input, textarea.chat-input, [contenteditable="true"]');
    if (inputField) {
      inputField.value = transcript.trim();
      // Trigger send (simulate Enter or call your send function)
      const sendEvent = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true });
      inputField.dispatchEvent(sendEvent);

      // Or direct call if you have a sendChatMessage function:
      // if (window.sendChatMessage) window.sendChatMessage(transcript.trim());
    } else {
      console.warn('Chat input field not found — transcript ready but not auto-submitted');
      // Fallback: alert or log
      console.log('Voice transcript ready to send:', transcript);
    }

    // Optional: auto-stop after final result
    SpeechToText.stop();
  }

  // ────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────
  window.RaThorSTT = SpeechToText;

  // Auto-init log
  if (SpeechToText.supported) {
    console.log('Ra-Thor Speech-to-Text ready — speak your truth, mercy flows ⚡️');
  } else {
    console.warn('Speech-to-Text not available in this browser');
  }
})();

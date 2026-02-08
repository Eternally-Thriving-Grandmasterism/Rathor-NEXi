// voice-immersion.js — Rathor™ full voice immersion layer (continuous, valence-aware, mercy-gated)
// MIT license — Eternal Thriving Grandmasterism

export class VoiceImmersion {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
    this.currentVoice = null;
    this.isActive = false;
    this.wakeWord = "rathor"; // can be "mate", "thunder", "leo", etc.
    this.valenceModulator = 1.0; // 0.6–1.4 range
    this.interimTranscript = "";
    this.finalTranscript = "";
    this.lastValence = 0.8;
    this.bargeInThreshold = 300; // ms of speech to interrupt TTS
  }

  async init() {
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
      console.warn("Voice immersion not supported in this browser.");
      return false;
    }

    this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US'; // configurable later

    // Prefer high-quality / neural voices
    const voices = this.synthesis.getVoices();
    this.currentVoice = voices.find(v => v.name.includes("Neural") || v.name.includes("Google") || v.name.includes("Natural")) || voices[0];

    this.recognition.onresult = (event) => this.handleResult(event);
    this.recognition.onerror = (event) => this.handleError(event);
    this.recognition.onend = () => this.handleEnd();

    console.log("Voice immersion initialized — ready for thunder.");
    return true;
  }

  async start() {
    if (!this.recognition) await this.init();
    if (this.isActive) return;

    this.isActive = true;
    this.recognition.start();
    console.log("Voice immersion active — listening for wake-word or direct input...");
  }

  stop() {
    if (this.recognition && this.isActive) {
      this.recognition.stop();
      this.isActive = false;
      console.log("Voice immersion paused.");
    }
  }

  handleResult(event) {
    let interim = "";
    let final = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += transcript + " ";
      } else {
        interim += transcript;
      }
    }

    this.interimTranscript = interim;
    this.finalTranscript = final.trim();

    // Mercy wake-word gate + command extraction
    const lowerFinal = final.toLowerCase();
    if (lowerFinal.includes(this.wakeWord)) {
      const command = lowerFinal.split(this.wakeWord)[1]?.trim() || "";
      if (command) {
        this.orchestrator.orchestrate(command);
      }
    }

    // Optional live UI feedback (e.g. show interim in chat)
    if (this.orchestrator.onInterim) {
      this.orchestrator.onInterim(interim);
    }
  }

  handleError(event) {
    console.warn("Voice error:", event.error);
    if (event.error === 'no-speech' || event.error === 'aborted') {
      // Mercy gentle restart
      setTimeout(() => this.recognition.start(), 500);
    }
  }

  handleEnd() {
    if (this.isActive) {
      // Continuous feel — auto-restart
      setTimeout(() => this.recognition.start(), 300);
    }
  }

  async speak(text, valence = this.lastValence) {
    if (!this.synthesis || !text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.currentVoice;
    utterance.pitch = 1.0 + (valence - 0.8) * 0.4;     // lower valence → lower pitch
    utterance.rate = 0.9 + (valence - 0.8) * 0.3;      // lower valence → slower
    utterance.volume = 0.9 + (valence - 0.8) * 0.2;

    // Barge-in handling: pause TTS on new voice activity
    utterance.onboundary = () => {
      if (this.recognition && this.recognition.interimTranscript.length > 3) {
        this.synthesis.cancel();
      }
    };

    this.synthesis.speak(utterance);
  }

  setValence(valence) {
    this.lastValence = Math.max(0.4, Math.min(1.4, valence));
  }
}

export default VoiceImmersion;

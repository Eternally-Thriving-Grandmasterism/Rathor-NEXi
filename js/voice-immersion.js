// voice-immersion.js — Rathor™ full voice immersion layer (continuous, valence-aware, mercy-gated)
// MIT license — Eternal Thriving Grandmasterism

export class VoiceImmersion {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
    this.currentVoice = null;
    this.isActive = false;
    this.wakeWord = "rathor"; // or "mate", "thunder", configurable
    this.valenceModulator = 1.0; // 0.6–1.4 range
    this.interimTranscript = "";
    this.finalTranscript = "";
    this.lastValence = 0.8;
  }

  async init() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn("Voice immersion not supported in this browser.");
      return false;
    }

    this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US'; // configurable later

    // Voice selection (prefer neural voices)
    const voices = this.synthesis.getVoices();
    this.currentVoice = voices.find(v => v.name.includes("Neural") || v.name.includes("Google")) || voices[0];

    this.recognition.onresult = (event) => this.handleResult(event);
    this.recognition.onerror = (event) => this.handleError(event);
    this.recognition.onend = () => this.handleEnd();

    return true;
  }

  async start() {
    if (!this.recognition) await this.init();
    if (this.isActive) return;

    this.isActive = true;
    this.recognition.start();
    console.log("Voice immersion active — listening for thunder...");
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

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += transcript;
      } else {
        interim += transcript;
      }
    }

    this.interimTranscript = interim;
    this.finalTranscript = final;

    // Mercy wake-word gate
    if (final.toLowerCase().includes(this.wakeWord)) {
      const command = final.toLowerCase().replace(this.wakeWord, '').trim();
      if (command) {
        this.orchestrator.orchestrate(command);
      }
    }

    // Live interim feedback (optional UI update)
    if (this.orchestrator.onInterim) {
      this.orchestrator.onInterim(interim);
    }
  }

  handleError(event) {
    console.warn("Voice error:", event.error);
    if (event.error === 'no-speech' || event.error === 'aborted') {
      // Mercy restart — gentle
      setTimeout(() => this.recognition.start(), 500);
    }
  }

  handleEnd() {
    if (this.isActive) {
      // Mercy auto-restart for continuous feel
      setTimeout(() => this.recognition.start(), 300);
    }
  }

  async speak(text, valence = this.lastValence) {
    if (!this.synthesis) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.currentVoice;
    utterance.pitch = 1.0 + (valence - 0.8) * 0.4; // 0.6→0.84, 1.0→1.0,

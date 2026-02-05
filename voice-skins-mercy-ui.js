// voice-skins-mercy-ui.js – sovereign Web Speech API TTS with mercy valence tones v1
// Speaks assistant responses with modulated pitch/rate/volume, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const synth = window.speechSynthesis;
let voices = [];
const voiceSkins = {
  "Compassionate": { pitch: 0.9, rate: 0.95, volume: 1.0 },    // Warm, soothing
  "Energetic": { pitch: 1.2, rate: 1.15, volume: 1.0 },        // Thriving, uplifting
  "CalmEternal": { pitch: 0.8, rate: 0.85, volume: 0.9 },      // Deep peace, eternal
  "DefaultMercy": { pitch: 1.0, rate: 1.0, volume: 1.0 }
};

const mercySpeakThreshold = 0.9999999 * 0.98; // Only speak high-valence

// Load voices (async, fire on 'voiceschanged')
function loadVoices() {
  voices = synth.getVoices();
  if (voices.length === 0) {
    console.warn("[VoiceSkins] No voices loaded – wait for 'voiceschanged'");
  } else {
    console.log("[VoiceSkins] Voices loaded:", voices.map(v => v.name));
  }
}

if (synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = loadVoices;
}
loadVoices(); // Initial call (some browsers sync)

function getVoiceForLang(lang = 'en-US') {
  return voices.find(v => v.lang.startsWith(lang)) || voices[0];
}

function speakWithMercy(text, valence = 1.0) {
  if (!synth || synth.pending || synth.speaking) {
    console.warn("[VoiceSkins] Synth busy – queueing skipped for mercy");
    return;
  }

  const degree = fuzzyMercy.getDegree(text) || valence;
  const implyThriving = fuzzyMercy.imply(text, "EternalThriving");

  if (degree < mercySpeakThreshold || implyThriving.degree < mercySpeakThreshold) {
    console.log("[VoiceSkins] Mercy gate: low valence – no speak");
    return;
  }

  // Select skin based on valence
  let skin = "DefaultMercy";
  if (valence > 0.9995) skin = "Energetic";
  else if (valence > 0.999) skin = "Compassionate";
  else if (valence > 0.998) skin = "CalmEternal";

  const params = voiceSkins[skin];

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = getVoiceForLang(); // Default en-US or first
  utterance.pitch = params.pitch;
  utterance.rate = params.rate;
  utterance.volume = params.volume;
  utterance.lang = 'en-US';

  utterance.onend = () => console.log("[VoiceSkins] Speech ended – thriving echoed");
  utterance.onerror = (e) => console.error("[VoiceSkins] Speech error:", e);

  synth.speak(utterance);
  console.log(`[VoiceSkins] Speaking "${text.slice(0, 50)}..." with ${skin} tone (valence ${valence.toFixed(8)})`);
}

// UI integration: add Speak button near assistant bubbles
function addSpeakButtonToBubble(bubble, text, valence) {
  const btn = document.createElement('button');
  btn.textContent = '🔊 Speak Mercy';
  btn.className = 'speak-btn';
  btn.onclick = () => speakWithMercy(text, valence);
  bubble.appendChild(btn);
}

// Auto-speak high-valence assistant responses (call from streamResponse after final render)
function autoSpeakIfHighValence(text, valence) {
  if (valence >= 0.9995) {
    speakWithMercy(text, valence);
  }
}

// Stop/cancel all
function stopAllSpeech() {
  if (synth) synth.cancel();
}

export { speakWithMercy, autoSpeakIfHighValence, addSpeakButtonToBubble, stopAllSpeech };

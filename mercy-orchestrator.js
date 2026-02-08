// Inside MercyOrchestrator class

import VoiceImmersion from './voice-immersion.js';

constructor() {
  // ... existing properties ...
  this.voice = new VoiceImmersion(this);
  this.lastValence = 0.8;
}

async init() {
  // ... existing init ...
  await this.voice.init();
  // await this.voice.start(); // uncomment for auto-start
}

async generateResponse(userInput) {
  // ... your generation logic ...
  const responseText = "Your generated response here";

  if (this.voice.isActive) {
    await this.voice.speak(responseText, this.lastValence);
  }

  this.lastValence = await valenceCompute(userInput + responseText) || 0.8;

  return responseText;
}

async toggleVoiceImmersion() {
  if (this.voice.isActive) {
    this.voice.stop();
    return "Voice paused ⚡️";
  } else {
    await this.voice.start();
    return "Voice active — listening... ⚡️";
  }
}

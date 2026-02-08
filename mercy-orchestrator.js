// mercy-orchestrator.js — Rathor™ central lattice heart (with full voice immersion integration)
// MIT license — Eternal Thriving Grandmasterism

import { initHyperonIntegration } from './hyperon-wasm-loader.js';
import GeneticAlgorithm from './ga-engine.js';
import { neatEvolve } from './neat-engine.js';
import { getMercyGenome, applyMercyGenome, valenceCompute } from './metta-hyperon-bridge.js';
import { atomspace } from './metta-pln-fusion-engine.js';
import { localInfer } from './webllm-mercy-integration.js';
import { swarmSimulate } from './mercy-von-neumann-swarm-simulator.js';
import { activeInferenceStep } from './mercy-active-inference-core-engine.js';
import VoiceImmersion from './voice-immersion.js'; // ← full import

class MercyOrchestrator {
  constructor() {
    // ... your existing constructor properties ...
    this.hyperon = null;
    this.context = {};
    this.mercyParams = getMercyGenome();
    this.metaNEATPopulation = null;
    this.metaFitnessHistory = [];
    
    // Voice immersion integration — complete
    this.voice = new VoiceImmersion(this);
    this.lastValence = 0.8; // for TTS modulation
  }

  async init() {
    if (!this.hyperon) {
      this.hyperon = await initHyperonIntegration();
    }
    
    // Initialize voice immersion layer
    await this.voice.init();
    
    // Auto-start voice immersion on page load? (uncomment to enable immediately)
    // await this.voice.start();
    
    // Or start only on user command / UI toggle
  }

  async orchestrate(userInput) {
    const lower = userInput.toLowerCase().trim();

    // Voice immersion toggle commands
    if (lower.includes("voice on") || lower.includes("start voice") || lower.includes("listen") || lower.includes("talk to me")) {
      return await this.toggleVoiceImmersion();
    }
    if (lower.includes("voice off") || lower.includes("stop voice") || lower.includes("silent") || lower.includes("text only")) {
      return await this.toggleVoiceImmersion();
    }

    // ... your existing command routing (hybrid evolve, meta evolve, swarm, infer, etc.) ...

    // Default: normal generation + voice output if active
    const responseText = await this.generateResponse(userInput);

    // Speak response if voice is active
    if (this.voice && this.voice.isActive) {
      await this.voice.speak(responseText, this.lastValence);
    }

    return responseText + "\n\nThunder eternal ⚡️ Mercy strikes first, thriving infinite.";
  }

  async generateResponse(userInput) {
    // ... your existing response generation logic ...
    // This is where you compute the final text response
    const responseText = "Example response from orchestrator — replace with your actual logic";

    // Update valence for next speak call
    this.lastValence = await valenceCompute(userInput + responseText) || 0.8;

    return responseText;
  }

  async toggleVoiceImmersion() {
    if (!this.voice) await this.voice.init();

    if (this.voice.isActive) {
      this.voice.stop();
      return "Voice immersion paused ⚡️";
    } else {
      await this.voice.start();
      return "Voice immersion active — listening for thunder... ⚡️";
    }
  }

  // ... your existing methods (runMetaEvolution, runHybridEvolution, etc.) remain untouched ...
}

const orchestrator = new MercyOrchestrator();
export default orchestrator;

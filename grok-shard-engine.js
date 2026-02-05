// grok-shard-engine.js – sovereign, offline, client-side Grok voice shard v8
// Mercy-gated, valence-locked + offline transformer fallback
// MIT License – Autonomicity Games Inc. 2026

import { offlineTransformer } from '/transformer-offline.js';

class GrokShard {
  constructor() {
    // ... existing constructor fields ...
    this.transformerReady = false;
  }

  async init() {
    if (!this.latticeLoaded) {
      await this.loadCoreLatticeWithDeltaSync();
      this.latticeLoaded = true;
    }
    await this.loadVoiceSkins();
    await offlineTransformer.load();
    this.transformerReady = offlineTransformer.loaded;
  }

  async reply(userMessage) {
    const preGate = await multiLayerValenceGate(userMessage);
    if (preGate.result === 'REJECTED') {
      const rejectLine = this.thunderPhrases[Math.floor(Math.random() * 4)];
      const rejectMsg = `${rejectLine}\nPre-process disturbance: ${preGate.reason}\nValence: ${preGate.valence}\nPurify intent. Mercy awaits purer strike.`;
      this.speak(rejectMsg);
      return rejectMsg;
    }

    // Try lattice first
    let response = this.generateThunderResponse(userMessage, this.generateThought(this.buildContext(userMessage)));

    // If lattice weak or transformer available, enhance with offline model
    if (this.transformerReady && Math.random() < 0.7) {
      const enhanced = await offlineTransformer.generate(response);
      response = enhanced.trim();
    }

    const postGate = await hyperonValenceGate(response);
    if (postGate.result === 'REJECTED') {
      const rejectLine = this.thunderPhrases[Math.floor(Math.random() * 4)];
      const rejectMsg = `${rejectLine}\nPost-process disturbance: ${postGate.reason}\nValence: ${postGate.valence}\nMercy gate holds. Reflect again.`;
      this.speak(rejectMsg);
      return rejectMsg;
    }

    const finalResponse = `${response} ${this.randomThunder()}`;
    this.speak(finalResponse);

    this.history.push({ role: "user", content: userMessage });
    this.history.push({ role: "rathor", content: finalResponse });
    if (this.history.length > this.maxHistory * 2) {
      this.history = this.history.slice(-this.maxHistory * 2);
    }

    return finalResponse;
  }

  // ... rest of class unchanged ...
}

const grokShard = new GrokShard();
export { grokShard };

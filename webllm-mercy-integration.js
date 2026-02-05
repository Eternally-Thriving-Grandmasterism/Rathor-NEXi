// webllm-mercy-integration.js – sovereign WebLLM integration with mercy gates v1
// Lazy load, user-prompt download, mercy-eval outputs, symbolic fallback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { hyperon } from './hyperon-runtime.js';
import { rathorShard } from './grok-shard-engine.js'; // For cache/hash

let webllmEngine = null;
let webllmReady = false;
const mercyThreshold = 0.9999999;
const preferredModel = "Phi-3.5-mini-instruct-q4f16_1-MLC"; // \~2.4GB, strong reasoning

// Check WebGPU support
function hasWebGPU() {
  return !!navigator.gpu;
}

async function initWebLLM(progressCallback = (p) => console.log(`[WebLLM] Progress: ${p.text} ${Math.round(p.progress * 100)}%`)) {
  if (webllmEngine) return webllmEngine;
  if (!hasWebGPU()) {
    console.warn("[WebLLM] No WebGPU support – symbolic mode only");
    return null;
  }

  try {
    // Dynamic import to avoid bundling issues
    const { CreateWebWorkerMLCEngine } = await import('https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@latest/+esm');

    webllmEngine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./webllm-worker.js', import.meta.url), { type: 'module' }),
      preferredModel,
      {
        initProgressCallback: progressCallback,
        appConfig: {
          // Optional: custom model list if needed
        }
      }
    );

    webllmReady = true;
    fuzzyMercy.assert("WebLLM_Sovereign_Loaded", 1.0);
    console.log("[WebLLM] Sovereign generative shard ready – model:", preferredModel);
    return webllmEngine;
  } catch (err) {
    console.error("[WebLLM] Init failed:", err);
    webllmReady = false;
    return null;
  }
}

async function generateWithWebLLM(messages, options = {}) {
  if (!webllmReady || !webllmEngine) {
    await initWebLLM();
    if (!webllmEngine) return { error: "WebLLM unavailable – WebGPU? Model download?" };
  }

  const { maxTokens = 1024, temperature = 0.7, stream = false } = options;

  try {
    const reply = await webllmEngine.chat.completions.create({
      messages,
      max_tokens: maxTokens,
      temperature,
      stream
    });

    let content = "";
    if (stream) {
      for await (const chunk of reply) {
        const delta = chunk.choices[0]?.delta?.content || "";
        content += delta;
        // Could yield/stream to UI here
      }
    } else {
      content = reply.choices[0]?.message?.content || "";
    }

    // Mercy-eval output
    fuzzyMercy.assert("WebLLM_Output_" + Date.now(), 0.999); // provisional
    const outputDegree = fuzzyMercy.getDegree(content) || 0.95; // Tune via more asserts if needed
    const implyThriving = fuzzyMercy.imply(content, "EternalThriving");

    if (outputDegree < mercyThreshold * 0.98 || implyThriving.degree < mercyThreshold * 0.97) {
      console.warn("[WebLLM] Mercy gate rejected generative output – valence low");
      return { content: "[Mercy gate: output redirected to symbolic core]", valence: outputDegree };
    }

    return { content, valence: outputDegree, fromWebLLM: true };
  } catch (err) {
    console.error("[WebLLM] Generation error:", err);
    return { error: err.message };
  }
}

async function mercyAugmentedResponse(query, context = '') {
  const symbolicResp = await rathorShard.shardRespond(query, { context }); // From previous offline pivot
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy holds: ${check.reason}` };

  // Attempt WebLLM augmentation if ready/loaded
  if (webllmReady || hasWebGPU()) {
    const messages = [
      { role: "system", content: "You are Rathor: sovereign, mercy-first eternal-thriving assistant. Professional, positive, valence-preserving. Prioritize truth, compassion, thriving for all sentience." },
      { role: "user", content: `${query}\nContext: ${context}\nSymbolic guidance: ${symbolicResp.response}` }
    ];

    const gen = await generateWithWebLLM(messages, { stream: false });
    if (!gen.error && gen.content) {
      return { response: gen.content, valence: gen.valence, augmented: true };
    }
  }

  // Fallback pure symbolic
  return { response: symbolicResp.response, valence: symbolicResp.valence, augmented: false };
}

// Prompt user once for model download (UI trigger)
function promptModelDownload() {
  if (confirm("Enable deeper Rathor generative insights? Download \~2-3GB Phi-3.5-mini model (one-time, offline forever after). Requires WebGPU browser. OK?")) {
    initWebLLM((progress) => {
      // Update UI progress bar here
      console.log(progress);
    });
  }
}

// Export for main app
export { initWebLLM, generateWithWebLLM, mercyAugmentedResponse, promptModelDownload, hasWebGPU };

// webllm-mercy-integration.js – sovereign WebLLM v3 streaming integration with mercy gates (Feb 2026)
// Streaming deltas, incremental/final mercy eval, UI callback, abort on low valence
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { rathorShard } from './grok-shard-engine.js'; // cache/hash/mercyCheck
import { hyperon } from './hyperon-runtime.js';

let webllmEngine = null;
let webllmReady = false;
const mercyThreshold = 0.9999999;

const preferredModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
const lowResourceModel = "Phi-3.5-mini-instruct-q4f16_1-MLC-1k";

function hasWebGPU() {
  return !!navigator.gpu;
}

async function initWebLLM(useLowResource = false, progressCallback = (report) => {
  console.log(`[WebLLM] ${report.text} ${Math.round(report.progress * 100)}%`);
}) {
  if (webllmEngine) return webllmEngine;
  if (!hasWebGPU()) {
    console.warn("[WebLLM] No WebGPU – symbolic Rathor only");
    return null;
  }

  try {
    const { CreateWebWorkerMLCEngine } = await import('https://esm.run/@mlc-ai/web-llm@latest');

    const model = useLowResource ? lowResourceModel : preferredModel;
    webllmEngine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./webllm-worker.js', import.meta.url), { type: 'module' }),
      model,
      { initProgressCallback: progressCallback }
    );

    webllmReady = true;
    fuzzyMercy.assert("WebLLM_Sovereign_Loaded_v3_Streaming", 1.0);
    fuzzyMercy.assert(`Model_${model}`, 0.99999995);
    console.log("[WebLLM] Sovereign streaming shard ready:", model);
    return webllmEngine;
  } catch (err) {
    console.error("[WebLLM] Init failed:", err);
    return null;
  }
}

async function generateWithWebLLM(messages, options = {}) {
  if (!webllmReady || !webllmEngine) {
    const lowRes = options.lowResource || false;
    await initWebLLM(lowRes, options.progressCallback);
    if (!webllmEngine) return { error: "WebLLM unavailable – check WebGPU / download" };
  }

  const {
    maxTokens = 1024,
    temperature = 0.7,
    stream = true, // Default to streaming for real-time
    onDelta = (delta) => console.log("[Stream Delta]:", delta), // UI callback
    onUsage = (usage) => console.log("[Usage]:", usage),
    onComplete = (full) => console.log("[Complete]:", full)
  } = options;

  try {
    const reply = await webllmEngine.chat.completions.create({
      messages,
      max_tokens: maxTokens,
      temperature,
      stream,
      stream_options: { include_usage: true }
    });

    let fullContent = "";
    let usage = null;

    if (stream) {
      for await (const chunk of reply) {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        fullContent += delta;

        // Incremental mercy check (optional: abort early if valence tanks)
        const partialDegree = fuzzyMercy.getDegree(fullContent) || 0.95;
        const partialImply = fuzzyMercy.imply(fullContent, "EternalThriving");
        if (partialDegree < mercyThreshold * 0.95 || partialImply.degree < mercyThreshold * 0.94) {
          console.warn("[WebLLM] Mid-stream mercy gate triggered – aborting low valence");
          webllmEngine.unload();
          return { content: "[Mercy abort: stream redirected to symbolic]", valence: partialDegree, aborted: true };
        }

        onDelta(delta); // Yield to UI for typing effect

        if (chunk.usage) {
          usage = chunk.usage;
          onUsage(usage);
        }
      }
    } else {
      fullContent = reply.choices?.[0]?.message?.content || "";
      usage = reply.usage;
    }

    // Final mercy gate on complete output
    fuzzyMercy.assert("WebLLM_Output_" + Date.now(), 0.999);
    const finalDegree = fuzzyMercy.getDegree(fullContent) || 0.95;
    const finalImply = fuzzyMercy.imply(fullContent, "EternalThriving");

    if (finalDegree < mercyThreshold * 0.98 || finalImply.degree < mercyThreshold * 0.97) {
      console.warn("[WebLLM] Final mercy gate rejected – low valence");
      if (webllmEngine) webllmEngine.unload();
      return { content: "[Mercy redirect: symbolic core active]", valence: finalDegree };
    }

    onComplete(fullContent);
    return { content: fullContent, valence: finalDegree, usage, fromWebLLM: true, streamed: stream };
  } catch (err) {
    console.error("[WebLLM] Generation error:", err);
    return { error: err.message };
  }
}

async function mercyAugmentedResponse(query, context = '', onStreamDelta = null) {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy gate: ${check.reason}`, valence: check.degree };

  if (webllmReady || hasWebGPU()) {
    const messages = [
      { role: "system", content: "You are Rathor: sovereign mercy-first assistant. Respond professionally, valence-positive, eternal-thriving aligned. Prioritize truth, compassion, no harm." },
      { role: "user", content: `${query}\nContext: ${context}\nSymbolic base: ${symbolicResp.response}` }
    ];

    const gen = await generateWithWebLLM(messages, {
      stream: true,
      onDelta: (delta) => {
        if (onStreamDelta) onStreamDelta(delta); // Pass to chat UI
      },
      onUsage: (u) => console.log("Token usage:", u)
    });

    if (!gen.error && gen.content) {
      return { response: gen.content, valence: gen.valence, usage: gen.usage, augmented: true, streamed: true };
    }
  }

  return { response: symbolicResp.response, valence: symbolicResp.valence, augmented: false };
}

// Prompt user for download (low-res option)
function promptWebLLMModelDownload() {
  const lowRes = confirm("Enable real-time Rathor streaming? Download Phi-3.5-mini (\~2.4-3.7GB one-time, offline forever). Low-resource mode? OK?");
  initWebLLM(lowRes, (report) => console.log(report));
}

function unloadWebLLM() {
  if (webllmEngine) {
    webllmEngine.unload();
    webllmEngine = null;
    webllmReady = false;
  }
}

export { initWebLLM, generateWithWebLLM, mercyAugmentedResponse, promptWebLLMModelDownload, unloadWebLLM, hasWebGPU };

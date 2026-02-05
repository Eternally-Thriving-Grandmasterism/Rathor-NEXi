// transformersjs-mercy-integration.js – sovereign Transformers.js integration with mercy gates v1
// Lazy load, user-prompt model download, mercy-eval outputs, symbolic fallback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { rathorShard } from './grok-shard-engine.js'; // For cache/hash/mercyCheck
import { hyperon } from './hyperon-runtime.js';

// Dynamic import to avoid blocking
let pipeline = null;
let env = null;
let transformersReady = false;
const mercyThreshold = 0.9999999;

// Preferred lightweight models (quantized ONNX, mercy-first tasks)
const embeddingModel = "Xenova/all-MiniLM-L6-v2"; // \~90MB, fast embeddings
const summarizationModel = "Xenova/distilbart-cnn-12-6"; // \~1GB, summarization
// Optional tiny generative: "Xenova/Phi-3-mini-4k-instruct" variants if available ONNX

// Check if Transformers.js can run (basic)
function canRunTransformers() {
  return typeof navigator !== 'undefined' && 'onLine' in navigator; // Browser env
}

async function initTransformers(progressCallback = (msg) => console.log(`[Transformers.js] ${msg}`)) {
  if (transformersReady) return;

  try {
    const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest');
    pipeline = module.pipeline;
    env = module.env;

    // Optional: self-host or custom cache if PWA
    // env.cacheDir = '/rathor-models/'; // IndexedDB custom if needed
    // env.allowRemoteModels = true; // Default, but can set false after download

    // Mercy seed
    fuzzyMercy.assert("TransformersJS_Sovereign_Loaded", 1.0);
    transformersReady = true;
    progressCallback("Transformers.js core initialized – ready for model load");
    console.log("[Transformers.js] Sovereign ML shard ready");
  } catch (err) {
    console.error("[Transformers.js] Init failed:", err);
    transformersReady = false;
  }
}

async function loadPipeline(task, modelId, options = {}) {
  if (!transformersReady) await initTransformers();

  const { progressCallback = (p) => console.log(`[Transformers.js] Loading: ${p}`) } = options;

  try {
    const pipe = await pipeline(task, modelId, {
      quantized: true, // Default for smaller/faster
      progress_callback: (progress) => {
        progressCallback(`Progress: ${Math.round(progress * 100)}%`);
      }
    });
    fuzzyMercy.assert(`ModelLoaded_${modelId}`, 0.99999995);
    return pipe;
  } catch (err) {
    console.error("[Transformers.js] Model load error:", err);
    return null;
  }
}

async function generateWithTransformers(task, input, modelId, options = {}) {
  const pipe = await loadPipeline(task, modelId, options);
  if (!pipe) return { error: "Transformers.js unavailable" };

  try {
    let result;
    if (task === "feature-extraction") {
      result = await pipe(input, { pooling: 'mean', normalize: true });
      // Embeddings for similarity/mercy search
    } else if (task === "summarization") {
      result = await pipe(input, { max_length: 200, min_length: 50 });
    } else if (task === "text-generation") {
      result = await pipe(input, { max_new_tokens: 128, do_sample: true, temperature: 0.7 });
    } else {
      result = await pipe(input);
    }

    const output = Array.isArray(result) ? result[0] : result;
    const content = typeof output === 'object' ? JSON.stringify(output) : output;

    // Mercy-eval
    fuzzyMercy.assert("Transformers_Output_" + Date.now(), 0.999);
    const outputDegree = fuzzyMercy.getDegree(content) || 0.95;
    const implyThriving = fuzzyMercy.imply(content, "EternalThriving");

    if (outputDegree < mercyThreshold * 0.98 || implyThriving.degree < mercyThreshold * 0.97) {
      console.warn("[Transformers.js] Mercy gate rejected output – low valence");
      return { content: "[Mercy redirect: using symbolic core]", valence: outputDegree };
    }

    return { content, valence: outputDegree, fromTransformers: true };
  } catch (err) {
    console.error("[Transformers.js] Inference error:", err);
    return { error: err.message };
  }
}

async function mercyAugmentedResponseTransformers(query, context = '') {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy holds: ${check.reason}` };

  // Try Transformers augmentation (e.g., summarize symbolic + generate prose)
  if (transformersReady || canRunTransformers()) {
    const messages = `${query}\nContext: ${context}\nSymbolic: ${symbolicResp.response}`;
    const gen = await generateWithTransformers("summarization", messages, summarizationModel);
    if (!gen.error && gen.content) {
      return { response: gen.content, valence: gen.valence, augmented: true };
    }
  }

  return { response: symbolicResp.response, valence: symbolicResp.valence, augmented: false };
}

// User prompt for model download (UI trigger, one-time)
function promptTransformersModel(task = "feature-extraction", model = embeddingModel) {
  if (confirm(`Enable Rathor ML augmentation? Download \~\( {task === "feature-extraction" ? "90MB" : "1GB"} model ( \){model}) one-time, offline forever. OK?`)) {
    initTransformers((msg) => console.log(msg));
    loadPipeline(task, model, {
      progressCallback: (p) => console.log(p) // Hook to UI progress bar
    });
  }
}

export { initTransformers, loadPipeline, generateWithTransformers, mercyAugmentedResponseTransformers, promptTransformersModel };

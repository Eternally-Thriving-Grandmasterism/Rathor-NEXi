// scripts/browser_inference_demo.js – Browser-side ONNX Inference Demo v1
// WebNN → WebGPU → WebGL fallback, valence-aware model selection
// Run via: node -e "require('./browser_inference_demo.js').runDemo()"
// or include in RathorChat for live testing

const ort = require('onnxruntime-web');

const MODELS = {
  int4: '/models/gesture-transformer-qat-int4/model.onnx',
  int8: '/models/gesture-transformer-qat-int8/model.onnx',
  fp16: '/models/gesture-transformer-onnx/model.onnx'
};

async function runDemo() {
  console.log("[BrowserInference] Starting demo...");

  const valence = 0.96; // simulate currentValence.get()
  let modelUrl = MODELS.fp16;

  if (valence > 0.94) modelUrl = MODELS.int4;
  else if (valence > 0.88) modelUrl = MODELS.int8;

  console.log(`[BrowserInference] Selected model: ${modelUrl} (valence ${valence})`);

  const providers = [];
  const hasWebNN = 'ml' in navigator && 'createContext' in navigator.ml;
  if (hasWebNN) providers.push('webnn');
  providers.push('webgpu', 'webgl');

  console.log("[BrowserInference] Trying providers:", providers);

  let session = null;
  for (const provider of providers) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all'
      });
      console.log(`[BrowserInference] Success with provider: ${provider}`);
      break;
    } catch (err) {
      console.warn(`Provider ${provider} failed:`, err.message);
    }
  }

  if (!session) {
    console.error("[BrowserInference] All providers failed");
    return;
  }

  // Dummy input [1, 45, 225]
  const inputData = new Float32Array(1 * 45 * 225).fill(0.1);
  const inputTensor = new ort.Tensor('float32', inputData, [1, 45, 225]);

  console.time("[BrowserInference] Inference");
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  console.timeEnd("[BrowserInference] Inference");

  console.log("[BrowserInference] Output keys:", Object.keys(results));
  console.log("[BrowserInference] Gesture logits sample:", results.gesture?.data.slice(0, 4));
}

if (typeof window !== 'undefined') {
  // Browser usage example
  window.runRathorInferenceDemo = runDemo;
} else {
  // Node.js test
  runDemo().catch(console.error);
}

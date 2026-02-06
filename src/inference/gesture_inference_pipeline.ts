// src/inference/gesture_inference_pipeline.ts – Gesture Inference Pipeline v1.1
// ONNX Runtime Web with model selection logic (INT4/INT8/FP16 by valence)
// Provider preference: WebNN > WebGPU > WebGL > CPU, dynamic batch, latency metrics
// MIT License – Autonomicity Games Inc. 2026

import * as ort from 'onnxruntime-web';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MODEL_PATHS = {
  high: '/models/gesture-transformer-qat-int4.onnx',    // INT4 – fastest, high valence only
  medium: '/models/gesture-transformer-qat-int8.onnx',  // INT8 – balanced, safe default
  low: '/models/gesture-transformer-fp16.onnx',         // FP16 – highest accuracy fallback
};

const VALENCE_HIGH_THRESHOLD = 0.94;     // Use INT4 + prefer WebNN
const VALENCE_MEDIUM_THRESHOLD = 0.88;   // Use INT8 + WebGPU/WebNN
const VALENCE_LOW_THRESHOLD = 0.80;      // Use FP16 + CPU fallback

const SEQUENCE_LENGTH = 45;
const LANDMARK_DIM = 225;
const NUM_GESTURE_CLASSES = 5;
const FUTURE_VALENCE_HORIZON = 10;
const CONFIDENCE_THRESHOLD = 0.75;
const MERCY_VALENCE_DROP_THRESHOLD = 0.05;

let currentSession: ort.InferenceSession | null = null;
let currentModelKey: string = 'none';
let currentProvider: string = 'none';

export class GestureInferencePipeline {
  static async initialize(forceReload = false): Promise<void> {
    const actionName = 'Initialize gesture inference pipeline with model selection';
    if (!await mercyGate(actionName)) return;

    const valence = currentValence.get();
    let targetModelKey = 'low';

    if (valence > VALENCE_HIGH_THRESHOLD) {
      targetModelKey = 'high';
    } else if (valence > VALENCE_MEDIUM_THRESHOLD) {
      targetModelKey = 'medium';
    }

    if (!forceReload && currentSession && currentModelKey === targetModelKey) {
      console.log(`[InferencePipeline] Already initialized – model: ${currentModelKey}, provider: ${currentProvider}`);
      return;
    }

    console.log(`[InferencePipeline] Initializing – valence ${valence.toFixed(3)}, selecting model: ${targetModelKey}`);

    currentModelKey = targetModelKey;
    const modelPath = MODEL_PATHS[currentModelKey as keyof typeof MODEL_PATHS];

    const providers = [];
    if ('ml' in navigator && 'createContext' in (navigator as any).ml) {
      providers.push('webnn');
    }
    providers.push('webgpu', 'webgl', 'cpu');

    console.log("[InferencePipeline] Trying providers in order:", providers);

    for (const provider of providers) {
      try {
        currentSession = await ort.InferenceSession.create(modelPath, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: false,
        });

        currentProvider = provider;
        break;
      } catch (err) {
        console.warn(`Provider ${provider} failed for model ${currentModelKey}:`, err);
      }
    }

    if (!currentSession) {
      throw new Error("No suitable execution provider found for selected model");
    }

    // Warm-up inference
    const dummyInput = new ort.Tensor(
      'float32',
      new Float32Array(1 * SEQUENCE_LENGTH * LANDMARK_DIM),
      [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    );

    await currentSession.run({ input: dummyInput });
    dummyInput.dispose();

    mercyHaptic.playPattern('cosmicHarmony', valence);
    console.log(`[InferencePipeline] Initialized – model: ${currentModelKey}, provider: ${currentProvider}`);
  }

  static async infer(landmarks: Float32Array): Promise<{
    gesture: string;
    confidence: number;
    futureValence: number[];
    projectedValence: number;
    isSafe: boolean;
    modelUsed: string;
    providerUsed: string;
  }> {
    if (!currentSession) {
      await this.initialize();
    }

    const actionName = 'Run gesture inference';
    if (!await mercyGate(actionName)) {
      return {
        gesture: 'none',
        confidence: 0,
        futureValence: Array(FUTURE_VALENCE_HORIZON).fill(0.5),
        projectedValence: currentValence.get(),
        isSafe: false,
        modelUsed: 'none',
        providerUsed: 'none',
      };
    }

    const tensor = new ort.Tensor(
      'float32',
      landmarks,
      [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    );

    const feeds = { input: tensor };
    const results = await currentSession.run(feeds);

    const gestureLogits = results.gesture_logits.data as Float32Array;
    const futureValence = results.future_valence.data as Float32Array;

    // Softmax on logits
    const maxLogit = Math.max(...gestureLogits);
    const expLogits = gestureLogits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);

    const confidence = Math.max(...probs);
    const gestureIndex = probs.indexOf(confidence);
    const gesture = confidence > CONFIDENCE_THRESHOLD ? GESTURE_NAMES[gestureIndex] : 'none';

    const projectedValence = futureValence.reduce((a, b) => a + b, 0) / futureValence.length;
    const currentVal = currentValence.get();
    const isSafe = projectedValence >= currentVal - MERCY_VALENCE_DROP_THRESHOLD;

    if (!isSafe) {
      mercyHaptic.playPattern('warningPulse', currentVal * 0.7);
    }

    tensor.dispose();

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValence),
      projectedValence,
      isSafe,
      modelUsed: currentModelKey,
      providerUsed: currentProvider,
    };
  }

  static async dispose() {
    if (currentSession) {
      await currentSession.release();
      currentSession = null;
      currentModelKey = 'none';
      currentProvider = 'none';
      console.log("[InferencePipeline] Session disposed");
    }
  }

  static getCurrentModel(): string {
    return currentModelKey;
  }

  static getCurrentProvider(): string {
    return currentProvider;
  }
}

export default GestureInferencePipeline;

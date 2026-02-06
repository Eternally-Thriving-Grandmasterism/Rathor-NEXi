// src/inference/gesture_inference_pipeline.ts – Gesture Inference Pipeline v1.0
// ONNX Runtime Web with WebNN → WebGPU → WebGL fallback
// Valence-aware model selection, mercy gating, future valence trajectory
// MIT License – Autonomicity Games Inc. 2026

import * as ort from 'onnxruntime-web';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MODEL_PATHS = {
  high: '/models/gesture-transformer-qat-int4.onnx',    // INT4 for high valence (fastest)
  medium: '/models/gesture-transformer-qat-int8.onnx',  // INT8 balanced
  low: '/models/gesture-transformer-fp16.onnx',         // FP16 fallback
};

const SEQUENCE_LENGTH = 45;
const LANDMARK_DIM = 225;
const NUM_GESTURE_CLASSES = 5;
const FUTURE_VALENCE_HORIZON = 10;

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_HIGH_PIVOT = 0.94;     // INT4 + WebNN
const VALENCE_SAFE_PIVOT = 0.88;     // INT8 + WebGPU/WebNN
const CONFIDENCE_THRESHOLD = 0.75;

let session: ort.InferenceSession | null = null;
let currentProvider = 'none';

export class GestureInferencePipeline {
  static async initialize(): Promise<void> {
    const actionName = 'Initialize gesture inference pipeline';
    if (!await mercyGate(actionName)) return;

    if (session) {
      console.log("[InferencePipeline] Already initialized – provider:", currentProvider);
      return;
    }

    console.log("[InferencePipeline] Initializing ONNX Runtime Web...");

    const valence = currentValence.get();
    let modelPath = MODEL_PATHS.low;

    if (valence > VALENCE_HIGH_PIVOT) {
      modelPath = MODEL_PATHS.high;
    } else if (valence > VALENCE_SAFE_PIVOT) {
      modelPath = MODEL_PATHS.medium;
    }

    const providers = [];
    if ('ml' in navigator && 'createContext' in (navigator as any).ml) {
      providers.push('webnn');
    }
    providers.push('webgpu', 'webgl');

    console.log("[InferencePipeline] Trying providers:", providers);

    for (const provider of providers) {
      try {
        session = await ort.InferenceSession.create(modelPath, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: false,
        });

        currentProvider = provider;
        break;
      } catch (err) {
        console.warn(`Provider ${provider} failed:`, err);
      }
    }

    if (!session) {
      throw new Error("No suitable execution provider found");
    }

    // Warm-up
    const dummyInput = new ort.Tensor(
      'float32',
      new Float32Array(1 * SEQUENCE_LENGTH * LANDMARK_DIM),
      [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    );

    await session.run({ input: dummyInput });
    dummyInput.dispose();

    mercyHaptic.playPattern('cosmicHarmony', valence);
    console.log(`[InferencePipeline] Loaded model: ${modelPath} | Provider: ${currentProvider}`);
  }

  static async infer(landmarks: Float32Array): Promise<{
    gesture: string;
    confidence: number;
    futureValence: number[];
    projectedValence: number;
    isSafe: boolean;
  }> {
    if (!session) {
      throw new Error("Inference pipeline not initialized");
    }

    const actionName = 'Run gesture inference';
    if (!await mercyGate(actionName)) {
      return {
        gesture: 'none',
        confidence: 0,
        futureValence: Array(FUTURE_VALENCE_HORIZON).fill(0.5),
        projectedValence: currentValence.get(),
        isSafe: false,
      };
    }

    const tensor = new ort.Tensor(
      'float32',
      landmarks,
      [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    );

    const feeds = { input: tensor };
    const results = await session.run(feeds);

    const gestureLogits = results.gesture_logits.data as Float32Array;
    const futureValence = results.future_valence.data as Float32Array;

    // Softmax on logits
    const maxLogit = Math.max(...gestureLogits);
    const expLogits = gestureLogits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);

    const confidence = Math.max(...probs);
    const gestureIndex = probs.indexOf(confidence);
    const gestureNames = ['none', 'pinch', 'spiral', 'figure8', 'wave'];

    const gesture = confidence > 0.75 ? gestureNames[gestureIndex] : 'none';

    const projectedValence = futureValence.reduce((a, b) => a + b, 0) / futureValence.length;
    const isSafe = projectedValence >= currentValence.get() - 0.05;

    if (!isSafe) {
      mercyHaptic.playPattern('warningPulse', currentValence.get() * 0.7);
    }

    tensor.dispose();

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValence),
      projectedValence,
      isSafe,
    };
  }

  static async dispose() {
    if (session) {
      await session.release();
      session = null;
      currentProvider = 'none';
      console.log("[InferencePipeline] Session disposed");
    }
  }

  static getCurrentProvider(): string {
    return currentProvider;
  }
}

export default GestureInferencePipeline;

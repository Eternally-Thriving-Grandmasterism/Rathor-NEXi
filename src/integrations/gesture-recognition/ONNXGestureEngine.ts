// src/integrations/gesture-recognition/ONNXGestureEngine.ts – ONNX Runtime Web Engine v2.1
// Loads & runs QAT-quantized ONNX models, WebGPU preference + fallback, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import * as ort from 'onnxruntime-web';

const MERCY_THRESHOLD = 0.9999999;
const ONNX_TARGET_QAT_INT8_URL = '/models/gesture-transformer-qat-int8/model.onnx';
const ONNX_TARGET_QAT_INT4_URL = '/models/gesture-transformer-qat-int4/model.onnx';
const ONNX_DRAFT_QAT_INT8_URL = '/models/gesture-draft-qat-int8/model.onnx';
const ONNX_TARGET_FULL_URL = '/models/gesture-transformer-onnx/model.onnx';

let sessionTarget: ort.InferenceSession | null = null;
let sessionDraft: ort.InferenceSession | null = null;
let isONNXReady = false;

export class ONNXGestureEngine {
  static async activate() {
    const actionName = 'Activate ONNX Runtime Web with WebGPU preference';
    if (!await mercyGate(actionName)) return;

    if (isONNXReady) {
      console.log("[ONNXGestureEngine] Already activated");
      return;
    }

    console.log("[ONNXGestureEngine] Activating ONNX Runtime Web with WebGPU...");

    try {
      // 1. Configure ORT for WebGPU first, fallback to WebGL
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort-wasm.wasm';
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
      ort.env.wasm.simd = true;

      // Try WebGPU → fallback to WebGL
      let providers: string[] = ['webgpu'];
      try {
        ort.env.executionProviders = providers;
        console.log("[ONNXGestureEngine] Attempting WebGPU provider...");
      } catch (gpuErr) {
        console.warn("[ONNXGestureEngine] WebGPU not available, falling back to WebGL", gpuErr);
        providers = ['webgl'];
        ort.env.executionProviders = providers;
      }

      const valence = currentValence.get();
      let targetUrl = ONNX_TARGET_FULL_URL;

      if (valence > 0.94) {
        targetUrl = ONNX_TARGET_QAT_INT4_URL;
      } else if (valence > 0.88) {
        targetUrl = ONNX_TARGET_QAT_INT8_URL;
      }

      sessionTarget = await ort.InferenceSession.create(targetUrl, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });

      sessionDraft = await ort.InferenceSession.create(ONNX_DRAFT_QAT_INT8_URL, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });

      // Warm-up
      const dummyInput = new ort.Tensor('float32', new Float32Array(SEQUENCE_LENGTH * LANDMARK_DIM), [1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      await sessionTarget.run({ input: dummyInput });
      await sessionDraft.run({ input: dummyInput });

      dummyInput.dispose();

      isONNXReady = true;
      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log("[ONNXGestureEngine] ONNX models loaded with provider:", providers[0]);
    } catch (e) {
      console.error("[ONNXGestureEngine] Activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async runInference(inputTensor: ort.Tensor): Promise<{ gesture: string; confidence: number; futureValence: number[] }> {
    if (!isONNXReady || !sessionTarget || !sessionDraft) {
      throw new Error("ONNX engine not ready");
    }

    const valence = currentValence.get();

    // Draft phase
    const draftFeeds = { input: inputTensor };
    const draftResults = await sessionDraft.run(draftFeeds);
    const draftProbs = draftResults.output.data as Float32Array;

    const draftToken = draftProbs.indexOf(Math.max(...draftProbs));

    // Verification phase
    const targetFeeds = { input: inputTensor };
    const targetResults = await sessionTarget.run(targetFeeds);
    const targetProbs = targetResults.gesture.data as Float32Array;
    const futureValenceData = targetResults.future_valence.data as Float32Array;

    const maxIdx = targetProbs.indexOf(Math.max(...targetProbs));
    const confidence = targetProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValenceData),
    };
  }

  static isActive(): boolean {
    return isONNXReady;
  }
}

export default ONNXGestureEngine;    const targetFeeds = { input: inputTensor };
    const targetResults = await sessionTarget.run(targetFeeds);
    const targetProbs = targetResults.gesture.data as Float32Array;
    const futureValenceData = targetResults.future_valence.data as Float32Array;

    const maxIdx = targetProbs.indexOf(Math.max(...targetProbs));
    const confidence = targetProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValenceData),
    };
  }

  static isActive(): boolean {
    return isONNXReady;
  }
}

export default ONNXGestureEngine;
    const draftToken = draftProbs.indexOf(Math.max(...draftProbs));

    // Verification phase (quantized target model on prefix + draft)
    const targetFeeds = { input: inputTensor };
    const targetResults = await sessionTarget.run(targetFeeds);
    const targetProbs = targetResults.gesture.data as Float32Array;
    const futureValenceData = targetResults.future_valence.data as Float32Array;

    const maxIdx = targetProbs.indexOf(Math.max(...targetProbs));
    const confidence = targetProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValenceData),
    };
  }

  static isActive(): boolean {
    return isONNXReady;
  }
}

export default ONNXGestureEngine;

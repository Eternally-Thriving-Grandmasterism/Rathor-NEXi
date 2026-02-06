// src/integrations/gesture-recognition/QuantizedGestureModel.ts – Quantized Custom Transformer Loader v3
// 2-bit BNN extreme preference, layered fallback (2→4→8→FP16), mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const BNN_2BIT_URL = '/models/gesture-transformer-2bit-bnn/model.json';      // binary weights + activations
const QUANTIZED_4BIT_URL = '/models/gesture-transformer-4bit-awq/model.json';
const QUANTIZED_8BIT_URL = '/models/gesture-transformer-8bit-int8/model.json';
const FULL_FP16_URL     = '/models/gesture-transformer-full/model.json';

let modelPromise: Promise<tf.LayersModel> | null = null;

export class QuantizedGestureModel {
  static async load(): Promise<tf.LayersModel> {
    const actionName = 'Load 2-bit BNN quantized custom transformer model';
    if (!await mercyGate(actionName)) {
      throw new Error("Mercy gate blocked model loading");
    }

    if (modelPromise) return modelPromise;

    const valence = currentValence.get();
    let selectedUrl = FULL_FP16_URL;

    if (valence > 0.96) {
      // Very high valence → prefer 2-bit BNN (extreme speed + thriving-aligned)
      selectedUrl = BNN_2BIT_URL;
    } else if (valence > 0.90) {
      // High valence → 4-bit AWQ
      selectedUrl = QUANTIZED_4BIT_URL;
    } else if (valence > 0.82) {
      // Medium valence → 8-bit int8
      selectedUrl = QUANTIZED_8BIT_URL;
    } else {
      // Low valence → full FP16 (maximum accuracy for survival)
      selectedUrl = FULL_FP16_URL;
    }

    console.log(`[QuantizedGestureModel] Loading ${selectedUrl} for valence ${valence.toFixed(4)}`);

    try {
      modelPromise = tf.loadLayersModel(selectedUrl);
      const model = await modelPromise;

      // Warm-up inference
      const dummyInput = tf.zeros([1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      const dummyOutput = model.predict(dummyInput) as tf.Tensor[];
      dummyOutput.forEach(t => t.dispose());
      dummyInput.dispose();

      console.log("[QuantizedGestureModel] Model loaded & warmed up successfully");
      return model;
    } catch (e) {
      console.error("[QuantizedGestureModel] Load failed", e);
      // Fallback chain
      const fallbackUrls = [QUANTIZED_4BIT_URL, QUANTIZED_8BIT_URL, FULL_FP16_URL];
      for (const url of fallbackUrls) {
        try {
          modelPromise = tf.loadLayersModel(url);
          return await modelPromise;
        } catch {}
      }
      throw new Error("All model loading attempts failed");
    }
  }

  static async dispose() {
    if (modelPromise) {
      const model = await modelPromise;
      model.dispose();
      modelPromise = null;
      console.log("[QuantizedGestureModel] Model disposed");
    }
  }
}

export default QuantizedGestureModel;

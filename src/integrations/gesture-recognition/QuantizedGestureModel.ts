// src/integrations/gesture-recognition/QuantizedGestureModel.ts – Quantized Model Loader v7
// QAT preference chain (INT8/INT4/ternary), valence gating, mercy fallback
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const QAT_INT8_URL = '/models/gesture-transformer-qat-int8/model.json';
const QAT_INT4_URL = '/models/gesture-transformer-qat-int4/model.json';
const TERNARY_QAT_URL = '/models/gesture-transformer-ternary-qat/model.json';
const QUANTIZED_8BIT_URL = '/models/gesture-transformer-8bit-int8/model.json';
const FULL_FP16_URL     = '/models/gesture-transformer-full/model.json';

let modelPromise: Promise<tf.LayersModel> | null = null;

export class QuantizedGestureModel {
  static async load(): Promise<tf.LayersModel> {
    const actionName = 'Load QAT-quantized custom transformer model';
    if (!await mercyGate(actionName)) {
      throw new Error("Mercy gate blocked model loading");
    }

    if (modelPromise) return modelPromise;

    const valence = currentValence.get();
    let selectedUrl = FULL_FP16_URL;

    if (valence > 0.96) {
      selectedUrl = TERNARY_QAT_URL;
    } else if (valence > 0.92) {
      selectedUrl = QAT_INT4_URL;
    } else if (valence > 0.88) {
      selectedUrl = QAT_INT8_URL;
    } else if (valence > 0.82) {
      selectedUrl = QUANTIZED_8BIT_URL;
    }

    console.log(`[QuantizedGestureModel] Loading QAT model ${selectedUrl} for valence ${valence.toFixed(4)}`);

    try {
      modelPromise = tf.loadLayersModel(selectedUrl);
      const model = await modelPromise;

      // Warm-up
      const dummyInput = tf.zeros([1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      const dummyOutput = model.predict(dummyInput) as tf.Tensor[];
      dummyOutput.forEach(t => t.dispose());
      dummyInput.dispose();

      console.log("[QuantizedGestureModel] QAT model loaded & warmed up");
      return model;
    } catch (e) {
      console.error("[QuantizedGestureModel] Load failed", e);
      const fallbackUrls = [QAT_INT8_URL, QUANTIZED_8BIT_URL, FULL_FP16_URL];
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

// src/integrations/gesture-recognition/GestureEngineLazyLoader.ts – tfjs + BlazePose Lazy Loader v1
// Deferred import, model loading, backend init — only on explicit activation
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const LAZY_ACTIVATION_VALENCE = 0.85; // auto-activate above this valence if user intent detected

let isLoaded = false;
let holisticPromise: Promise<any> | null = null;
let tfPromise: Promise<typeof import('@tensorflow/tfjs')> | null = null;

export class GestureEngineLazyLoader {
  static async activate(onReady?: () => void) {
    const actionName = 'Lazy-load tfjs & BlazePose engine';
    if (!await mercyGate(actionName)) return;

    if (isLoaded) {
      console.log("[GestureLazyLoader] Already activated");
      onReady?.();
      return;
    }

    console.log("[GestureLazyLoader] Activating tfjs + BlazePose (first load)...");

    try {
      // 1. Load tfjs core + webgl backend (parallel)
      tfPromise = import('@tensorflow/tfjs').then(async tf => {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log("[GestureLazyLoader] tfjs backend ready:", tf.getBackend());
        return tf;
      });

      // 2. Load @mediapipe/holistic (WASM heavy)
      holisticPromise = import('@mediapipe/holistic').then(async ({ Holistic }) => {
        const holistic = new Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });

        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.7
        });

        await holistic.initialize();
        console.log("[GestureLazyLoader] BlazePose Holistic initialized");
        return holistic;
      });

      // 3. Await both
      const [tf, holistic] = await Promise.all([tfPromise, holisticPromise]);

      isLoaded = true;
      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log("[GestureLazyLoader] Activation complete – ready for inference");

      onReady?.();
    } catch (e) {
      console.error("[GestureLazyLoader] Activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static getHolistic(): Promise<any> | null {
    return holisticPromise;
  }

  static isActive(): boolean {
    return isLoaded;
  }

  /**
   * Auto-activation on high valence or user intent
   */
  static tryAutoActivate() {
    if (currentValence.get() > LAZY_ACTIVATION_VALENCE && !isLoaded) {
      console.log("[GestureLazyLoader] Auto-activation triggered by high valence");
      this.activate();
    }
  }
}

// Auto-check on valence change (optional background warm-up)
currentValence.subscribe(() => {
  GestureEngineLazyLoader.tryAutoActivate();
});

// Export for use in GestureOverlay / MR components
export default GestureEngineLazyLoader;

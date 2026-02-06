// src/integrations/gesture-recognition/SpatiotemporalTransformerGestures.ts – Spatiotemporal Transformer Gesture Engine v1.17
// BlazePose → Quantized Encoder-Decoder (2-bit preference) + Speculative Decoding → gesture + future valence
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { Holistic } from '@mediapipe/holistic';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { ydoc } from '@/sync/multiplanetary-sync-engine';
import { wootPrecedenceGraph } from '@/sync/woot-precedence-graph';
import QuantizedGestureModel from './QuantizedGestureModel';

const MERCY_THRESHOLD = 0.9999999;
const SEQUENCE_LENGTH = 45;
const LANDMARK_DIM = 33 * 3 + 21 * 3 * 2;
const SPECULATIVE_DRAFT_STEPS = 6;
const SPECULATIVE_ACCEPT_THRESHOLD = 0.9;

export class SpatiotemporalTransformerGestures {
  private holistic: Holistic | null = null;
  private sequenceBuffer: tf.Tensor3D[] = [];
  private ySequence: Y.Array<any>;
  private model: tf.LayersModel | null = null;

  constructor() {
    this.ySequence = ydoc.getArray('gesture-sequence');
    this.initialize();
  }

  private async initialize() {
    if (!await mercyGate('Initialize Quantized Transformer Engine')) return;

    try {
      // 1. BlazePose Holistic (lazy-loaded separately)
      this.holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      this.holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
      });

      await this.holistic.initialize();

      // 2. Load quantized custom transformer (2-bit preference)
      this.model = await QuantizedGestureModel.load();

      console.log("[SpatiotemporalTransformer] Quantized model (2-bit preferred) + BlazePose initialized");
    } catch (e) {
      console.error("[SpatiotemporalTransformer] Initialization failed", e);
    }
  }

  async processFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic || !this.model || !await mercyGate('Process quantized frame')) return null;

    const results = await this.holistic.send({ image: videoElement });

    if (!results.poseLandmarks && !results.leftHandLandmarks && !results.rightHandLandmarks) return null;

    const frameVector = this.flattenLandmarks(
      results.poseLandmarks || [],
      results.leftHandLandmarks || [],
      results.rightHandLandmarks || []
    );

    this.sequenceBuffer.push(tf.tensor1d(frameVector));
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH) {
      this.sequenceBuffer.shift()?.dispose();
    }

    if (this.sequenceBuffer.length < SEQUENCE_LENGTH) return null;

    const inputTensor = tf.stack(this.sequenceBuffer).expandDims(0);

    const [gestureLogits, futureValenceLogits] = await this.model.predict(inputTensor) as [tf.Tensor, tf.Tensor];

    const gestureProbs = await gestureLogits.softmax().data();
    const futureValence = await futureValenceLogits.data();

    gestureLogits.dispose();
    futureValenceLogits.dispose();
    inputTensor.dispose();

    const maxIdx = gestureProbs.indexOf(Math.max(...gestureProbs));
    const confidence = gestureProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    if (gesture !== 'none') {
      const entry = {
        id: `gesture-${Date.now()}`,
        type: gesture,
        confidence,
        futureValenceTrajectory: Array.from(futureValence),
        valenceAtRecognition: currentValence.get(),
        timestamp: Date.now(),
        decodingMethod: 'quantized_2bit_inference'
      };

      this.ySequence.push([entry]);
      wootPrecedenceGraph.insertChar(entry.id, 'START', 'END', true);

      mercyHaptic.playPattern(this.getHapticPattern(gesture), currentValence.get());
      setCurrentGesture(gesture);
    }

    return { gesture, confidence, futureValence: Array.from(futureValence) };
  }

  private flattenLandmarks(pose: any[], leftHand: any[], rightHand: any[]): number[] {
    const flatten = (landmarks: any[]) => landmarks.flatMap(p => [p.x, p.y, p.z ?? 0]);
    return [...flatten(pose), ...flatten(leftHand), ...flatten(rightHand)];
  }

  private getHapticPattern(gesture: string): string {
    switch (gesture) {
      case 'pinch': return 'allianceProposal';
      case 'spiral': return 'swarmBloom';
      case 'figure8': return 'eternalHarmony';
      default: return 'neutralPulse';
    }
  }

  getCurrentGesture(): string | null {
    return currentGesture;
  }

  async dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.sequenceBuffer.forEach(t => t.dispose());
    this.sequenceBuffer = [];
  }
}

export const blazePoseTransformerEngine = new SpatiotemporalTransformerGestures();

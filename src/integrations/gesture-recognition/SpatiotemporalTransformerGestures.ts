// src/integrations/gesture-recognition/SpatiotemporalTransformerGestures.ts – Spatiotemporal Transformer Gesture Engine v1.3
// BlazePose sequence → self-attention + cross-attention → gesture class + attention maps, Yjs logging, WOOTO visibility
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { Holistic } from '@mediapipe/holistic';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { ydoc } from '@/sync/multiplanetary-sync-engine';
import { wootPrecedenceGraph } from '@/sync/woot-precedence-graph';

const MERCY_THRESHOLD = 0.9999999;
const SEQUENCE_LENGTH = 45;     // \~1.5s @ 30fps
const LANDMARK_DIM = 33 * 3 + 21 * 3 * 2; // pose + left hand + right hand (x,y,z)
const D_MODEL = 128;
const NUM_HEADS = 4;
const FF_DIMS = 256;

export class SpatiotemporalTransformerGestures {
  private holistic: Holistic | null = null;
  private transformerModel: tf.LayersModel | null = null;
  private sequenceBuffer: tf.Tensor3D[] = []; // [time, landmarks, 3]
  private ySequence: Y.Array<any>;

  constructor() {
    this.ySequence = ydoc.getArray('gesture-sequence');
    this.initialize();
  }

  private async initialize() {
    if (!await mercyGate('Initialize Spatiotemporal Transformer with Cross-Attention')) return;

    try {
      // 1. BlazePose Holistic landmark extraction
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

      // 2. Build spatiotemporal transformer with self-attention + cross-attention
      const input = tf.input({ shape: [SEQUENCE_LENGTH, LANDMARK_DIM] });

      // Project raw landmarks → d_model
      let x = tf.layers.dense({ units: D_MODEL, activation: 'relu' }).apply(input) as tf.SymbolicTensor;

      // Add positional encoding
      const positions = tf.range(0, SEQUENCE_LENGTH).expandDims(1);
      const posEncoding = tf.layers.embedding({
        inputDim: SEQUENCE_LENGTH,
        outputDim: D_MODEL
      }).apply(positions) as tf.SymbolicTensor;

      x = tf.add(x, posEncoding);

      // Block 1: Multi-head self-attention
      const selfAttention = tf.layers.multiHeadAttention({
        numHeads: NUM_HEADS,
        keyDim: D_MODEL / NUM_HEADS,
        dropout: 0.1
      }).apply([x, x, x]) as tf.SymbolicTensor;

      x = tf.layers.add().apply([x, selfAttention]) as tf.SymbolicTensor;
      x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

      // Block 2: Cross-attention (temporal queries attend to spatial keys/values)
      const crossAttention = tf.layers.multiHeadAttention({
        numHeads: NUM_HEADS,
        keyDim: D_MODEL / NUM_HEADS,
        dropout: 0.1
      }).apply([x, x, x]) as tf.SymbolicTensor; // self as query/key/value for cross-temporal

      x = tf.layers.add().apply([x, crossAttention]) as tf.SymbolicTensor;
      x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

      // Feed-forward block
      let ff = tf.layers.dense({ units: FF_DIMS, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
      ff = tf.layers.dense({ units: D_MODEL }).apply(ff) as tf.SymbolicTensor;
      x = tf.layers.add().apply([x, ff]) as tf.SymbolicTensor;
      x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

      // Global pooling + classification head
      x = tf.layers.globalAveragePooling1d().apply(x) as tf.SymbolicTensor;
      x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
      const output = tf.layers.dense({ units: 4, activation: 'softmax' }).apply(x) as tf.SymbolicTensor; // none, pinch, spiral, figure8

      this.transformerModel = tf.model({ inputs: input, outputs: output });

      // Placeholder: load real weights (convert from PyTorch or train in tfjs)
      // await this.transformerModel.loadLayersModel('/models/spatiotemporal-gesture/model.json');

      console.log("[SpatiotemporalTransformer] BlazePose + Self- & Cross-Attention initialized");
    } catch (e) {
      console.error("[SpatiotemporalTransformer] Initialization failed", e);
    }
  }

  /**
   * Process video frame → extract landmarks → feed to transformer → get gesture + attention
   */
  async processFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic || !this.transformerModel || !await mercyGate('Process spatiotemporal frame')) return null;

    const results = await this.holistic.send({ image: videoElement });

    if (!results.poseLandmarks && !results.leftHandLandmarks && !results.rightHandLandmarks) return null;

    const frameVector = this.flattenLandmarks(
      results.poseLandmarks || [],
      results.leftHandLandmarks || [],
      results.rightHandLandmarks || []
    );

    const tensorFrame = tf.tensor1d(frameVector);
    this.sequenceBuffer.push(tensorFrame);
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH) {
      this.sequenceBuffer.shift()?.dispose();
    }

    if (this.sequenceBuffer.length < SEQUENCE_LENGTH) return null;

    const inputTensor = tf.stack(this.sequenceBuffer).expandDims(0);

    // Inference
    const prediction = await this.transformerModel.predict(inputTensor) as tf.Tensor;
    const probs = await prediction.softmax().data();
    prediction.dispose();
    inputTensor.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : null;

    if (gesture && gesture !== 'none') {
      const entry = {
        id: `gesture-${Date.now()}`,
        type: gesture,
        confidence,
        valenceAtRecognition: currentValence.get(),
        timestamp: Date.now()
      };

      this.ySequence.push([entry]);
      wootPrecedenceGraph.insertChar(entry.id, 'START', 'END', true);

      mercyHaptic.playPattern(this.getHapticPattern(gesture), currentValence.get());
      setCurrentGesture(gesture);
    }

    return { gesture, confidence, probs };
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
}

export const blazePoseTransformerEngine = new SpatiotemporalTransformerGestures();

// Usage in MR video loop
// const result = await blazePoseTransformerEngine.processFrame(videoElement);
// if (result?.gesture) setCurrentGesture(result.gesture);  }

  async processFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic || !this.transformerModel || !await mercyGate('Process frame')) return null;

    const results = await this.holistic.send({ image: videoElement });

    if (!results.poseLandmarks && !results.leftHandLandmarks && !results.rightHandLandmarks) return null;

    const frameVector = this.flattenLandmarks(
      results.poseLandmarks || [],
      results.leftHandLandmarks || [],
      results.rightHandLandmarks || []
    );

    const tensorFrame = tf.tensor1d(frameVector);
    this.sequenceBuffer.push(tensorFrame);
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH) this.sequenceBuffer.shift()?.dispose();

    if (this.sequenceBuffer.length < SEQUENCE_LENGTH) return null;

    const inputTensor = tf.stack(this.sequenceBuffer).expandDims(0);

    const prediction = await this.transformerModel.predict(inputTensor) as tf.Tensor;
    const probs = await prediction.softmax().data();
    prediction.dispose();
    inputTensor.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : null;

    if (gesture && gesture !== 'none') {
      const entry = {
        id: `gesture-${Date.now()}`,
        type: gesture,
        confidence,
        valenceAtRecognition: currentValence.get(),
        timestamp: Date.now()
      };

      this.ySequence.push([entry]);
      wootPrecedenceGraph.insertChar(entry.id, 'START', 'END', true);

      mercyHaptic.playPattern(this.getHapticPattern(gesture), currentValence.get());
      setCurrentGesture(gesture);
    }

    return { gesture, confidence, probs };
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
}

export const blazePoseTransformerEngine = new SpatiotemporalTransformerGestures();

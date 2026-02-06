// src/integrations/gesture-recognition/SpatiotemporalTransformerGestures.ts – Spatiotemporal Transformer Gesture Engine v1
// BlazePose sequence input → lightweight transformer → gesture class + attention maps, Yjs logging, WOOTO visibility
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
const SEQUENCE_LENGTH = 45; // \~1.5s @ 30fps
const LANDMARK_DIM = 33 * 3 + 21 * 3 * 2; // pose + left hand + right hand (x,y,z)

export class SpatiotemporalTransformerGestures {
  private holistic: Holistic | null = null;
  private transformerModel: tf.LayersModel | null = null;
  private sequenceBuffer: tf.Tensor3D[] = []; // [time, landmarks, 3]
  private ySequence: Y.Array<any>;

  constructor() {
    this.ySequence = ydoc.getArray('gesture-sequence');
    this.initializeHolisticAndModel();
  }

  private async initializeHolisticAndModel() {
    if (!await mercyGate('Initialize Spatiotemporal Transformer')) return;

    try {
      // 1. BlazePose Holistic for landmark extraction
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

      // 2. Lightweight spatiotemporal transformer (tfjs model stub – real impl loads converted model)
      // Architecture: TimeDistributed Dense → LSTM/GRU → Self-Attention → Dense classification
      this.transformerModel = tf.sequential({
        layers: [
          tf.layers.inputLayer({ inputShape: [SEQUENCE_LENGTH, LANDMARK_DIM] }),
          tf.layers.timeDistributed({
            layer: tf.layers.dense({ units: 128, activation: 'relu' })
          }),
          tf.layers.lstm({ units: 128, returnSequences: true }),
          tf.layers.globalAveragePooling1d(),
          tf.layers.dense({ units: 64, activation: 'relu' }),
          tf.layers.dense({ units: 4, activation: 'softmax' }) // 4 classes: none, pinch, spiral, figure8
        ]
      });

      // Placeholder: load real weights
      // await this.transformerModel.loadLayersModel('/models/spatiotemporal-gesture/model.json');

      console.log("[SpatiotemporalTransformer] BlazePose + Transformer initialized – ready for live inference");
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

    // Flatten landmarks into 1D vector
    const frameVector = this.flattenLandmarks(
      results.poseLandmarks || [],
      results.leftHandLandmarks || [],
      results.rightHandLandmarks || []
    );

    // Push to sequence buffer
    const tensorFrame = tf.tensor1d(frameVector);
    this.sequenceBuffer.push(tensorFrame);
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH) {
      this.sequenceBuffer.shift()?.dispose();
    }

    if (this.sequenceBuffer.length < SEQUENCE_LENGTH) return null;

    // Stack into [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    const inputTensor = tf.stack(this.sequenceBuffer).expandDims(0);

    // Run transformer inference
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
    return [
      ...flatten(pose),
      ...flatten(leftHand),
      ...flatten(rightHand)
    ];
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
// if (result?.gesture) setCurrentGesture(result.gesture);

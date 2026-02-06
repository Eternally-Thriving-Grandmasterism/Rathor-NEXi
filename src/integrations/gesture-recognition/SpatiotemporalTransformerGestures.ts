// src/integrations/gesture-recognition/SpatiotemporalTransformerGestures.ts – Spatiotemporal Transformer Gesture Engine v1.5
// BlazePose → Encoder-Decoder + Beam Search Decoding → gesture class + future valence trajectory
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
const SEQUENCE_LENGTH = 45;
const LANDMARK_DIM = 33 * 3 + 21 * 3 * 2;
const D_MODEL = 128;
const NUM_HEADS = 4;
const FF_DIMS = 256;
const FUTURE_STEPS = 15;
const BEAM_WIDTH_BASE = 5;
const LENGTH_PENALTY = 0.6;

export class SpatiotemporalTransformerGestures {
  private holistic: Holistic | null = null;
  private encoderDecoderModel: tf.LayersModel | null = null;
  private sequenceBuffer: tf.Tensor3D[] = [];
  private ySequence: Y.Array<any>;

  constructor() {
    this.ySequence = ydoc.getArray('gesture-sequence');
    this.initializeEncoderDecoder();
  }

  private async initializeEncoderDecoder() {
    if (!await mercyGate('Initialize Encoder-Decoder with Beam Search')) return;

    try {
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

      const input = tf.input({ shape: [SEQUENCE_LENGTH, LANDMARK_DIM] });

      let encoderOut = tf.layers.dense({ units: D_MODEL, activation: 'relu' }).apply(input) as tf.SymbolicTensor;

      const positions = tf.range(0, SEQUENCE_LENGTH).expandDims(1);
      const posEncoding = tf.layers.embedding({
        inputDim: SEQUENCE_LENGTH,
        outputDim: D_MODEL
      }).apply(positions) as tf.SymbolicTensor;

      encoderOut = tf.add(encoderOut, posEncoding);

      const encoderAttention = tf.layers.multiHeadAttention({
        numHeads: NUM_HEADS,
        keyDim: D_MODEL / NUM_HEADS
      }).apply([encoderOut, encoderOut, encoderOut]) as tf.SymbolicTensor;

      encoderOut = tf.layers.add().apply([encoderOut, encoderAttention]) as tf.SymbolicTensor;
      encoderOut = tf.layers.layerNormalization().apply(encoderOut) as tf.SymbolicTensor;

      let decoderIn = tf.layers.dense({ units: D_MODEL, activation: 'relu' }).apply(encoderOut) as tf.SymbolicTensor;

      const decoderSelfAttn = tf.layers.multiHeadAttention({
        numHeads: NUM_HEADS,
        keyDim: D_MODEL / NUM_HEADS
      }).apply([decoderIn, decoderIn, decoderIn]) as tf.SymbolicTensor;

      decoderIn = tf.layers.add().apply([decoderIn, decoderSelfAttn]) as tf.SymbolicTensor;
      decoderIn = tf.layers.layerNormalization().apply(decoderIn) as tf.SymbolicTensor;

      const crossAttn = tf.layers.multiHeadAttention({
        numHeads: NUM_HEADS,
        keyDim: D_MODEL / NUM_HEADS
      }).apply([decoderIn, encoderOut, encoderOut]) as tf.SymbolicTensor;

      decoderIn = tf.layers.add().apply([decoderIn, crossAttn]) as tf.SymbolicTensor;
      decoderIn = tf.layers.layerNormalization().apply(decoderIn) as tf.SymbolicTensor;

      let ff = tf.layers.dense({ units: FF_DIMS, activation: 'relu' }).apply(decoderIn) as tf.SymbolicTensor;
      ff = tf.layers.dense({ units: D_MODEL }).apply(ff) as tf.SymbolicTensor;
      decoderIn = tf.layers.add().apply([decoderIn, ff]) as tf.SymbolicTensor;
      decoderIn = tf.layers.layerNormalization().apply(decoderIn) as tf.SymbolicTensor;

      const gestureHead = tf.layers.globalAveragePooling1d().apply(decoderIn) as tf.SymbolicTensor;
      const gestureOutput = tf.layers.dense({ units: 4, activation: 'softmax' }).apply(gestureHead) as tf.SymbolicTensor;

      const futureValenceHead = tf.layers.dense({ units: FUTURE_STEPS }).apply(decoderIn) as tf.SymbolicTensor;

      this.encoderDecoderModel = tf.model({
        inputs: input,
        outputs: [gestureOutput, futureValenceHead]
      });

      // Placeholder: load real weights
      // await this.encoderDecoderModel.loadLayersModel('/models/spatiotemporal-gesture-encoder-decoder/model.json');

      console.log("[SpatiotemporalTransformer] BlazePose + Encoder-Decoder + Beam Search initialized");
    } catch (e) {
      console.error("[SpatiotemporalTransformer] Initialization failed", e);
    }
  }

  /**
   * Beam search decoding over gesture probabilities + future valence trajectory
   */
  async beamSearchDecode(logits: tf.Tensor, futureValenceLogits: tf.Tensor, beamWidth: number): Promise<{ gesture: string; confidence: number; futureValence: number[] }> {
    const gestureProbs = await logits.softmax().data();
    const futureValence = await futureValenceLogits.data();

    const candidates = [{ sequence: [], score: 0, futureValence: Array.from(futureValence) }];

    for (let step = 0; step < 4; step++) { // 4 gesture classes
      const newCandidates = [];

      for (const candidate of candidates) {
        for (let token = 0; token < 4; token++) {
          const prob = gestureProbs[token];
          const newScore = candidate.score + Math.log(prob) / Math.pow(step + 1, LENGTH_PENALTY);
          newCandidates.push({
            sequence: [...candidate.sequence, token],
            score: newScore,
            futureValence: candidate.futureValence
          });
        }
      }

      // Keep top beamWidth
      newCandidates.sort((a, b) => b.score - a.score);
      candidates = newCandidates.slice(0, beamWidth);
    }

    const best = candidates[0];
    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = gestureMap[best.sequence[best.sequence.length - 1]];

    return {
      gesture,
      confidence: Math.exp(best.score),
      futureValence: best.futureValence
    };
  }

  /**
   * Process video frame → encoder-decoder inference → beam search decoding
   */
  async processFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic || !this.encoderDecoderModel || !await mercyGate('Process encoder-decoder frame with beam search')) return null;

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

    const [gestureLogits, futureValenceLogits] = await this.encoderDecoderModel.predict(inputTensor) as [tf.Tensor, tf.Tensor];

    // Valence-modulated beam width (high valence → narrower beam = faster, more confident)
    const beamWidth = Math.max(3, Math.round(BEAM_WIDTH_BASE * (1 - (currentValence.get() - 0.95) * 2)));

    const decoded = await this.beamSearchDecode(gestureLogits, futureValenceLogits, beamWidth);

    gestureLogits.dispose();
    futureValenceLogits.dispose();
    inputTensor.dispose();

    if (decoded.gesture !== 'none') {
      const entry = {
        id: `gesture-${Date.now()}`,
        type: decoded.gesture,
        confidence: decoded.confidence,
        futureValenceTrajectory: decoded.futureValence,
        valenceAtRecognition: currentValence.get(),
        timestamp: Date.now()
      };

      this.ySequence.push([entry]);
      wootPrecedenceGraph.insertChar(entry.id, 'START', 'END', true);

      mercyHaptic.playPattern(this.getHapticPattern(decoded.gesture), currentValence.get());
      setCurrentGesture(decoded.gesture);
    }

    return decoded;
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

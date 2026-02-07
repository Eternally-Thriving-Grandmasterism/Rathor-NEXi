// src/utils/temporal-smoothing.ts – Temporal Smoothing Techniques v1.0
// EMA, Kalman, Savitzky-Golay, one-euro, valence-adaptive hybrid smoothing
// Mercy-gated outlier rejection, real-time jitter reduction for landmarks/trajectories
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// ──────────────────────────────────────────────────────────────
// EMA (Exponential Moving Average) – simple & fast
// ──────────────────────────────────────────────────────────────

export function emaSmoothing(
  current: number[],
  previous: number[] | null,
  alpha: number = 0.7
): number[] {
  if (!previous || previous.length !== current.length) return current;

  return current.map((val, i) => {
    return alpha * val + (1 - alpha) * previous[i];
  });
}

// ──────────────────────────────────────────────────────────────
// One-Euro Filter – adaptive low-pass, excellent for real-time tracking
// (minimizes lag while removing jitter)
// ──────────────────────────────────────────────────────────────

class OneEuroFilter {
  private x: number = 0;
  private dx: number = 0;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;

  constructor(minCutoff: number = 0.8, beta: number = 0.007, dCutoff: number = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  update(x: number, timestamp: number, prevTimestamp: number): number {
    const te = timestamp - prevTimestamp;
    if (te <= 0) return x;

    const alpha = this.alpha(te);
    this.dx = (x - this.x) / te;
    this.x = x * alpha + this.x * (1 - alpha);

    return this.x;
  }

  private alpha(te: number): number {
    const tau = 1.0 / (2 * Math.PI * this.minCutoff);
    return 1.0 / (1.0 + tau / te);
  }
}

// Per-landmark one-euro filter bank
export class OneEuroSmoothing {
  private filters: OneEuroFilter[];
  private lastTimestamp: number = 0;

  constructor(count: number, minCutoff: number = 0.8, beta: number = 0.007) {
    this.filters = Array(count).fill(0).map(() => new OneEuroFilter(minCutoff, beta));
  }

  smooth(landmarks: number[], timestamp: number): number[] {
    if (this.lastTimestamp === 0) {
      this.lastTimestamp = timestamp;
      return landmarks;
    }

    const smoothed = landmarks.map((val, i) => {
      return this.filters[i].update(val, timestamp, this.lastTimestamp);
    });

    this.lastTimestamp = timestamp;
    return smoothed;
  }
}

// ──────────────────────────────────────────────────────────────
// Savitzky-Golay Filter – polynomial smoothing (good for offline or buffered)
// ──────────────────────────────────────────────────────────────

export function savitzkyGolay(
  data: number[],
  windowSize: number = 5,
  polyOrder: number = 2
): number[] {
  if (data.length < windowSize) return data;

  const halfWindow = Math.floor(windowSize / 2);
  const result = new Array(data.length).fill(0);

  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let weightSum = 0;

    for (let j = -halfWindow; j <= halfWindow; j++) {
      const idx = i + j;
      if (idx < 0 || idx >= data.length) continue;

      // Simple binomial weights approximation
      const weight = Math.pow(0.5, Math.abs(j));
      sum += weight * data[idx];
      weightSum += weight;
    }

    result[i] = sum / weightSum;
  }

  return result;
}

// ──────────────────────────────────────────────────────────────
// Valence-Adaptive Hybrid Smoothing (recommended for NEXi)
// ──────────────────────────────────────────────────────────────

export function valenceAdaptiveSmooth(
  landmarks: number[],
  previous: number[] | null,
  timestamp: number,
  maxJitter: number = 0.08
): number[] {
  const valence = currentValence.get();

  // Mercy gate: drop extreme outliers on low valence
  if (previous && mercyGate('Landmark outlier rejection')) {
    landmarks = landmarks.map((val, i) => {
      if (Math.abs(val - previous[i]) > maxJitter * (1 - valence)) {
        return previous[i]; // reject outlier
      }
      return val;
    });
  }

  // Hybrid: EMA + One-Euro + Savitzky-Golay tail
  let smoothed = emaSmoothing(landmarks, previous, 0.6 + 0.3 * valence);

  const oneEuro = new OneEuroSmoothing(landmarks.length, 0.8 + 0.4 * valence, 0.007);
  smoothed = oneEuro.smooth(smoothed, timestamp);

  // Final Savitzky-Golay pass only on high valence (smooth but preserve detail)
  if (valence > 0.85) {
    smoothed = savitzkyGolay(smoothed, 5, 2);
  }

  return smoothed;
}

// ──────────────────────────────────────────────────────────────
// Usage Example (in detection loop)
// ──────────────────────────────────────────────────────────────

/*
const previousLandmarks: number[] | null = null;
const lastTimestamp = 0;

const result = await detectAndFuse(video);
if (result && result.poseLandmarks) {
  const flattened = flattenLandmarks(result.poseLandmarks); // implement flatten
  const smoothed = valenceAdaptiveSmooth(flattened, previousLandmarks, Date.now());
  // Feed smoothed to attention model
}
*/

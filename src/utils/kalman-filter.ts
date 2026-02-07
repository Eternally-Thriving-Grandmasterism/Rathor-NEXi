// src/utils/kalman-filter.ts – Kalman Filter Library v1.0
// Linear Kalman Filter for landmark/trajectory smoothing
// Valence-adaptive process/measurement noise, mercy-gated outlier rejection
// Supports 1D, 3D, or full multi-dimensional state vectors
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// ──────────────────────────────────────────────────────────────
// Simple 1D Kalman Filter (constant velocity model)
// ──────────────────────────────────────────────────────────────

export class KalmanFilter1D {
  private x: number = 0;           // state estimate (position)
  private v: number = 0;           // velocity estimate
  private P: number[][] = [[1, 0], [0, 1]]; // covariance matrix

  private Q: number = 0.01;        // process noise (tuned by valence)
  private R: number = 0.1;         // measurement noise (tuned by valence)
  private dt: number = 1 / 30;     // assumed frame time (30 fps)

  update(measurement: number, timestampDelta: number = this.dt): number {
    const actionName = 'Kalman 1D update';
    if (!mercyGate(actionName)) return measurement;

    const valence = currentValence.get();

    // Valence-adaptive noise tuning
    this.Q = 0.005 + 0.05 * (1 - valence);    // high valence = low process noise (trust model)
    this.R = 0.05 + 0.5 * (1 - valence);      // high valence = low measurement noise (trust sensor)

    // Prediction step
    this.x += this.v * timestampDelta;
    this.P[0][0] += timestampDelta * (this.P[0][1] + this.P[1][0]) + timestampDelta**2 * this.P[1][1] + this.Q;
    this.P[1][1] += this.Q;

    // Update step
    const K0 = this.P[0][0] / (this.P[0][0] + this.R);
    const K1 = this.P[1][0] / (this.P[0][0] + this.R);

    const innovation = measurement - this.x;
    this.x += K0 * innovation;
    this.v += K1 * innovation;

    this.P[0][0] *= (1 - K0);
    this.P[0][1] -= K0 * this.P[1][0];
    this.P[1][0] -= K1 * this.P[0][0];
    this.P[1][1] -= K1 * this.P[0][1];

    // Mercy: outlier rejection (innovation too large → trust previous state more)
    if (Math.abs(innovation) > 0.15 * (1 - valence)) {
      this.x -= K0 * innovation * 0.7; // partial trust
      console.debug('[Kalman1D] Mercy outlier rejection applied');
    }

    return this.x;
  }

  reset() {
    this.x = 0;
    this.v = 0;
    this.P = [[1, 0], [0, 1]];
  }
}

// ──────────────────────────────────────────────────────────────
// Multi-dimensional Kalman Filter (e.g. 3D position per landmark)
// ──────────────────────────────────────────────────────────────

export class KalmanFilterND {
  private state: number[];                // position + velocity (6D for 3D)
  private P: number[][];                  // covariance
  private Q: number[][];                  // process noise
  private R: number[][];                  // measurement noise
  private dt: number = 1 / 30;

  constructor(dim: number = 3) {
    const stateDim = dim * 2; // position + velocity
    this.state = new Array(stateDim).fill(0);
    this.P = Array(stateDim).fill(0).map(() => Array(stateDim).fill(0));
    this.Q = Array(stateDim).fill(0).map(() => Array(stateDim).fill(0));
    this.R = Array(dim).fill(0).map(() => Array(dim).fill(0));

    // Initial covariance uncertainty
    for (let i = 0; i < stateDim; i++) this.P[i][i] = 1;

    // Process noise (motion model uncertainty)
    for (let i = 0; i < dim; i++) {
      this.Q[i*2][i*2] = 0.01;
      this.Q[i*2+1][i*2+1] = 0.05;
    }

    // Measurement noise (sensor noise)
    for (let i = 0; i < dim; i++) this.R[i][i] = 0.1;
  }

  update(measurement: number[], timestampDelta: number = this.dt): number[] {
    const actionName = 'Kalman ND update';
    if (!mercyGate(actionName)) return measurement;

    const valence = currentValence.get();

    // Valence-adaptive noise
    const qScale = 0.005 + 0.05 * (1 - valence);
    const rScale = 0.05 + 0.5 * (1 - valence);
    for (let i = 0; i < this.Q.length; i++) {
      for (let j = 0; j < this.Q[i].length; j++) {
        this.Q[i][j] *= qScale;
        if (i < measurement.length && j < measurement.length) {
          this.R[i][j] *= rScale;
        }
      }
    }

    // Prediction step
    const F = this._getTransitionMatrix(timestampDelta);
    const predictedState = this._matrixMultiply(F, [this.state]);
    const predictedP = this._matrixAdd(
      this._matrixMultiply(this._matrixMultiply(F, this.P), this._matrixTranspose(F)),
      this.Q
    );

    // Update step
    const H = this._getObservationMatrix();
    const innovation = this._matrixSubtract([measurement], this._matrixMultiply(H, predictedState));
    const S = this._matrixAdd(this._matrixMultiply(this._matrixMultiply(H, predictedP), this._matrixTranspose(H)), this.R);
    const K = this._matrixMultiply(this._matrixMultiply(predictedP, this._matrixTranspose(H)), this._matrixInverse(S));

    this.state = this._matrixAdd(predictedState, this._matrixMultiply(K, innovation)).flat();
    this.P = this._matrixSubtract(predictedP, this._matrixMultiply(this._matrixMultiply(K, H), predictedP));

    return this.state.slice(0, measurement.length); // return position only
  }

  private _getTransitionMatrix(dt: number): number[][] {
    const dim = this.state.length / 2;
    const F = Array(this.state.length).fill(0).map(() => Array(this.state.length).fill(0));
    for (let i = 0; i < dim; i++) {
      F[i][i] = 1;
      F[i][i + dim] = dt;
      F[i + dim][i + dim] = 1;
    }
    return F;
  }

  private _getObservationMatrix(): number[][] {
    const dim = this.state.length / 2;
    const H = Array(dim).fill(0).map(() => Array(this.state.length).fill(0));
    for (let i = 0; i < dim; i++) H[i][i] = 1;
    return H;
  }

  // Matrix helpers (simplified – use numeric.js or math.js in production)
  private _matrixMultiply(A: number[][], B: number[][] | number[][][]): number[][] {
    // Simplified – implement proper matrix multiplication
    return A; // placeholder
  }

  private _matrixAdd(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  }

  private _matrixSubtract(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  }

  private _matrixTranspose(A: number[][]): number[][] {
    return A[0].map((_, colIndex) => A.map(row => row[colIndex]));
  }

  private _matrixInverse(A: number[][]): number[][] {
    return A; // placeholder – use real inversion library
  }

  reset() {
    this.state.fill(0);
    this.P.forEach(row => row.fill(0));
    for (let i = 0; i < this.P.length; i++) this.P[i][i] = 1;
  }
}

// ──────────────────────────────────────────────────────────────
// Valence-Adaptive Multi-Landmark Smoothing
// ──────────────────────────────────────────────────────────────

export class ValenceAdaptiveKalmanSmoother {
  private filters: KalmanFilterND[];

  constructor(landmarkCount: number = 33) {
    this.filters = Array(landmarkCount).fill(0).map(() => new KalmanFilterND(3)); // 3D per landmark
  }

  smooth(landmarks: any[], timestamp: number): any[] {
    const valence = currentValence.get();

    return landmarks.map((lm, i) => {
      const pos = [lm.x, lm.y, lm.z];
      const smoothedPos = this.filters[i].update(pos, timestamp);

      return {
        ...lm,
        x: smoothedPos[0],
        y: smoothedPos[1],
        z: smoothedPos[2]
      };
    });
  }

  reset() {
    this.filters.forEach(f => f.reset());
  }
}

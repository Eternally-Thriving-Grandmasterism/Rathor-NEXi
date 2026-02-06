// src/core/valence-projection.ts – Valence Projection Engine v1.0
// Simulates future valence trajectory for mercy gating, QAT-KD decisions, scaling gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from './valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const PROJECTION_HORIZON_STEPS = 30;       // simulate next 30 steps
const VALENCE_DROP_TOLERANCE = 0.05;       // max allowable drop vs baseline
const SMOOTHING_FACTOR = 0.85;             // EMA smoothing for trajectory
const DEFAULT_BASELINE_VALENCE = 0.92;     // fallback teacher baseline

interface ProjectionResult {
  currentValence: number;
  projectedValence: number;
  trajectory: number[];                     // next PROJECTION_HORIZON_STEPS values
  dropFromBaseline: number;
  isSafe: boolean;
  reason?: string;
}

let lastProjection: ProjectionResult | null = null;
let projectionCacheTTL = 5000;              // 5 seconds cache

export class ValenceProjection {
  /**
   * Project future valence trajectory based on current state & simulated dynamics
   * @param customBaseline optional teacher baseline override
   * @returns ProjectionResult with safety decision
   */
  static async project(customBaseline?: number): Promise<ProjectionResult> {
    const now = Date.now();

    // Cache hit
    if (lastProjection && now - lastProjection.timestamp < projectionCacheTTL) {
      return lastProjection;
    }

    const actionName = 'Project future valence trajectory';
    if (!await mercyGate(actionName)) {
      return {
        currentValence: currentValence.get(),
        projectedValence: 0,
        trajectory: [],
        dropFromBaseline: 1.0,
        isSafe: false,
        reason: 'Mercy gate blocked projection'
      };
    }

    const current = currentValence.get();
    const baseline = customBaseline ?? DEFAULT_BASELINE_VALENCE;

    // Simplified trajectory simulation (real impl would use learned dynamics)
    const trajectory: number[] = [];
    let projected = current;

    for (let step = 0; step < PROJECTION_HORIZON_STEPS; step++) {
      // Decay + random walk + mean-reversion toward baseline
      const decay = 0.005 * (projected - baseline);          // pull toward baseline
      const noise = (Math.random() - 0.5) * 0.02;
      projected = SMOOTHING_FACTOR * projected + (1 - SMOOTHING_FACTOR) * baseline;
      projected += decay + noise;

      // Hard clamp [0,1]
      projected = Math.max(0, Math.min(1, projected));
      trajectory.push(projected);
    }

    const finalProjected = trajectory[trajectory.length - 1];
    const maxDrop = Math.max(...trajectory.map(v => baseline - v));

    const result: ProjectionResult = {
      currentValence: current,
      projectedValence: finalProjected,
      trajectory,
      dropFromBaseline: maxDrop,
      isSafe: maxDrop <= VALENCE_DROP_TOLERANCE,
      reason: maxDrop <= VALENCE_DROP_TOLERANCE
        ? `Safe projection (max drop ${maxDrop.toFixed(4)})`
        : `Dangerous drop detected: ${maxDrop.toFixed(4)} > ${VALENCE_DROP_TOLERANCE}`
    };

    // Cache result
    lastProjection = { ...result, timestamp: now };

    // Haptic feedback on dangerous projection
    if (!result.isSafe) {
      mercyHaptic.playPattern('warningPulse', current);
      console.warn("[ValenceProjection] Low future trajectory detected – gating action");
    }

    return result;
  }

  /**
   * Quick safety check – used in mercy gates
   */
  static async isFutureSafe(customBaseline?: number): Promise<boolean> {
    const proj = await this.project(customBaseline);
    return proj.isSafe;
  }

  /**
   * Get last known projection (cached)
   */
  static getLastProjection(): ProjectionResult | null {
    return lastProjection;
  }

  /**
   * Reset cache (e.g. after major state change)
   */
  static resetCache() {
    lastProjection = null;
  }
}

export default ValenceProjection;

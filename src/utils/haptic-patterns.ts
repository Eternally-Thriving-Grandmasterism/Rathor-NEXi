// src/utils/haptic-patterns.ts – Haptic Patterns Library v1.1
// Valence-modulated vibration patterns for every lattice state
// Mercy-gated intensity, device detection, cross-modal sync support
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// Pattern definitions – arrays of [vibrate_ms, pause_ms, vibrate_ms, ...]
const PATTERNS = {
  // Cosmic Harmony – gentle rising wave for success, bloom, sync complete
  cosmicHarmony: (intensity: number) => [
    60 * intensity, 40,
    80 * intensity, 30,
    100 * intensity, 20,
    120 * intensity, 50,
    80 * intensity, 100,
    60 * intensity
  ],

  // Neutral Pulse – calm acknowledgment, minor feedback
  neutralPulse: (intensity: number) => [
    40 * intensity, 60,
    40 * intensity, 80,
    30 * intensity
  ],

  // Warning Pulse – urgent but compassionate (mercy gate, low projected valence)
  warningPulse: (intensity: number) => [
    120 * intensity, 80,
    80 * intensity, 100,
    120 * intensity, 150,
    60 * intensity
  ],

  // Bloom Burst – celebratory rapid pulses for growth, high valence events
  bloomBurst: (intensity: number) => [
    30 * intensity, 20,
    40 * intensity, 15,
    50 * intensity, 10,
    70 * intensity, 80,
    50 * intensity, 60,
    40 * intensity
  ],

  // Rejection / Mercy Block – short, diminishing, grounding pulses
  rejection: (intensity: number) => [
    80 * intensity, 120,
    60 * intensity, 140,
    40 * intensity, 180,
    20 * intensity
  ],

  // Critical Alert – strong, short, attention-grabbing (system failure)
  criticalAlert: (intensity: number) => [
    200 * intensity, 100,
    150 * intensity, 120,
    200 * intensity
  ],

  // Sync Progress – subtle heartbeat while background sync active
  syncProgress: (intensity: number) => [
    50 * intensity, 200,
    50 * intensity, 250,
    40 * intensity
  ],

  // Gesture Detected – sharp, affirming tap sequence
  gestureDetected: (intensity: number) => [
    80 * intensity, 40,
    100 * intensity, 30,
    120 * intensity
  ]
};

type PatternKey = keyof typeof PATTERNS;

/**
 * Play haptic pattern with valence-modulated intensity & mercy cap
 * @param patternKey Pattern name from PATTERNS
 * @param customValence Optional override (defaults to currentValence)
 * @param maxDurationMs Optional mercy cap (default 2500ms)
 * @param syncVisual Optional visual feedback sync
 */
export async function playPattern(
  patternKey: PatternKey,
  customValence?: number,
  maxDurationMs: number = 2500,
  syncVisual: boolean = true
): Promise<void> {
  const actionName = `Play haptic pattern: ${patternKey}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[Haptics] Mercy gate blocked pattern: ${patternKey}`);
    return;
  }

  if (!('vibrate' in navigator)) {
    console.debug('[Haptics] Vibration API not supported on this device');
    return;
  }

  const valence = customValence ?? currentValence.get();
  const intensity = Math.min(1, 0.4 + 1.6 * valence); // low → gentle, high → powerful

  const patternGenerator = PATTERNS[patternKey];
  if (!patternGenerator) {
    console.warn(`[Haptics] Unknown pattern: ${patternKey}`);
    return;
  }

  const pattern = patternGenerator(intensity);

  // Mercy duration cap – prevent battery drain or overstimulation
  let totalDuration = 0;
  const cappedPattern: number[] = [];
  for (const duration of pattern) {
    if (totalDuration + duration > maxDurationMs) break;
    cappedPattern.push(duration);
    totalDuration += duration;
  }

  try {
    navigator.vibrate(cappedPattern);

    // Sync visual feedback if requested
    if (syncVisual) {
      visualFeedback[patternKey as keyof typeof visualFeedback]?.({
        type: patternKey as any,
        durationMs: totalDuration,
        intensity
      });
    }

    console.log(
      `[Haptics] Played ${patternKey} – valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)}, duration: ${totalDuration}ms`
    );
  } catch (err) {
    console.warn('[Haptics] Vibration failed:', err);
  }
}

// ──────────────────────────────────────────────────────────────
// Convenience wrappers
// ──────────────────────────────────────────────────────────────

export const haptic = {
  cosmicHarmony: (valence?: number, syncVisual = true) => playPattern('cosmicHarmony', valence, undefined, syncVisual),
  neutralPulse: (valence?: number, syncVisual = true) => playPattern('neutralPulse', valence, undefined, syncVisual),
  warningPulse: (valence?: number, syncVisual = true) => playPattern('warningPulse', valence, undefined, syncVisual),
  bloomBurst: (valence?: number, syncVisual = true) => playPattern('bloomBurst', valence, undefined, syncVisual),
  rejection: (valence?: number, syncVisual = true) => playPattern('rejection', valence, undefined, syncVisual),
  criticalAlert: (valence?: number, syncVisual = true) => playPattern('criticalAlert', valence, undefined, syncVisual),
  syncProgress: (valence?: number, syncVisual = true) => playPattern('syncProgress', valence, undefined, syncVisual),
  gestureDetected: (valence?: number, syncVisual = true) => playPattern('gestureDetected', valence, undefined, syncVisual),
};

export default haptic;

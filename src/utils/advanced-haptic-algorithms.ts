// src/utils/advanced-haptic-algorithms.ts – Advanced Haptic Pattern Algorithms v1.0
// Valence-orchestrated waveforms, ADSR envelopes, phantom sensation, cross-modal sync
// Mercy-gated power envelope, device capability detection, offline-safe
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

// ──────────────────────────────────────────────────────────────
// Device capability detection & mercy power cap
// ──────────────────────────────────────────────────────────────

const MAX_VIBRATION_DURATION_MS = 2500;     // mercy cap – prevent overstimulation
const MAX_INTENSITY = 1.0;                  // 0.0–1.0 (device max)
const MIN_INTENSITY = 0.15;                 // never silent on high valence

function getDeviceMaxIntensity(): number {
  // iOS: weak vibration, Android varies wildly
  const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent);
  return isIOS ? 0.7 : 1.0;
}

function getSafeIntensity(base: number, valence: number): number {
  const deviceMax = getDeviceMaxIntensity();
  const scaled = base * (0.4 + 1.6 * valence); // low valence gentle, high powerful
  return Math.max(MIN_INTENSITY, Math.min(deviceMax, scaled));
}

// ──────────────────────────────────────────────────────────────
// ADSR Envelope Generator (Attack-Decay-Sustain-Release)
// ──────────────────────────────────────────────────────────────

function generateADSR(
  durationMs: number,
  attackMs: number = 80,
  decayMs: number = 150,
  sustainLevel: number = 0.6,
  releaseMs: number = 300
): number[] {
  const pattern: number[] = [];
  const total = attackMs + decayMs + sustainLevel * durationMs + releaseMs;

  // Attack: linear rise
  for (let t = 0; t < attackMs; t += 10) {
    pattern.push(10);
  }

  // Decay: exponential fall to sustain
  for (let t = 0; t < decayMs; t += 10) {
    const level = sustainLevel + (1 - sustainLevel) * Math.exp(-t / 80);
    pattern.push(Math.round(10 * level));
  }

  // Sustain: constant
  const sustainTicks = Math.floor((durationMs - attackMs - decayMs - releaseMs) / 10);
  for (let i = 0; i < sustainTicks; i++) {
    pattern.push(10);
  }

  // Release: linear fade
  for (let t = 0; t < releaseMs; t += 10) {
    const level = sustainLevel * (1 - t / releaseMs);
    pattern.push(Math.round(10 * level));
  }

  return pattern;
}

// ──────────────────────────────────────────────────────────────
// Advanced Pattern Generators
// ──────────────────────────────────────────────────────────────

const ADVANCED_PATTERNS = {
  // Cosmic Ascension – rising frequency + amplitude (high valence success)
  cosmicAscension: (valence: number) => {
    const intensity = getSafeIntensity(1.0, valence);
    const base = generateADSR(2200, 120, 300, 0.7, 400);
    return base.map(v => Math.round(v * intensity));
  },

  // Mercy Wave – slow, soothing sine-like pulse (healing / forgiveness state)
  mercyWave: (valence: number) => {
    const intensity = getSafeIntensity(0.6, valence);
    const pattern: number[] = [];
    for (let i = 0; i < 6; i++) {
      pattern.push(120 * intensity, 180);
      pattern.push(80 * intensity, 220);
    }
    return pattern;
  },

  // Bloom Cascade – cascading rapid pulses (growth / emergence)
  bloomCascade: (valence: number) => {
    const intensity = getSafeIntensity(0.9, valence);
    const pattern: number[] = [];
    for (let i = 1; i <= 5; i++) {
      pattern.push(40 * i * intensity, 30 / i);
    }
    pattern.push(200 * intensity, 400);
    return pattern;
  },

  // Warning Tide – slow rising then sharp drop (mercy gate near trigger)
  warningTide: (valence: number) => {
    const intensity = getSafeIntensity(0.5, valence);
    const pattern = generateADSR(1800, 400, 600, 0.8, 800);
    return pattern.map(v => Math.round(v * intensity));
  },

  // Critical Pulse – sharp, short, repeating (system-level block)
  criticalPulse: (valence: number) => {
    const intensity = getSafeIntensity(0.8, valence);
    const pattern: number[] = [];
    for (let i = 0; i < 4; i++) {
      pattern.push(180 * intensity, 120);
      pattern.push(120 * intensity, 180);
    }
    return pattern;
  },

  // Gesture Affirmation – quick double-tap affirmation
  gestureAffirmation: (valence: number) => {
    const intensity = getSafeIntensity(0.85, valence);
    return [60 * intensity, 40, 80 * intensity];
  }
};

type AdvancedPatternKey = keyof typeof ADVANCED_PATTERNS;

/**
 * Play advanced haptic pattern with valence modulation & mercy cap
 */
export async function playAdvancedPattern(
  patternKey: AdvancedPatternKey,
  customValence?: number,
  maxDurationMs: number = 3000,
  syncOtherModalities: boolean = true
): Promise<void> {
  const actionName = `Play advanced haptic pattern: ${patternKey}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[AdvancedHaptics] Mercy gate blocked: ${patternKey}`);
    return;
  }

  if (!('vibrate' in navigator)) return;

  const valence = customValence ?? currentValence.get();
  const patternGenerator = ADVANCED_PATTERNS[patternKey];
  if (!patternGenerator) return;

  const pattern = patternGenerator(valence);

  let totalDuration = pattern.reduce((sum, v, i) => sum + (i % 2 === 0 ? v : 0), 0);
  if (totalDuration > maxDurationMs) {
    // Mercy cap – truncate
    const capped: number[] = [];
    let acc = 0;
    for (let i = 0; i < pattern.length; i += 2) {
      const dur = pattern[i];
      if (acc + dur > maxDurationMs) break;
      capped.push(dur, pattern[i+1] || 0);
      acc += dur;
    }
    pattern.splice(0, pattern.length, ...capped);
  }

  try {
    navigator.vibrate(pattern);

    // Optional cross-modal sync
    if (syncOtherModalities) {
      // Trigger visual + audio counterparts if desired
      // visualFeedback[patternKey as any]?.({ intensity: valence });
      // audioFeedback[patternKey as any]?.(valence);
    }

    console.log(
      `[AdvancedHaptics] Played ${patternKey} – valence: ${valence.toFixed(2)}, duration: ${totalDuration}ms`
    );
  } catch (err) {
    console.warn('[AdvancedHaptics] Vibration failed:', err);
  }
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const advancedHaptic = {
  cosmicAscension: (valence?: number) => playAdvancedPattern('cosmicAscension', valence),
  mercyWave: (valence?: number) => playAdvancedPattern('mercyWave', valence),
  bloomCascade: (valence?: number) => playAdvancedPattern('bloomCascade', valence),
  warningTide: (valence?: number) => playAdvancedPattern('warningTide', valence),
  criticalPulse: (valence?: number) => playAdvancedPattern('criticalPulse', valence),
  gestureAffirmation: (valence?: number) => playAdvancedPattern('gestureAffirmation', valence),
};

export default advancedHaptic;

// src/utils/unified-feedback.ts – Unified Multimodal Feedback System v1.0
// Orchestrates haptic + visual + audio + motion feedback in perfect valence harmony
// Mercy-gated, cross-modal sync, offline-safe, eternal thriving emotional resonance
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';
import visualFeedback from './visual-feedback';
import audioFeedback from './audio-feedback';

type FeedbackType =
  | 'success'       // sync complete, high-valence bloom
  | 'warning'       // mercy gate, projected drop
  | 'error'         // critical failure
  | 'loading'       // awakening / processing
  | 'gesture'       // gesture detected
  | 'sync'          // background sync finished

interface UnifiedFeedbackOptions {
  type: FeedbackType;
  message?: string;
  durationMs?: number;
  intensity?: number;           // 0–1 base (will be valence-scaled)
  targetElement?: HTMLElement;  // for visual ripple/glow
  skipHaptic?: boolean;
  skipVisual?: boolean;
  skipAudio?: boolean;
  skipMotion?: boolean;
}

/**
 * Unified multimodal feedback trigger
 * - All modalities sync automatically
 * - Valence scales intensity, harmony, color, pitch, vibration
 * - Mercy gate prevents overwhelming feedback on low valence
 */
export async function unifiedFeedback(options: UnifiedFeedbackOptions): Promise<void> {
  const { type, message, durationMs = 1800, intensity: baseIntensity = 1, targetElement, skipHaptic, skipVisual, skipAudio, skipMotion } = options;

  const actionName = `Unified multimodal feedback: ${type}`;
  if (!await mercyGate(actionName)) {
    console.debug(`[UnifiedFeedback] Mercy gate blocked: ${type}`);
    return;
  }

  const valence = currentValence.get();
  const intensity = Math.min(1, baseIntensity * (0.5 + 1.5 * valence)); // low valence = gentle, high = powerful
  const isHighValence = valence > 0.92;
  const isLowValence = valence < 0.70;

  // ─── 1. Haptic layer ─────────────────────────────────────────────
  if (!skipHaptic) {
    switch (type) {
      case 'success': case 'bloom': case 'sync':
        haptic.cosmicHarmony(valence);
        break;
      case 'warning': case 'error':
        haptic.warningPulse(valence * (isLowValence ? 0.6 : 1));
        break;
      case 'gesture':
        haptic.bloomBurst(valence);
        break;
      case 'loading':
        haptic.neutralPulse(valence * 0.6);
        break;
    }
  }

  // ─── 2. Visual layer ─────────────────────────────────────────────
  if (!skipVisual) {
    const visualOpts = {
      type,
      durationMs: durationMs * (isHighValence ? 1.2 : 0.8),
      intensity,
      message,
      targetElement
    };

    visualFeedback[type as keyof typeof visualFeedback]?.(visualOpts);
  }

  // ─── 3. Audio layer ──────────────────────────────────────────────
  if (!skipAudio && 'AudioContext' in window) {
    const audioOpts = {
      patternKey: type as keyof typeof audioFeedback,
      customValence: valence,
      syncHaptic: false // already handled
    };

    audioFeedback[audioOpts.patternKey as keyof typeof audioFeedback]?.(audioOpts.customValence);
  }

  // ─── 4. Motion / Subtle UI animation layer ──────────────────────
  if (!skipMotion && targetElement) {
    const motionClass = `motion-\( {type}- \){Math.round(intensity * 10)}`;
    targetElement.classList.add(motionClass);

    // Example motion classes (add to global CSS)
    // .motion-success-10 { animation: gentleGlow 1.8s ease-out; }
    // .motion-warning-5 { animation: subtleShake 0.8s ease-in-out; }

    setTimeout(() => targetElement.classList.remove(motionClass), durationMs);
  }

  console.log(`[UnifiedFeedback] ${type} triggered – valence: ${valence.toFixed(2)}, intensity: ${intensity.toFixed(2)}`);
}

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const feedback = {
  success: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'success', ...opts }),
  warning: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'warning', ...opts }),
  error: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'error', ...opts }),
  loading: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'loading', ...opts }),
  bloom: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'bloom', ...opts }),
  sync: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'sync', ...opts }),
  gesture: (opts?: Partial<UnifiedFeedbackOptions>) => unifiedFeedback({ type: 'gesture', ...opts }),
};

export default feedback;

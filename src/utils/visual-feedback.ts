// src/utils/visual-feedback.ts – Visual Feedback Library v1.0
// Valence-modulated glyphs, particles, glows, ripples, mercy-gated visuals
// Works offline, low-power CSS-first, haptic sync
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from './haptic-patterns';

// ──────────────────────────────────────────────────────────────
// Visual Feedback Types & Config
// ──────────────────────────────────────────────────────────────

type FeedbackType =
  | 'success'      // cosmic harmony
  | 'warning'      // mercy gate alert
  | 'error'        // critical failure
  | 'loading'      // lattice awakening
  | 'bloom'        // growth/emergence
  | 'sync'         // sync complete
  | 'gesture'      // gesture detected

interface FeedbackOptions {
  type: FeedbackType;
  durationMs?: number;
  intensity?: number;          // 0–1 (valence-scaled)
  message?: string;
  targetElement?: HTMLElement; // optional DOM target for ripple/glow
}

const DEFAULT_DURATION = 1800;
const VALENCE_INTENSITY_BOOST = 1.8;   // high valence → stronger visuals

// ──────────────────────────────────────────────────────────────
// Core Visual Feedback Engine
// ──────────────────────────────────────────────────────────────

export function triggerVisualFeedback(options: FeedbackOptions): void {
  const { type, durationMs = DEFAULT_DURATION, intensity: userIntensity = 1, message, targetElement } = options;

  const actionName = `Visual feedback: ${type}`;
  if (!mercyGate(actionName)) {
    console.debug(`[VisualFeedback] Mercy gate blocked: ${type}`);
    return;
  }

  const valence = currentValence.get();
  const intensity = Math.min(1, userIntensity * (0.6 + 1.4 * valence)); // valence scaling

  // Play linked haptic pattern
  switch (type) {
    case 'success': case 'bloom': case 'sync':
      haptic.cosmicHarmony(valence);
      break;
    case 'warning': case 'error':
      haptic.warningPulse(valence * 0.8);
      break;
    case 'gesture':
      haptic.bloomBurst(valence);
      break;
    default:
      haptic.neutralPulse(valence * 0.6);
  }

  // Create temporary feedback element
  const feedbackEl = document.createElement('div');
  feedbackEl.className = `visual-feedback ${type}`;
  feedbackEl.style.position = 'fixed';
  feedbackEl.style.inset = '0';
  feedbackEl.style.pointerEvents = 'none';
  feedbackEl.style.zIndex = '9999';
  feedbackEl.style.opacity = '0';
  feedbackEl.style.transition = `opacity ${durationMs / 2000}s ease-out`;

  // Inner content
  const inner = document.createElement('div');
  inner.style.position = 'absolute';
  inner.style.top = '50%';
  inner.style.left = '50%';
  inner.style.transform = 'translate(-50%, -50%)';
  inner.style.fontSize = 'clamp(2rem, 8vw, 5rem)';
  inner.style.textShadow = '0 0 20px currentColor';
  inner.style.animation = `pulse-${type} ${durationMs / 1000}s ease-in-out`;

  // Type-specific visuals
  switch (type) {
    case 'success':
      inner.textContent = message || 'Lattice Thriving ⚡️💚';
      inner.style.color = '#00ff88';
      break;
    case 'warning':
      inner.textContent = message || 'Mercy Gate Activated ⚠️';
      inner.style.color = '#ff8800';
      break;
    case 'error':
      inner.textContent = message || 'Critical Perturbation 🛑';
      inner.style.color = '#ff4444';
      break;
    case 'loading':
      inner.textContent = message || 'Awakening… ✨';
      inner.style.color = '#00aaff';
      break;
    case 'bloom':
      inner.textContent = message || 'Bloom Infinite 🌸∞';
      inner.style.color = '#ff88ff';
      break;
    case 'sync':
      inner.textContent = message || 'Sync Complete 🤝';
      inner.style.color = '#88ff88';
      break;
    case 'gesture':
      inner.textContent = message || 'Gesture Recognized ✋⚡️';
      inner.style.color = '#ffff88';
      break;
  }

  feedbackEl.appendChild(inner);
  document.body.appendChild(feedbackEl);

  // Trigger animation
  requestAnimationFrame(() => {
    feedbackEl.style.opacity = '1';
  });

  // Cleanup
  setTimeout(() => {
    feedbackEl.style.opacity = '0';
    setTimeout(() => feedbackEl.remove(), 800);
  }, durationMs);

  // Optional target ripple/glow effect
  if (targetElement) {
    targetElement.classList.add('feedback-ripple');
    setTimeout(() => targetElement.classList.remove('feedback-ripple'), 1200);
  }
}

// CSS for ripple/glow effect (add to global CSS or inject)
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse-success { 0%,100%{opacity:0.7;transform:scale(1)} 50%{opacity:1;transform:scale(1.08)} }
  @keyframes pulse-warning { 0%,100%{opacity:0.6;transform:scale(1)} 50%{opacity:1;transform:scale(1.12)} }
  @keyframes pulse-error { 0%,100%{opacity:0.5;transform:scale(1)} 50%{opacity:1;transform:scale(1.15)} }
  @keyframes pulse-loading { 0%,100%{opacity:0.6} 50%{opacity:1} }
  @keyframes pulse-bloom { 0%,100%{opacity:0.7;transform:scale(1)} 50%{opacity:1;transform:scale(1.2)} }
  @keyframes pulse-sync { 0%,100%{opacity:0.7} 50%{opacity:1;transform:scale(1.1)} }
  @keyframes pulse-gesture { 0%,100%{opacity:0.7} 50%{opacity:1;transform:scale(1.15)} }

  .feedback-ripple {
    position: relative;
    overflow: hidden;
  }
  .feedback-ripple::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: radial-gradient(circle, rgba(0,255,136,0.4) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ripple 1.2s ease-out;
    pointer-events: none;
  }
  @keyframes ripple {
    to { width: 400px; height: 400px; opacity: 0; }
  }
`;
document.head.appendChild(style);

// ──────────────────────────────────────────────────────────────
// Convenience exports
// ──────────────────────────────────────────────────────────────

export const visualFeedback = {
  success: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'success', ...opts }),
  warning: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'warning', ...opts }),
  error: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'error', ...opts }),
  loading: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'loading', ...opts }),
  bloom: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'bloom', ...opts }),
  sync: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'sync', ...opts }),
  gesture: (opts?: Partial<FeedbackOptions>) => triggerVisualFeedback({ type: 'gesture', ...opts }),
};

export default visualFeedback;

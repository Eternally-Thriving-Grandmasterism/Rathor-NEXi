// src/core/touch-screen-accessibility.ts – Touch Screen Accessibility Layer v1.0
// WCAG 2.2 AA+ touch support: 44×44px targets, gesture conflict prevention,
// valence-modulated touch feedback, mercy-gated sensitivity, offline-safe
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import haptic from '@/utils/haptic-patterns';
import visualFeedback from '@/utils/visual-feedback';

// ──────────────────────────────────────────────────────────────
// Touch Target Size Enforcement (WCAG 2.5.8 Level AA – 44×44px min)
// ──────────────────────────────────────────────────────────────

const MIN_TOUCH_TARGET_SIZE = 44; // pixels

export function enforceTouchTargetSize() {
  const styleId = 'touch-target-size';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    /* Ensure interactive elements meet 44×44px touch target */
    button, [role="button"], a[href], input, select, textarea, [tabindex]:not([tabindex="-1"]) {
      min-width: ${MIN_TOUCH_TARGET_SIZE}px;
      min-height: ${MIN_TOUCH_TARGET_SIZE}px;
      padding: calc((44px - 1em) / 2) 1em; /* inner padding fallback */
      touch-action: manipulation;
    }

    /* Expand touch area with invisible pseudo-element */
    .touch-target-expand::before {
      content: '';
      position: absolute;
      inset: -10px; /* extends hit area */
      pointer-events: auto;
    }

    /* Valence-modulated larger targets on high valence (joyful interaction) */
    [data-valence-high] button,
    [data-valence-high] [role="button"] {
      min-width: 56px;
      min-height: 56px;
    }
  `;
  document.head.appendChild(style);
}

// ──────────────────────────────────────────────────────────────
// Prevent gesture conflicts (swipe vs scroll, double-tap zoom, etc.)
// ──────────────────────────────────────────────────────────────

export function preventGestureConflicts() {
  document.addEventListener('touchstart', e => {
    if (e.touches.length > 1) {
      // Multi-touch – prevent pinch-zoom if not desired
      if (!mercyGate('Multi-touch gesture allowance')) {
        e.preventDefault();
      }
    }
  }, { passive: false });

  // Disable double-tap zoom on mobile
  document.addEventListener('dblclick', e => {
    e.preventDefault();
  }, { passive: false });

  // Mercy: allow zoom on high valence (joyful exploration)
  const valence = currentValence.get();
  if (valence > 0.92) {
    document.documentElement.style.touchAction = 'pinch-zoom pan-x pan-y';
  }
}

// ──────────────────────────────────────────────────────────────
// Valence-modulated touch feedback (haptic + visual ripple)
// ──────────────────────────────────────────────────────────────

export function triggerTouchFeedback(
  element: HTMLElement,
  type: 'tap' | 'hold' | 'swipe' | 'longpress' = 'tap'
) {
  const valence = currentValence.get();

  // Visual ripple effect
  const ripple = document.createElement('div');
  ripple.style.position = 'absolute';
  ripple.style.borderRadius = '50%';
  ripple.style.background = `rgba(0, 255, 136, ${0.3 + 0.4 * valence})`;
  ripple.style.transform = 'scale(0)';
  ripple.style.animation = `touch-ripple ${0.6 + 0.4 * valence}s ease-out`;
  ripple.style.pointerEvents = 'none';

  element.style.position = 'relative';
  element.appendChild(ripple);

  setTimeout(() => ripple.remove(), 1000);

  // Haptic feedback
  if ('vibrate' in navigator) {
    const hapticPattern = type === 'tap'
      ? [30 + 20 * valence]
      : type === 'longpress'
      ? [80, 50, 80]
      : [50];

    navigator.vibrate(hapticPattern);
  }

  // Announce to screen readers
  announceToScreenReader(`${type} action detected`, 'polite');
}

// CSS for ripple animation (add to global styles or inject)
const style = document.createElement('style');
style.textContent = `
  @keyframes touch-ripple {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);

// ──────────────────────────────────────────────────────────────
// Touch target expansion utility (for small buttons/links)
// ──────────────────────────────────────────────────────────────

export function expandTouchTarget(element: HTMLElement) {
  element.classList.add('touch-target-expand');
  element.style.position = 'relative';
}

// ──────────────────────────────────────────────────────────────
// Initialize touch accessibility features
// ──────────────────────────────────────────────────────────────

if (typeof window !== 'undefined') {
  enforceTouchTargetSize();
  preventGestureConflicts();

  // Auto-expand touch targets on interactive elements
  document.querySelectorAll('button, a[href], input, select, textarea').forEach(el => {
    expandTouchTarget(el as HTMLElement);
  });

  // Touch event listener for feedback
  document.addEventListener('touchstart', e => {
    const target = e.target as HTMLElement;
    if (target.closest('button, a[href], [role="button"]')) {
      triggerTouchFeedback(target);
    }
  }, { passive: true });
}

// ──────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────

export const touchA11y = {
  enforceTouchTargetSize,
  preventGestureConflicts,
  triggerTouchFeedback,
  expandTouchTarget,
};

export default touchA11y;

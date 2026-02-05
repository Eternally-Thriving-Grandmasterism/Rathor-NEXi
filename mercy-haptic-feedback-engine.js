// mercy-haptic-feedback-engine.js – sovereign Mercy Haptic Feedback Engine v1
// Dual-rumble patterns, mercy-gated, valence-modulated intensity/duration
// MIT License – Autonomicity Games Inc. 2026

class MercyHapticEngine {
  constructor() {
    this.controllers = new Map(); // inputSource → gamepad
    this.valence = 1.0;
  }

  registerControllers(session) {
    session.addEventListener('inputsourceschange', e => {
      e.added.forEach(source => {
        if (source.gamepad) {
          this.controllers.set(source, source.gamepad);
          console.log("[MercyHaptic] Haptic controller registered");
        }
      });
      e.removed.forEach(source => this.controllers.delete(source));
    });
  }

  gateHaptic(eventType, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(eventType) || valence;
    if (degree < 0.9999999) {
      console.log("[MercyHaptic] Gate holds: low valence – haptic skipped");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Pattern presets
  playPattern(pattern = 'uplift', intensityMultiplier = 1.0) {
    const baseIntensity = 0.5 * this.valence * intensityMultiplier;

    const patterns = {
      uplift: [{ duration: 80, strong: baseIntensity, weak: baseIntensity * 0.7 }, { duration: 40, strong: 0, weak: 0 }, { duration: 120, strong: baseIntensity * 1.2, weak: baseIntensity }],
      calm: [{ duration: 200, strong: baseIntensity * 0.4, weak: baseIntensity * 0.6 }],
      thrivePulse: [{ duration: 50, strong: baseIntensity, weak: baseIntensity }, { duration: 20, strong: 0, weak: 0 }, { duration: 50, strong: baseIntensity * 0.8, weak: baseIntensity * 0.8 }]
    };

    const sequence = patterns[pattern] || patterns.uplift;

    this.controllers.forEach(gamepad => {
      if (gamepad?.hapticActuators) {
        let delay = 0;
        sequence.forEach(step => {
          setTimeout(() => {
            gamepad.hapticActuators[0]?.playEffect('dual-rumble', {
              duration: step.duration,
              strongMagnitude: step.strong || 0,
              weakMagnitude: step.weak || 0
            });
          }, delay);
          delay += step.duration + 10;
        });
        console.log(`[MercyHaptic] ${pattern} pattern played – valence ${this.valence.toFixed(8)}`);
      }
    });
  }
}

const mercyHaptic = new MercyHapticEngine();

export { mercyHaptic };

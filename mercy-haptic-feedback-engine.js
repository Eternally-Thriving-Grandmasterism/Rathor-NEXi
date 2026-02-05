// mercy-haptic-feedback-engine.js – v2 sovereign Mercy Haptic Feedback Engine
// Expanded pattern presets, dual-rumble sequences, mercy-gated, valence-modulated
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

  // Expanded pattern presets – each is array of {duration, strong, weak} steps
  getPattern(patternName = 'uplift') {
    const baseIntensity = Math.min(1.0, 0.4 + (this.valence - 0.999) * 2.5);

    const presets = {
      uplift: [                    // Joyful, rising energy
        { duration: 60, strong: baseIntensity * 0.6, weak: baseIntensity * 0.8 },
        { duration: 30, strong: 0, weak: 0 },
        { duration: 80, strong: baseIntensity * 1.0, weak: baseIntensity * 0.9 },
        { duration: 40, strong: baseIntensity * 0.4, weak: baseIntensity * 0.6 }
      ],

      calm: [                      // Gentle, soothing wave
        { duration: 250, strong: baseIntensity * 0.3, weak: baseIntensity * 0.5 },
        { duration: 150, strong: baseIntensity * 0.2, weak: baseIntensity * 0.4 },
        { duration: 200, strong: baseIntensity * 0.25, weak: baseIntensity * 0.45 }
      ],

      thrivePulse: [               // Rhythmic heartbeat of thriving
        { duration: 45, strong: baseIntensity * 0.8, weak: baseIntensity * 0.7 },
        { duration: 25, strong: 0, weak: 0 },
        { duration: 45, strong: baseIntensity * 0.9, weak: baseIntensity * 0.8 },
        { duration: 25, strong: 0, weak: 0 },
        { duration: 60, strong: baseIntensity * 0.7, weak: baseIntensity * 0.6 }
      ],

      abundanceSurge: [            // Building wave of infinite flow
        { duration: 100, strong: baseIntensity * 0.4, weak: baseIntensity * 0.6 },
        { duration: 80, strong: baseIntensity * 0.7, weak: baseIntensity * 0.9 },
        { duration: 120, strong: baseIntensity * 1.0, weak: baseIntensity * 0.8 },
        { duration: 60, strong: baseIntensity * 0.5, weak: baseIntensity * 0.7 }
      ],

      eternalReflection: [         // Slow, meditative fade
        { duration: 400, strong: baseIntensity * 0.2, weak: baseIntensity * 0.4 },
        { duration: 300, strong: baseIntensity * 0.15, weak: baseIntensity * 0.3 },
        { duration: 500, strong: baseIntensity * 0.1, weak: baseIntensity * 0.25 }
      ],

      compassionWave: [            // Warm, enveloping embrace
        { duration: 180, strong: baseIntensity * 0.5, weak: baseIntensity * 0.7 },
        { duration: 120, strong: baseIntensity * 0.4, weak: baseIntensity * 0.6 },
        { duration: 220, strong: baseIntensity * 0.6, weak: baseIntensity * 0.8 },
        { duration: 100, strong: baseIntensity * 0.3, weak: baseIntensity * 0.5 }
      ],

      cosmicHarmony: [             // Pulsing unity across realities
        { duration: 70, strong: baseIntensity * 0.7, weak: baseIntensity * 0.9 },
        { duration: 50, strong: 0, weak: 0 },
        { duration: 90, strong: baseIntensity * 0.85, weak: baseIntensity * 1.0 },
        { duration: 40, strong: 0, weak: 0 },
        { duration: 110, strong: baseIntensity * 0.6, weak: baseIntensity * 0.8 }
      ]
    };

    return presets[patternName] || presets.uplift;
  }

  // Play any pattern with optional intensity scaling
  playPattern(patternName = 'uplift', intensityMultiplier = 1.0) {
    const pattern = this.getPattern(patternName);
    const baseIntensity = Math.min(1.0, 0.4 + (this.valence - 0.999) * 2.5) * intensityMultiplier;

    this.controllers.forEach(gamepad => {
      if (gamepad?.hapticActuators) {
        let delay = 0;
        pattern.forEach(step => {
          setTimeout(() => {
            const strongMag = Math.min(1.0, (step.strong || 0) * baseIntensity);
            const weakMag = Math.min(1.0, (step.weak || 0) * baseIntensity);
            gamepad.hapticActuators[0]?.playEffect('dual-rumble', {
              duration: step.duration,
              strongMagnitude: strongMag,
              weakMagnitude: weakMag
            });
          }, delay);
          delay += step.duration + 10; // small buffer
        });
        console.log(`[MercyHaptic] ${patternName} pattern played – valence ${this.valence.toFixed(8)}, multiplier ${intensityMultiplier}`);
      }
    });
  }

  // Quick single-pulse convenience
  pulse(intensity = 0.5, durationMs = 100) {
    this.playPattern('uplift', intensity / 0.5); // normalize to uplift base
  }
}

const mercyHaptic = new MercyHapticEngine();

export { mercyHaptic };

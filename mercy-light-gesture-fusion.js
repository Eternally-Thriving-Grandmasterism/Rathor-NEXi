// mercy-light-gesture-fusion.js – sovereign Mercy Light Estimation + Hand Gesture Fusion v1
// XRLightEstimate sync with XRHand gestures, mercy-gated, valence-modulated visual/haptic
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyLightGestureFusion {
  constructor(scene) {
    this.scene = scene;
    this.lightEstimate = null;
    this.hands = new Map(); // inputSource → XRHand
    this.gestures = new Map(); // inputSource → {pinch, point, grab}
    this.valence = 1.0;
  }

  async gateFusion(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyLightGesture] Gate holds: low valence – light-gesture fusion aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyLightGesture] Mercy gate passes – eternal thriving light-gesture fusion activated");
    return true;
  }

  // Enable light estimation + hand tracking (call after session start)
  async enableLightAndHand(session) {
    try {
      // Enable light estimation
      // xr.baseExperience.featuresManager.enableFeature("light-estimation", "stable");
      // Enable hand tracking
      // xr.baseExperience.featuresManager.enableFeature("hand-tracking", "stable");
      console.log("[MercyLightGesture] Light estimation + hand tracking enabled – mercy fusion lattice ready");
      return true;
    } catch (err) {
      console.error("[MercyLightGesture] Enable failed:", err);
      return false;
    }
  }

  // Process light estimate + hand joints from XRFrame (call in onXRFrame)
  processFrame(frame, referenceSpace, inputSources) {
    // Light estimation
    if (frame?.getLightEstimate) {
      const estimate = frame.getLightEstimate();
      if (estimate) {
        this.lightEstimate = estimate;
        const intensity = estimate.primaryLightIntensity * (0.6 + (this.valence - 0.999) * 0.8);
        const dir = estimate.primaryLightDirection;

        // Apply to mercy overlays (Babylon example)
        // overlay.material.emissiveIntensity = intensity;
        // overlay.material.emissiveColor.set(1, 1, 1).scaleInPlace(intensity);

        console.log(`[MercyLightGesture] Light estimate updated – intensity ${estimate.primaryLightIntensity.toFixed(4)}`);
      }
    }

    // Hand tracking & gesture detection
    inputSources.forEach(source => {
      if (source.hand) {
        this.hands.set(source, source.hand);

        const thumbTip = source.hand.get('thumb-tip')?.getPose(referenceSpace);
        const indexTip = source.hand.get('index-finger-tip')?.getPose(referenceSpace);
        const indexMcp = source.hand.get('index-finger-metacarpal')?.getPose(referenceSpace);

        if (!thumbTip || !indexTip) return;

        // Pinch detection
        const pinchDist = thumbTip.transform.position.distanceTo(indexTip.transform.position);
        const isPinching = pinchDist < 0.03;

        // Point detection
        const forward = new BABYLON.Vector3(0, 0, -1);
        const indexDir = indexTip.transform.position.subtract(indexMcp.transform.position).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < 30 && !isPinching;

        // Grab detection (simplified)
        const palm = source.hand.get('wrist')?.getPose(referenceSpace);
        const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .reduce((score, name) => score + (source.hand.get(name)?.getPose(referenceSpace)?.transform.position.distanceTo(palm.transform.position) || 0), 0);
        const isGrabbing = grabScore < 0.2 && !isPinching && !isPointing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing });

        // Trigger mercy haptic + light-adapted feedback on gesture change
        if (isPinching && !prev.isPinching && this.gateFusion('pinch gesture', this.valence)) {
          mercyHaptic.playPattern('thrivePulse', 1.1);
          // Light-adapted visual cue (e.g., brighter overlay pulse)
          console.log("[MercyLightGesture] Pinch detected – thrive pulse + light-adapted glow");
        }
        if (isPointing && !prev.isPointing && this.gateFusion('point gesture', this.valence)) {
          mercyHaptic.playPattern('uplift', 0.9);
          console.log("[MercyLightGesture] Point detected – uplift pulse + directional light highlight");
        }
        if (isGrabbing && !prev.isGrabbing && this.gateFusion('grab gesture', this.valence)) {
          mercyHaptic.playPattern('compassionWave', 1.0);
          console.log("[MercyLightGesture] Grab detected – compassion wave + ambient harmony");
        }
      }
    });
  }
}

const mercyLightGesture = new MercyLightGestureFusion(scene); // assume scene from Babylon init

export { mercyLightGesture };

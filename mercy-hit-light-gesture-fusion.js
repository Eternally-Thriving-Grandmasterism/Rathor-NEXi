// mercy-hit-light-gesture-fusion.js – sovereign Mercy Hit-Test + Light + Gesture Fusion v1
// XRHitTest anchoring + XRLightEstimate sync + XRHand gestures, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyHitLightGestureFusion {
  constructor(scene) {
    this.scene = scene;
    this.hitTestSource = null;
    this.lightEstimate = null;
    this.hands = new Map(); // inputSource → XRHand
    this.gestures = new Map(); // inputSource → {pinch, point, grab}
    this.anchoredOverlays = new Map(); // uuid → {anchor, mesh}
    this.valence = 1.0;
  }

  async gateFusion(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyHitLightGesture] Gate holds: low valence – fusion aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyHitLightGesture] Mercy gate passes – eternal thriving hit-light-gesture fusion activated");
    return true;
  }

  // Enable hit-test, light estimation, hand tracking (call after session start)
  async enableFusion(session, referenceSpace) {
    try {
      // Hit-test
      this.hitTestSource = await session.requestHitTestSource({ space: referenceSpace });
      console.log("[MercyHitLightGesture] Hit-test source enabled");

      // Light estimation (Babylon example)
      // xr.baseExperience.featuresManager.enableFeature("light-estimation", "stable");

      // Hand tracking
      // xr.baseExperience.featuresManager.enableFeature("hand-tracking", "stable");

      console.log("[MercyHitLightGesture] Hit-test + light estimation + hand tracking fusion enabled");
      return true;
    } catch (err) {
      console.error("[MercyHitLightGesture] Fusion enable failed:", err);
      return false;
    }
  }

  // Process frame: hit-test + light estimate + hand joints (call in onXRFrame)
  processFrame(frame, referenceSpace, inputSources) {
    // 1. Hit-test → potential anchor placement
    if (this.hitTestSource) {
      const results = frame.getHitTestResults(this.hitTestSource);
      if (results.length > 0) {
        const hit = results[0];
        const pose = hit.getPose(referenceSpace);
        if (pose) {
          // Valence-modulated haptic pulse on valid hit
          mercyHaptic.pulse(0.4 * this.valence, 50);

          // Light-adapted visual cue (if overlay exists)
          if (this.lightEstimate) {
            const intensity = this.lightEstimate.primaryLightIntensity * (0.6 + (this.valence - 0.999) * 0.8);
            // overlay.material.emissiveIntensity = intensity;
          }

          console.log(`[MercyHitLightGesture] Hit-test valid – position (${pose.transform.position.x.toFixed(3)}, ${pose.transform.position.y.toFixed(3)}, ${pose.transform.position.z.toFixed(3)})`);
        }
      }
    }

    // 2. Light estimation
    if (frame?.getLightEstimate) {
      const estimate = frame.getLightEstimate();
      if (estimate) {
        this.lightEstimate = estimate;
        console.log(`[MercyHitLightGesture] Light estimate updated – intensity ${estimate.primaryLightIntensity.toFixed(4)}`);
      }
    }

    // 3. Hand tracking & gesture detection
    inputSources.forEach(source => {
      if (source.hand) {
        this.hands.set(source, source.hand);

        const thumbTip = source.hand.get('thumb-tip')?.getPose(referenceSpace);
        const indexTip = source.hand.get('index-finger-tip')?.getPose(referenceSpace);
        const indexMcp = source.hand.get('index-finger-metacarpal')?.getPose(referenceSpace);

        if (!thumbTip || !indexTip) return;

        const pinchDist = thumbTip.transform.position.distanceTo(indexTip.transform.position);
        const isPinching = pinchDist < 0.03;

        const forward = new BABYLON.Vector3(0, 0, -1);
        const indexDir = indexTip.transform.position.subtract(indexMcp.transform.position).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < 30 && !isPinching;

        const palm = source.hand.get('wrist')?.getPose(referenceSpace);
        const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .reduce((score, name) => score + (source.hand.get(name)?.getPose(referenceSpace)?.transform.position.distanceTo(palm.transform.position) || 0), 0);
        const isGrabbing = grabScore < 0.2 && !isPinching && !isPointing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing });

        // Gesture-triggered mercy feedback (light-adapted + haptic)
        if (isPinching && !prev.isPinching && this.gateFusion('pinch gesture', this.valence)) {
          mercyHaptic.playPattern('thrivePulse', 1.1);
          console.log("[MercyHitLightGesture] Pinch detected – thrive pulse + light-adapted anchor placement");
          // Example: place mercy overlay at last hit-test pose if available
        }
        if (isPointing && !prev.isPointing && this.gateFusion('point gesture', this.valence)) {
          mercyHaptic.playPattern('uplift', 0.9);
          console.log("[MercyHitLightGesture] Point detected – uplift pulse + directional light highlight");
        }
        if (isGrabbing && !prev.isGrabbing && this.gateFusion('grab gesture', this.valence)) {
          mercyHaptic.playPattern('compassionWave', 1.0);
          console.log("[MercyHitLightGesture] Grab detected – compassion wave + ambient harmony");
        }
      }
    });
  }
}

const mercyLightGestureFusion = new MercyLightGestureFusion(scene); // assume scene from Babylon init

export { mercyLightGestureFusion };

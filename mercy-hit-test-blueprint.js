// mercy-hit-test-blueprint.js – sovereign Mercy Hit-Test Blueprint v1
// WebXR hit-test (transient + persistent), mercy-gated placement, valence-modulated feedback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyHitTest {
  constructor() {
    this.hitTestSource = null;
    this.hitTestSourceRequested = false;
    this.persistentAnchors = new Map(); // id → anchor
    this.valence = 1.0;
  }

  async gateHitTest(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyHitTest] Gate holds: low valence – hit-test aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyHitTest] Mercy gate passes – eternal thriving hit-test activated");
    return true;
  }

  async requestHitTestSource(session, referenceSpace, persistent = false) {
    if (this.hitTestSourceRequested) return;

    try {
      const options = { space: referenceSpace };
      if (persistent) {
        this.hitTestSource = await session.requestHitTestSourceForTransientInput(options);
      } else {
        this.hitTestSource = await session.requestHitTestSource(options);
      }
      this.hitTestSourceRequested = true;
      console.log(`[MercyHitTest] ${persistent ? 'Persistent' : 'Transient'} hit-test source requested`);
      return true;
    } catch (err) {
      console.error("[MercyHitTest] Hit-test source request failed:", err);
      return false;
    }
  }

  // Call in XRFrame loop (onXRFrame)
  processHitTestResults(frame, referenceSpace, callback) {
    if (!this.hitTestSource || !frame) return;

    const results = frame.getHitTestResults(this.hitTestSource);
    if (results.length > 0) {
      const hit = results[0]; // best/first result
      const pose = hit.getPose(referenceSpace);
      if (pose) {
        const position = pose.transform.position;
        const orientation = pose.transform.orientation;

        // Valence-modulated feedback
        const intensity = Math.min(1.0, 0.4 + (this.valence - 0.999) * 2);
        mercyHaptic.pulse(intensity * 0.6, 60); // quick confirmation pulse

        // Callback for anchoring / overlay placement
        if (callback) callback({ position, orientation, hit });
      }
    }
  }

  // Persistent anchor creation (after user confirmation)
  async createPersistentAnchor(hitResult, referenceSpace) {
    try {
      const anchor = await hitResult.createAnchor(referenceSpace);
      const id = anchor.uuid || Date.now().toString();
      this.persistentAnchors.set(id, anchor);
      console.log(`[MercyHitTest] Persistent mercy anchor created – ID ${id}`);
      return id;
    } catch (err) {
      console.error("[MercyHitTest] Anchor creation failed:", err);
      return null;
    }
  }

  // Cleanup on session end
  cleanup() {
    this.hitTestSource = null;
    this.hitTestSourceRequested = false;
    this.persistentAnchors.clear();
  }
}

const mercyHitTest = new MercyHitTest();

export { mercyHitTest };

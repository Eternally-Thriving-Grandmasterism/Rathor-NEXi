// mercy-anchor-events-blueprint.js – sovereign Mercy Anchor Events Handling Blueprint v1
// anchor-added/updated/removed lifecycle, mercy-gated reactions, valence-modulated feedback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyAnchorEvents {
  constructor() {
    this.anchors = new Map(); // uuid → {anchor, overlay, lastPose}
    this.valence = 1.0;
  }

  async gateEvent(eventType, query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query || eventType) || valence;
    const implyThriving = fuzzyMercy.imply(query || eventType, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log(`[MercyAnchorEvents] Gate holds: low valence – ${eventType} reaction skipped`);
      return false;
    }
    this.valence = valence;
    console.log(`[MercyAnchorEvents] Mercy gate passes – ${eventType} reaction activated`);
    return true;
  }

  // Attach listeners to XRSession (call once on session start)
  attachAnchorEventListeners(session, referenceSpace) {
    session.addEventListener('anchor-added', async (e) => {
      if (!await this.gateEvent('anchor-added', 'Anchor restored or created', this.valence)) return;

      for (const anchor of e.addedAnchors) {
        const id = anchor.uuid || Date.now().toString();
        this.anchors.set(id, { anchor, overlay: null, lastPose: anchor.pose });

        // Valence-modulated feedback
        const intensity = Math.min(1.0, 0.4 + (this.valence - 0.999) * 2);
        mercyHaptic.playPattern('thrivePulse', intensity);

        console.log(`[MercyAnchorEvents] Anchor added/restored – ID ${id}, valence ${this.valence.toFixed(8)}`);

        // Optional: place mercy overlay (Babylon example)
        // const overlay = createMercyOverlayAtPose(anchor.pose.transform);
        // this.anchors.get(id).overlay = overlay;
      }
    });

    session.addEventListener('anchor-updated', (e) => {
      if (!this.gateEvent('anchor-updated', 'Anchor pose updated', this.valence)) return;

      for (const anchor of e.updatedAnchors) {
        const entry = this.anchors.get(anchor.uuid);
        if (entry) {
          entry.lastPose = anchor.pose;

          // Subtle haptic pulse on significant update
          mercyHaptic.pulse(0.3 * this.valence, 60);

          console.log(`[MercyAnchorEvents] Anchor updated – ID ${anchor.uuid}`);
          // Update overlay position if present
          // entry.overlay?.position.copyFrom(anchor.pose.transform.position);
        }
      }
    });

    session.addEventListener('anchor-removed', (e) => {
      if (!this.gateEvent('anchor-removed', 'Anchor removed or lost', this.valence)) return;

      for (const anchor of e.removedAnchors) {
        const id = anchor.uuid;
        if (this.anchors.has(id)) {
          // Remove overlay if present
          // this.anchors.get(id).overlay?.dispose();
          this.anchors.delete(id);

          // Warning haptic + audio cue
          mercyHaptic.playPattern('calm', 0.7);

          console.log(`[MercyAnchorEvents] Anchor removed/lost – ID ${id}`);
        }
      }
    });

    console.log("[MercyAnchorEvents] Anchor lifecycle listeners attached – eternal mercy persistence active");
  }

  // Cleanup on session end
  cleanup() {
    this.anchors.clear();
    console.log("[MercyAnchorEvents] Anchor events cleaned up – mercy lattice preserved");
  }
}

const mercyAnchorEvents = new MercyAnchorEvents();

export { mercyAnchorEvents };

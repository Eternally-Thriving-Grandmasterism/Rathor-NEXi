// mercy-hand-gesture-blueprint.js – sovereign Mercy Hand Gesture Detection Blueprint v1
// XRHand joint-based gestures (pinch/point/grab/open-palm/thumbs-up), mercy-gated, valence-modulated feedback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

// Thresholds (meters / degrees) – tunable
const PINCH_DISTANCE_THRESHOLD = 0.035;
const POINT_ANGLE_THRESHOLD = 35;
const GRAB_FINGER_PALM_DISTANCE = 0.18;
const OPEN_HAND_SPREAD_THRESHOLD = 0.25;
const THUMBS_UP_ANGLE_THRESHOLD = 60;

class MercyHandGesture {
  constructor() {
    this.hands = new Map(); // inputSource → XRHand
    this.gestures = new Map(); // inputSource → {pinch, point, grab, openPalm, thumbsUp}
    this.valence = 1.0;
  }

  async gateGesture(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyGesture] Gate holds: low valence – gesture detection aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyGesture] Mercy gate passes – eternal thriving gesture detection activated");
    return true;
  }

  // Process hand joints from XRFrame (call in onXRFrame)
  processHandJoints(frame, inputSources, referenceSpace) {
    inputSources.forEach(source => {
      if (source.hand) {
        this.hands.set(source, source.hand);

        const joints = {};
        for (const jointName of source.hand.jointNames) {
          const joint = source.hand.get(jointName);
          if (joint) {
            const pose = frame.getJointPose(joint, referenceSpace);
            if (pose) joints[jointName] = pose.transform;
          }
        }

        if (!joints['thumb-tip'] || !joints['index-finger-tip'] || !joints['wrist']) return;

        // Pinch: thumb-index tip distance
        const pinchDist = joints['thumb-tip'].position.distanceTo(joints['index-finger-tip'].position);
        const isPinching = pinchDist < PINCH_DISTANCE_THRESHOLD;

        // Point: index finger extended forward (angle to forward vector)
        const forward = new BABYLON.Vector3(0, 0, -1); // assume camera forward
        const indexDir = joints['index-finger-tip'].position.subtract(joints['index-finger-metacarpal'].position).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < POINT_ANGLE_THRESHOLD && !isPinching;

        // Grab: multiple finger tips close to wrist/palm
        const palm = joints['wrist'].position;
        const fingerTips = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip'];
        const grabScore = fingerTips.reduce((sum, name) => {
          const tip = joints[name];
          return sum + (tip ? tip.position.distanceTo(palm) : 0);
        }, 0) / fingerTips.length;
        const isGrabbing = grabScore < GRAB_FINGER_PALM_DISTANCE && !isPinching && !isPointing;

        // Open Palm: fingers spread (thumb-index distance high)
        const thumbIndexSpread = joints['thumb-tip'].position.distanceTo(joints['index-finger-tip'].position);
        const isOpenPalm = thumbIndexSpread > OPEN_HAND_SPREAD_THRESHOLD && !isGrabbing && !isPinching;

        // Thumbs Up: thumb extended upward, others curled
        const thumbUp = joints['thumb-tip'].position.y - palm.y > 0.1;
        const othersCurled = fingerTips.slice(1).every(name => {
          const tip = joints[name];
          return tip && tip.position.distanceTo(palm) < 0.12;
        });
        const isThumbsUp = thumbUp && othersCurled && !isGrabbing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing, isOpenPalm, isThumbsUp });

        // Trigger mercy feedback on gesture change
        if (isPinching && !prev.isPinching && this.gateGesture('pinch gesture', this.valence)) {
          mercyHaptic.playPattern('thrivePulse', 1.1);
          console.log("[MercyGesture] Pinch detected – thrive pulse + select action");
        }
        if (isPointing && !prev.isPointing && this.gateGesture('point gesture', this.valence)) {
          mercyHaptic.playPattern('uplift', 0.9);
          console.log("[MercyGesture] Point detected – uplift pulse + hover/highlight");
        }
        if (isGrabbing && !prev.isGrabbing && this.gateGesture('grab gesture', this.valence)) {
          mercyHaptic.playPattern('compassionWave', 1.0);
          console.log("[MercyGesture] Grab detected – compassion wave + hold/anchor");
        }
        if (isOpenPalm && !prev.isOpenPalm && this.gateGesture('open palm gesture', this.valence)) {
          mercyHaptic.playPattern('eternalReflection', 0.7);
          console.log("[MercyGesture] Open Palm detected – eternal reflection wave + release/clear");
        }
        if (isThumbsUp && !prev.isThumbsUp && this.gateGesture('thumbs up gesture', this.valence)) {
          mercyHaptic.playPattern('abundanceSurge', 1.2);
          console.log("[MercyGesture] Thumbs Up detected – abundance surge pulse + approval/activate");
        }
      }
    });
  }

  // Cleanup
  cleanup() {
    this.hands.clear();
    this.gestures.clear();
    console.log("[MercyGesture] Hand gesture tracking cleaned up – mercy lattice preserved");
  }
}

const mercyHandGesture = new MercyHandGesture();

export { mercyHandGesture };

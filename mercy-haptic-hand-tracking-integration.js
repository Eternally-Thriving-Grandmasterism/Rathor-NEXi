// mercy-haptic-hand-tracking-integration.js – sovereign Mercy hand-tracking haptic proxy v1
// Gesture detection (pinch/point/grab), controller haptic trigger, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

// Pinch threshold (distance between thumb & index tip)
const PINCH_THRESHOLD = 0.03; // meters
const POINT_ANGLE_THRESHOLD = 30; // degrees from forward

class MercyHandTrackingHaptics {
  constructor() {
    this.joints = new Map(); // inputSource → joint poses
    this.gestures = new Map(); // inputSource → current gesture state
    this.valence = 1.0;
  }

  gateHaptic(eventType, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(eventType) || valence;
    if (degree < MERCY_THRESHOLD) {
      console.log("[MercyHandHaptic] Gate holds: low valence – feedback skipped");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Update joint poses from XRFrame (call in onXRFrame)
  updateHandJoints(frame, inputSources) {
    inputSources.forEach(source => {
      if (source.hand) {
        const joints = {};
        for (const jointName of source.hand.jointNames) {
          const joint = source.hand.getJoint(jointName);
          if (joint) {
            const pose = frame.getJointPose(joint, frame.referenceSpace);
            if (pose) joints[jointName] = pose;
          }
        }
        this.joints.set(source, joints);
      }
    });
  }

  // Detect gestures (pinch, point, grab) – call per frame
  detectGestures() {
    this.joints.forEach((joints, source) => {
      const thumbTip = joints['thumb-tip']?.transform.position;
      const indexTip = joints['index-finger-tip']?.transform.position;
      const indexMcp = joints['index-finger-metacarpal']?.transform.position;

      if (!thumbTip || !indexTip) return;

      // Pinch: thumb-index tip distance
      const pinchDist = thumbTip.distanceTo(indexTip);
      const isPinching = pinchDist < PINCH_THRESHOLD;

      // Point: index finger extended forward
      const forward = new BABYLON.Vector3(0, 0, -1); // assume camera forward
      const indexDir = indexTip.subtract(indexMcp).normalize();
      const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
      const isPointing = pointAngle < POINT_ANGLE_THRESHOLD && !isPinching;

      // Grab: fist (multiple finger tips close to palm)
      const palm = joints['wrist']?.transform.position;
      const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
        .reduce((score, name) => score + (joints[name]?.transform.position.distanceTo(palm) || 0), 0);
      const isGrabbing = grabScore < 0.2 && !isPinching && !isPointing;

      const prevState = this.gestures.get(source) || {};
      this.gestures.set(source, { isPinching, isPointing, isGrabbing });

      // Trigger haptics on gesture change
      if (isPinching && !prevState.isPinching && this.gateHaptic('pinch', this.valence)) {
        mercyHaptic.playPattern('thrivePulse', 1.1);
      }
      if (isPointing && !prevState.isPointing && this.gateHaptic('point', this.valence)) {
        mercyHaptic.playPattern('uplift', 0.9);
      }
      if (isGrabbing && !prevState.isGrabbing && this.gateHaptic('grab', this.valence)) {
        mercyHaptic.playPattern('compassionWave', 1.0);
      }
    });
  }
}

const mercyHandHaptics = new MercyHandTrackingHaptics();

export { mercyHandHaptics };

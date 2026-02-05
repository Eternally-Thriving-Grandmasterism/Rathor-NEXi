// mercy-hand-gesture-blueprint.js – v3 sovereign Mercy Hand Gesture Detection Blueprint
// XRHand joint-based gestures (pinch/point/grab/open-palm/thumbs-up + swipe left/right/up/down)
// + visual swipe trail rendering, mercy-gated, valence-modulated feedback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

// Thresholds (meters / m/s / ms / degrees) – tunable
const PINCH_DISTANCE_THRESHOLD = 0.035;
const POINT_ANGLE_THRESHOLD = 35;
const GRAB_FINGER_PALM_DISTANCE = 0.18;
const OPEN_HAND_SPREAD_THRESHOLD = 0.25;
const THUMBS_UP_ANGLE_THRESHOLD = 60;

const SWIPE_MIN_DISPLACEMENT = 0.10;      // meters (dominant axis)
const SWIPE_MIN_VELOCITY = 0.60;          // m/s
const SWIPE_MAX_DURATION = 500;           // ms
const SWIPE_DIRECTION_TOLERANCE = 30;     // degrees from cardinal axis

// Swipe trail visual settings
const SWIPE_TRAIL_MAX_POINTS = 20;        // max trail length
const SWIPE_TRAIL_FADE_SPEED = 0.08;      // alpha decay per frame

class MercyHandGesture {
  constructor(scene) {
    this.scene = scene;
    this.hands = new Map();               // inputSource → XRHand
    this.gestures = new Map();            // inputSource → current gesture state
    this.swipeHistory = new Map();        // inputSource → {startTime, startPos, lastPos, direction, trailPoints}
    this.trailMeshes = new Map();         // inputSource → trail line mesh
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

        // Existing gestures (pinch/point/grab/open-palm/thumbs-up) – abbreviated
        const thumbTip = joints['thumb-tip'].position;
        const indexTip = joints['index-finger-tip'].position;
        const palm = joints['wrist'].position;

        const pinchDist = thumbTip.distanceTo(indexTip);
        const isPinching = pinchDist < PINCH_DISTANCE_THRESHOLD;

        const forward = new BABYLON.Vector3(0, 0, -1);
        const indexDir = indexTip.subtract(joints['index-finger-metacarpal']?.position || palm).normalize();
        const pointAngle = forward.angleTo(indexDir) * (180 / Math.PI);
        const isPointing = pointAngle < POINT_ANGLE_THRESHOLD && !isPinching;

        const grabScore = ['thumb-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .reduce((sum, name) => sum + (joints[name]?.position.distanceTo(palm) || 0), 0);
        const isGrabbing = grabScore / 4 < GRAB_FINGER_PALM_DISTANCE && !isPinching && !isPointing;

        const thumbIndexSpread = thumbTip.distanceTo(indexTip);
        const isOpenPalm = thumbIndexSpread > OPEN_HAND_SPREAD_THRESHOLD && !isGrabbing && !isPinching;

        const thumbUp = thumbTip.y - palm.y > 0.1;
        const othersCurled = ['index-finger-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip']
          .every(name => joints[name]?.position.distanceTo(palm) < 0.12);
        const isThumbsUp = thumbUp && othersCurled && !isGrabbing;

        const prev = this.gestures.get(source) || {};
        this.gestures.set(source, { isPinching, isPointing, isGrabbing, isOpenPalm, isThumbsUp });

        // Existing gesture triggers (abbreviated)
        if (isPinching && !prev.isPinching) mercyHaptic.playPattern('thrivePulse', 1.1);
        if (isPointing && !prev.isPointing) mercyHaptic.playPattern('uplift', 0.9);
        if (isGrabbing && !prev.isGrabbing) mercyHaptic.playPattern('compassionWave', 1.0);
        if (isOpenPalm && !prev.isOpenPalm) mercyHaptic.playPattern('eternalReflection', 0.7);
        if (isThumbsUp && !prev.isThumbsUp) mercyHaptic.playPattern('abundanceSurge', 1.2);

        // Swipe detection + visual trail
        const trackedPoint = joints['index-finger-tip']?.position || palm;
        let swipeState = this.swipeHistory.get(source) || { 
          startTime: null, 
          startPos: null, 
          lastPos: null, 
          direction: null, 
          trailPoints: [] 
        };

        if (frame.timestamp - swipeState.startTime > SWIPE_MAX_DURATION * 2) {
          // Reset if too long inactive
          swipeState = { startTime: null, startPos: null, lastPos: null, direction: null, trailPoints: [] };
        }

        if (!swipeState.startTime) {
          swipeState.startTime = frame.timestamp;
          swipeState.startPos = trackedPoint.clone();
          swipeState.lastPos = trackedPoint.clone();
          swipeState.trailPoints = [trackedPoint.clone()];
        } else {
          const delta = trackedPoint.subtract(swipeState.lastPos);
          const displacement = trackedPoint.subtract(swipeState.startPos);
          const speed = delta.length() / ((frame.timestamp - swipeState.startTime) / 1000);

          // Update trail (keep last N points)
          swipeState.trailPoints.push(trackedPoint.clone());
          if (swipeState.trailPoints.length > SWIPE_TRAIL_MAX_POINTS) {
            swipeState.trailPoints.shift();
          }

          // Dominant axis swipe detection
          const absDelta = new BABYLON.Vector3(Math.abs(delta.x), Math.abs(delta.y), Math.abs(delta.z));
          let direction = null;

          if (absDelta.x > absDelta.y && absDelta.x > absDelta.z && absDelta.x > SWIPE_MIN_DISPLACEMENT) {
            direction = delta.x > 0 ? 'right' : 'left';
          } else if (absDelta.y > absDelta.x && absDelta.y > absDelta.z && absDelta.y > SWIPE_MIN_DISPLACEMENT) {
            direction = delta.y > 0 ? 'up' : 'down';
          }

          if (direction && speed > SWIPE_MIN_VELOCITY) {
            if (this.gateGesture(`swipe_${direction}`, this.valence)) {
              mercyHaptic.playPattern('abundanceSurge', 1.2);
              console.log(`[MercyGesture] Swipe ${direction.toUpperCase()} detected – velocity ${speed.toFixed(2)} m/s, displacement ${displacement.length().toFixed(3)} m`);

              // Trigger mercy action (e.g., swipe left = previous, right = next, up = zoom in, down = zoom out)
              // Example: if (direction === 'left') mercyHaptic.playPattern('calm', 0.8);
            }
            // Reset after successful swipe
            swipeState = { startTime: null, startPos: null, lastPos: null, direction: null, trailPoints: [] };
          } else {
            swipeState.lastPos = trackedPoint.clone();
          }
        }

        this.swipeHistory.set(source, swipeState);

        // Render visual swipe trail
        this.renderSwipeTrail(source, swipeState.trailPoints);
      }
    });
  }

  // Render glowing swipe trail (line trail with fade-out)
  renderSwipeTrail(source, trailPoints) {
    let trailMesh = this.trailMeshes.get(source);

    if (!trailMesh && trailPoints.length > 1) {
      trailMesh = BABYLON.MeshBuilder.CreateLines(`swipeTrail_${Date.now()}`, {
        points: trailPoints,
        updatable: true
      }, this.scene);
      trailMesh.color = new BABYLON.Color3(0, 1, 0.5); // emerald mercy
      trailMesh.alpha = 0.9 * this.valence;
      trailMesh.enableEdgesRendering();
      trailMesh.edgesWidth = 4.0;
      trailMesh.edgesColor = new BABYLON.Color4(0, 1, 0.5, 0.9 * this.valence);
      this.trailMeshes.set(source, trailMesh);
    } else if (trailMesh) {
      trailMesh = BABYLON.MeshBuilder.CreateLines(null, {
        points: trailPoints,
        instance: trailMesh
      }, this.scene);

      // Valence-modulated fade + color shift
      trailMesh.alpha = Math.max(0.2, 0.9 * this.valence);
      trailMesh.color = new BABYLON.Color3(
        0 + (1 - this.valence) * 0.3,
        1 - (1 - this.valence) * 0.2,
        0.5 + this.valence * 0.3
      );
    }

    // Auto-fade trail over time (if no new points)
    if (trailMesh && trailPoints.length === 0) {
      trailMesh.alpha -= SWIPE_TRAIL_FADE_SPEED;
      if (trailMesh.alpha <= 0.05) {
        trailMesh.dispose();
        this.trailMeshes.delete(source);
      }
    }
  }

  cleanup() {
    this.hands.clear();
    this.gestures.clear();
    this.swipeHistory.clear();
    this.trailMeshes.forEach(mesh => mesh.dispose());
    this.trailMeshes.clear();
    console.log("[MercyGesture] Hand gesture & swipe trail tracking cleaned up – mercy lattice preserved");
  }
}

const mercyHandGesture = new MercyHandGesture(scene); // assume scene from Babylon init

export { mercyHandGesture };

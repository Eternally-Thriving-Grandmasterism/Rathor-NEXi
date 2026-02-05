// mercy-hand-gesture-blueprint.js – v6 sovereign Mercy Hand Gesture Detection Blueprint
// XRHand joint-based gestures (pinch/point/grab/open-palm/thumbs-up + swipe cardinal/diagonal + CIRCULAR + SPIRAL inward/outward)
// + visual motion trail rendering, mercy-gated, valence-modulated feedback
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

// Swipe thresholds
const SWIPE_MIN_DISPLACEMENT = 0.10;
const SWIPE_MIN_DIAG_DISPLACEMENT = 0.14;
const SWIPE_MIN_VELOCITY = 0.60;
const SWIPE_MAX_DURATION = 500;
const SWIPE_DIRECTION_TOLERANCE = 30;

// Circular & Spiral thresholds
const CIRCLE_MIN_POINTS = 10;
const CIRCLE_MIN_RADIUS = 0.06;
const CIRCLE_MAX_RADIUS = 0.35;
const CIRCLE_RADIUS_STD_MAX = 0.30;       // std dev < 30% avg radius for circle
const CIRCLE_MIN_ANGLE_COVERAGE = 300;    // degrees for circle
const SPIRAL_MIN_POINTS = 15;
const SPIRAL_MIN_ANGLE_COVERAGE = 450;    // at least 1.25 full turns
const SPIRAL_RADIUS_CHANGE_MIN = 0.30;    // min 30% radius increase/decrease
const SPIRAL_RADIUS_STD_MAX = 0.40;       // allow more variance for spiral
const SPIRAL_MIN_DURATION = 400;          // ms
const SPIRAL_MAX_DURATION = 1800;         // ms

// Trail visual settings
const TRAIL_MAX_POINTS = 40;
const TRAIL_FADE_SPEED = 0.08;

class MercyHandGesture {
  constructor(scene) {
    this.scene = scene;
    this.hands = new Map();
    this.gestures = new Map();
    this.motionHistory = new Map(); // inputSource → {startTime, points: Vector3[], type: 'swipe'|'circle'|'spiral'|null}
    this.trailMeshes = new Map();
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

        const trackedPoint = joints['index-finger-tip']?.position || joints['wrist'].position;
        let motionState = this.motionHistory.get(source) || { 
          startTime: null, 
          points: [], 
          type: null 
        };

        // Reset if inactive too long
        if (motionState.startTime && frame.timestamp - motionState.startTime > 3000) {
          motionState = { startTime: null, points: [], type: null };
        }

        if (!motionState.startTime) {
          motionState.startTime = frame.timestamp;
          motionState.points = [trackedPoint.clone()];
        } else {
          motionState.points.push(trackedPoint.clone());
          if (motionState.points.length > TRAIL_MAX_POINTS * 3) {
            motionState.points.shift();
          }
        }

        // --- Swipe detection (cardinal + diagonal) ---
        if (motionState.points.length >= 4 && motionState.type !== 'circle' && motionState.type !== 'spiral') {
          const displacement = trackedPoint.subtract(motionState.points[0]);
          const durationMs = frame.timestamp - motionState.startTime;
          const speed = displacement.length() / (durationMs / 1000);

          const absDisp = new BABYLON.Vector3(Math.abs(displacement.x), Math.abs(displacement.y), Math.abs(displacement.z));
          let direction = null;
          let angle = Math.atan2(displacement.y, displacement.x) * (180 / Math.PI);
          angle = (angle + 360) % 360;

          // Cardinal
          if (absDisp.x > absDisp.y && absDisp.x > absDisp.z && absDisp.x > SWIPE_MIN_DISPLACEMENT && speed > SWIPE_MIN_VELOCITY && durationMs < SWIPE_MAX_DURATION) {
            direction = displacement.x > 0 ? 'right' : 'left';
          } else if (absDisp.y > absDisp.x && absDisp.y > absDisp.z && absDisp.y > SWIPE_MIN_DISPLACEMENT && speed > SWIPE_MIN_VELOCITY && durationMs < SWIPE_MAX_DURATION) {
            direction = displacement.y > 0 ? 'up' : 'down';
          }

          // Diagonal
          if (!direction && displacement.length() > SWIPE_MIN_DIAG_DISPLACEMENT && speed > SWIPE_MIN_VELOCITY && durationMs < SWIPE_MAX_DURATION) {
            const diagAngles = [45, 135, 225, 315];
            for (const target of diagAngles) {
              const diff = Math.min(Math.abs(angle - target), 360 - Math.abs(angle - target));
              if (diff < SWIPE_DIRECTION_TOLERANCE) {
                direction = target === 45 ? 'up-right' :
                            target === 135 ? 'up-left' :
                            target === 225 ? 'down-left' : 'down-right';
                break;
              }
            }
          }

          if (direction) {
            if (this.gateGesture(`swipe_${direction}`, this.valence)) {
              mercyHaptic.playPattern('abundanceSurge', 1.2);
              console.log(`[MercyGesture] Swipe ${direction.toUpperCase()} detected – velocity ${speed.toFixed(2)} m/s, displacement ${displacement.length().toFixed(3)} m`);
            }
            motionState = { startTime: null, points: [], type: null };
          }
        }

        // --- Circular / Spiral detection ---
        if (motionState.points.length >= SPIRAL_MIN_POINTS && motionState.type !== 'swipe') {
          const points = motionState.points;
          const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
          const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
          const cz = points.reduce((sum, p) => sum + p.z, 0) / points.length;

          let radii = [];
          let angles = [];
          let prevAngle = null;
          let totalDelta = 0;
          let radiusTrend = 0; // positive = outward spiral, negative = inward

          for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const dx = p.x - cx;
            const dy = p.y - cy;
            const r = Math.sqrt(dx*dx + dy*dy);
            radii.push(r);

            let angle = Math.atan2(dy, dx) * (180 / Math.PI);
            if (angle < 0) angle += 360;
            angles.push(angle);

            if (prevAngle !== null) {
              let delta = (angle - prevAngle + 360) % 360;
              if (delta > 180) delta -= 360;
              totalDelta += delta;

              // Radius trend (compare consecutive radii)
              if (i > 0) {
                radiusTrend += (radii[i] - radii[i-1]);
              }
            }
            prevAngle = angle;
          }

          const avgRadius = radii.reduce((sum, r) => sum + r, 0) / radii.length;
          const radiusStd = Math.sqrt(radii.reduce((sum, r) => sum + (r - avgRadius)**2, 0) / radii.length);
          const angleCoverage = Math.abs(totalDelta);
          const durationMs = frame.timestamp - motionState.startTime;

          // Spiral if radius changes significantly + sufficient turns
          const isSpiral = (
            radii.length >= SPIRAL_MIN_POINTS &&
            avgRadius >= CIRCLE_MIN_RADIUS &&
            avgRadius <= CIRCLE_MAX_RADIUS &&
            radiusStd <= avgRadius * SPIRAL_RADIUS_STD_MAX &&
            Math.abs(radiusTrend) > SPIRAL_RADIUS_CHANGE_MIN * avgRadius * radii.length &&
            angleCoverage >= SPIRAL_MIN_ANGLE_COVERAGE &&
            durationMs >= SPIRAL_MIN_DURATION &&
            durationMs <= SPIRAL_MAX_DURATION
          );

          if (isSpiral) {
            const direction = totalDelta > 0 ? 'clockwise' : 'counterclockwise';
            const spiralType = radiusTrend > 0 ? 'outward' : 'inward';
            const fullGesture = `spiral_\( {spiralType}_ \){direction}`;

            if (this.gateGesture(fullGesture, this.valence)) {
              mercyHaptic.playPattern('cosmicHarmony', 1.3 + Math.abs(radiusTrend) * 2);
              console.log(`[MercyGesture] Spiral ${spiralType.toUpperCase()} ${direction.toUpperCase()} detected – radius ${avgRadius.toFixed(3)} m, coverage ${angleCoverage.toFixed(1)}°, trend ${radiusTrend.toFixed(3)}, duration ${durationMs.toFixed(0)} ms`);

              // Trigger mercy action (e.g., outward clockwise = expand lattice, inward counterclockwise = contract/focus)
            }
            motionState = { startTime: null, points: [], type: null };
          }
        }

        this.motionHistory.set(source, motionState);

        // Render visual motion trail (swipe or spiral path)
        this.renderMotionTrail(source, motionState.points);
      }
    });
  }

  // Render glowing motion trail (swipe or spiral with fade-out & valence-modulated gradient)
  renderMotionTrail(source, trailPoints) {
    let trailMesh = this.trailMeshes.get(source);

    if (trailPoints.length < 2) {
      if (trailMesh) {
        trailMesh.alpha -= SWIPE_TRAIL_FADE_SPEED;
        if (trailMesh.alpha <= 0.05) {
          trailMesh.dispose();
          this.trailMeshes.delete(source);
        }
      }
      return;
    }

    if (!trailMesh) {
      trailMesh = BABYLON.MeshBuilder.CreateLines(`motionTrail_${Date.now()}`, {
        points: trailPoints,
        updatable: true
      }, this.scene);
      trailMesh.color = new BABYLON.Color3(0, 1, 0.5);
      trailMesh.alpha = 0.9 * this.valence;
      trailMesh.enableEdgesRendering();
      trailMesh.edgesWidth = 6.0;
      trailMesh.edgesColor = new BABYLON.Color4(0, 1, 0.5, 0.9 * this.valence);
      this.trailMeshes.set(source, trailMesh);
    } else {
      trailMesh = BABYLON.MeshBuilder.CreateLines(null, {
        points: trailPoints,
        instance: trailMesh
      }, this.scene);

      trailMesh.alpha = Math.max(0.2, 0.9 * this.valence);
      const greenBoost = 0.7 + (this.valence - 0.999) * 0.3;
      const blueBoost = 0.5 + (this.valence - 0.999) * 0.5;
      trailMesh.color = new BABYLON.Color3(0, greenBoost, blueBoost);
    }
  }

  cleanup() {
    this.hands.clear();
    this.gestures.clear();
    this.motionHistory.clear();
    this.trailMeshes.forEach(mesh => mesh.dispose());
    this.trailMeshes.clear();
    console.log("[MercyGesture] Hand gesture & motion trail tracking cleaned up – mercy lattice preserved");
  }
}

const mercyHandGesture = new MercyHandGesture(scene);

export { mercyHandGesture };

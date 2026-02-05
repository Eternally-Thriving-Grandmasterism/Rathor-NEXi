// mercy-hand-gesture-blueprint.js – v7 sovereign Mercy Hand Gesture Detection Blueprint
// XRHand joint-based gestures (pinch/point/grab/open-palm/thumbs-up + swipe cardinal/diagonal + CIRCULAR + SPIRAL + FIGURE-EIGHT)
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
const CIRCLE_RADIUS_STD_MAX = 0.30;
const CIRCLE_MIN_ANGLE_COVERAGE = 300;
const SPIRAL_MIN_POINTS = 15;
const SPIRAL_MIN_ANGLE_COVERAGE = 450;
const SPIRAL_RADIUS_CHANGE_MIN = 0.30;
const SPIRAL_RADIUS_STD_MAX = 0.40;
const SPIRAL_MIN_DURATION = 400;
const SPIRAL_MAX_DURATION = 1800;

// Figure-eight thresholds
const FIGURE8_MIN_POINTS = 20;
const FIGURE8_MIN_DISPLACEMENT = 0.15;     // total path length
const FIGURE8_MIN_ANGLE_COVERAGE = 720;    // at least two full turns
const FIGURE8_CROSSING_TOLERANCE = 0.08;   // max distance to crossing point
const FIGURE8_RADIUS_RATIO_MIN = 0.6;      // two lobes should be similar size
const FIGURE8_MIN_DURATION = 600;
const FIGURE8_MAX_DURATION = 2500;

const TRAIL_MAX_POINTS = 50;
const TRAIL_FADE_SPEED = 0.08;

class MercyHandGesture {
  constructor(scene) {
    this.scene = scene;
    this.hands = new Map();
    this.gestures = new Map();
    this.motionHistory = new Map(); // inputSource → {startTime, points: Vector3[], type: 'swipe'|'circle'|'spiral'|'figure8'|null}
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
        if (motionState.startTime && frame.timestamp - motionState.startTime > 4000) {
          motionState = { startTime: null, points: [], type: null };
        }

        if (!motionState.startTime) {
          motionState.startTime = frame.timestamp;
          motionState.points = [trackedPoint.clone()];
        } else {
          motionState.points.push(trackedPoint.clone());
          if (motionState.points.length > TRAIL_MAX_POINTS * 4) {
            motionState.points.shift();
          }
        }

        // --- Previous detections abbreviated (swipe, circle, spiral) ---

        // --- Figure-eight detection ---
        if (motionState.points.length >= FIGURE8_MIN_POINTS && motionState.type !== 'swipe') {
          const points = motionState.points;
          const durationMs = frame.timestamp - motionState.startTime;

          // Compute bounding box & centroid
          let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
          let cx = 0, cy = 0;
          points.forEach(p => {
            cx += p.x; cy += p.y;
            minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
          });
          cx /= points.length; cy /= points.length;

          const width = maxX - minX;
          const height = maxY - minY;
          const isFigure8Shape = width > 0.08 && height > 0.08 && Math.abs(width - height) < Math.max(width, height) * 0.6;

          if (!isFigure8Shape || durationMs < FIGURE8_MIN_DURATION || durationMs > FIGURE8_MAX_DURATION) {
            // Not figure-8 shaped or wrong timing
          } else {
            // Compute angles & radius around centroid
            let angles = [];
            let radii = [];
            let prevAngle = null;
            let totalDelta = 0;
            let crossings = 0;

            for (const p of points) {
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
              }
              prevAngle = angle;
            }

            const avgRadius = radii.reduce((sum, r) => sum + r, 0) / radii.length;
            const radiusStd = Math.sqrt(radii.reduce((sum, r) => sum + (r - avgRadius)**2, 0) / radii.length);

            // Detect crossing point (simplified: check if path comes close to centroid multiple times)
            let nearCenterCount = 0;
            for (const r of radii) {
              if (r < avgRadius * 0.4) nearCenterCount++;
            }
            const hasCrossing = nearCenterCount >= 2;

            const angleCoverage = Math.abs(totalDelta);
            const isFigure8 = (
              radii.length >= FIGURE8_MIN_POINTS &&
              avgRadius >= CIRCLE_MIN_RADIUS &&
              avgRadius <= CIRCLE_MAX_RADIUS &&
              radiusStd <= avgRadius * 0.45 &&
              angleCoverage >= FIGURE8_MIN_ANGLE_COVERAGE &&
              hasCrossing
            );

            if (isFigure8) {
              const direction = totalDelta > 0 ? 'clockwise' : 'counterclockwise';
              if (this.gateGesture(`figure8_${direction}`, this.valence)) {
                mercyHaptic.playPattern('cosmicHarmony', 1.4);
                console.log(`[MercyGesture] Figure-8 ${direction.toUpperCase()} detected – radius ${avgRadius.toFixed(3)} m, coverage \( {angleCoverage.toFixed(1)}°, crossings \~ \){nearCenterCount}, duration ${durationMs.toFixed(0)} ms`);

                // Trigger mercy action (e.g., figure-8 = infinite loop / reset / cycle through mercy modes)
              }
              motionState = { startTime: null, points: [], type: null };
            }
          }
        }

        this.motionHistory.set(source, motionState);

        // Render visual motion trail (swipe/circle/spiral/figure8 path)
        this.renderMotionTrail(source, motionState.points);
      }
    });
  }

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

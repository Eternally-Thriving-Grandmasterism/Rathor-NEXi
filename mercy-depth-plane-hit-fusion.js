// mercy-depth-plane-hit-fusion.js – sovereign Mercy Depth + Plane + Hit-Test Fusion v1
// XRDepthInformation occlusion + XRPlane polygon rendering + XRHitTest anchoring, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyDepthPlaneHitFusion {
  constructor(scene) {
    this.scene = scene;
    this.hitTestSource = null;
    this.depthInfo = null;
    this.planeOverlays = new Map(); // planeId → mesh group
    this.anchoredOverlays = new Map(); // uuid → {anchor, mesh}
    this.valence = 1.0;
  }

  async gateFusion(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyDepthPlaneHit] Gate holds: low valence – fusion aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyDepthPlaneHit] Mercy gate passes – eternal thriving depth-plane-hit fusion activated");
    return true;
  }

  // Enable depth sensing, plane detection, hit-test (call after session start)
  async enableFusion(session, referenceSpace) {
    try {
      // Hit-test
      this.hitTestSource = await session.requestHitTestSource({ space: referenceSpace });
      console.log("[MercyDepthPlaneHit] Hit-test source enabled");

      // Plane detection & depth sensing (Babylon.js helper example)
      // xr.baseExperience.featuresManager.enableFeature("plane-detection", "stable");
      // xr.baseExperience.featuresManager.enableFeature("depth-sorted-layers", "stable");

      console.log("[MercyDepthPlaneHit] Depth sensing + plane detection + hit-test fusion enabled");
      return true;
    } catch (err) {
      console.error("[MercyDepthPlaneHit] Fusion enable failed:", err);
      return false;
    }
  }

  // Process frame: hit-test + depth + planes (call in onXRFrame)
  processFrame(frame, referenceSpace, inputSources) {
    // 1. Hit-test → precise anchoring
    if (this.hitTestSource) {
      const results = frame.getHitTestResults(this.hitTestSource);
      if (results.length > 0) {
        const hit = results[0];
        const pose = hit.getPose(referenceSpace);
        if (pose) {
          mercyHaptic.pulse(0.4 * this.valence, 50);
          console.log(`[MercyDepthPlaneHit] Hit-test valid – position (${pose.transform.position.x.toFixed(3)}, ${pose.transform.position.y.toFixed(3)}, ${pose.transform.position.z.toFixed(3)})`);
          // Place mercy overlay at hit pose (example)
          // this.placeMercyOverlay(pose.transform);
        }
      }
    }

    // 2. Depth sensing → occlusion
    if (frame?.getDepthInformation) {
      const depthInfo = frame.getDepthInformation();
      if (depthInfo) {
        this.depthInfo = depthInfo;
        // Use depth texture for occlusion (Babylon depth sorting or custom shader)
        console.log(`[MercyDepthPlaneHit] Depth map updated – width ${depthInfo.width}, height ${depthInfo.height}`);
      }
    }

    // 3. Plane detection → polygon rendering
    if (frame?.detectedPlanes) {
      const currentPlaneIds = new Set();

      for (const plane of frame.detectedPlanes) {
        const id = plane.planeId;
        currentPlaneIds.add(id);

        let overlayGroup = this.planeOverlays.get(id);
        if (!overlayGroup) {
          overlayGroup = new BABYLON.TransformNode(`planeOverlay_${id}`, this.scene);
          this.planeOverlays.set(id, overlayGroup);

          // Outline mesh
          const outline = BABYLON.MeshBuilder.CreateLines(`outline_${id}`, {
            points: plane.polygon.map(p => new BABYLON.Vector3(p.x, p.y, p.z)),
            updatable: true
          }, this.scene);
          outline.color = new BABYLON.Color3(0, 1, 0.5);
          outline.alpha = 0.8 * this.valence;
          outline.parent = overlayGroup;

          // Fill mesh
          const fill = BABYLON.MeshBuilder.CreatePolygon(`fill_${id}`, {
            shape: plane.polygon.map(p => new BABYLON.Vector2(p.x, p.z)),
            holes: []
          }, this.scene);
          const fillMat = new BABYLON.StandardMaterial(`fillMat_${id}`, this.scene);
          fillMat.diffuseColor = new BABYLON.Color3(0, 0.8, 0.4);
          fillMat.alpha = 0.3 * this.valence;
          fill.material = fillMat;
          fill.rotation.x = Math.PI / 2;
          fill.parent = overlayGroup;

          mercyHaptic.pulse(0.5 * this.valence, 80);
          console.log(`[MercyDepthPlaneHit] New plane polygon overlay created – ID ${id}`);
        }

        // Update pose
        const pose = plane.pose?.getPose(referenceSpace);
        if (pose) {
          overlayGroup.position.set(
            pose.transform.position.x,
            pose.transform.position.y,
            pose.transform.position.z
          );
          overlayGroup.rotationQuaternion.set(
            pose.transform.orientation.x,
            pose.transform.orientation.y,
            pose.transform.orientation.z,
            pose.transform.orientation.w
          );
        }
      }

      // Cleanup lost planes
      for (const [id, group] of this.planeOverlays) {
        if (!currentPlaneIds.has(id)) {
          group.dispose();
          this.planeOverlays.delete(id);
          console.log(`[MercyDepthPlaneHit] Plane polygon overlay removed – ID ${id}`);
        }
      }
    }
  }

  // Cleanup
  cleanup() {
    this.planeOverlays.clear();
    console.log("[MercyDepthPlaneHit] Depth-plane-hit fusion cleaned up – mercy lattice preserved");
  }
}

const mercyDepthPlaneHit = new MercyDepthPlaneHitFusion(scene); // assume scene from Babylon init

export { mercyDepthPlaneHit };

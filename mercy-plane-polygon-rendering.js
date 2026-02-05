// mercy-plane-polygon-rendering.js – sovereign Mercy plane polygon rendering v1
// XRPlane.polygon outline/highlight mesh, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyPlanePolygonRenderer {
  constructor(scene) {
    this.scene = scene;
    this.planeOverlays = new Map(); // planeId → mesh group
    this.valence = 1.0;
  }

  async gateRendering(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyPlanePoly] Gate holds: low valence – polygon rendering aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyPlanePoly] Mercy gate passes – eternal thriving polygon rendering activated");
    return true;
  }

  // Create or update glowing outline/highlight for plane polygon
  renderPlanePolygon(plane, referenceSpace) {
    const id = plane.planeId;
    let overlayGroup = this.planeOverlays.get(id);

    if (!overlayGroup) {
      overlayGroup = new BABYLON.TransformNode(`planeOverlay_${id}`, this.scene);
      this.planeOverlays.set(id, overlayGroup);

      // Outline mesh (lines)
      const outline = BABYLON.MeshBuilder.CreateLines(`outline_${id}`, {
        points: plane.polygon.map(p => new BABYLON.Vector3(p.x, p.y, p.z)),
        updatable: true
      }, this.scene);
      outline.color = new BABYLON.Color3(0, 1, 0.5);
      outline.alpha = 0.8 * this.valence;
      outline.parent = overlayGroup;

      // Highlight fill (semi-transparent plane)
      const fill = BABYLON.MeshBuilder.CreatePolygon(`fill_${id}`, {
        shape: plane.polygon.map(p => new BABYLON.Vector2(p.x, p.z)),
        holes: []
      }, this.scene);
      const fillMat = new BABYLON.StandardMaterial(`fillMat_${id}`, this.scene);
      fillMat.diffuseColor = new BABYLON.Color3(0, 0.8, 0.4);
      fillMat.alpha = 0.3 * this.valence;
      fill.material = fillMat;
      fill.rotation.x = Math.PI / 2; // lay flat
      fill.parent = overlayGroup;

      // Haptic pulse on new plane polygon
      mercyHaptic.pulse(0.5 * this.valence, 80);

      console.log(`[MercyPlanePoly] New polygon overlay created – plane ID ${id}`);
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
  cleanupLostPlanes(currentPlaneIds) {
    for (const [id, group] of this.planeOverlays) {
      if (!currentPlaneIds.has(id)) {
        group.dispose();
        this.planeOverlays.delete(id);
        console.log(`[MercyPlanePoly] Polygon overlay removed – plane ID ${id}`);
      }
    }
  }
}

const mercyPlanePoly = new MercyPlanePolygonRenderer(scene); // assume scene from Babylon init

export { mercyPlanePoly };

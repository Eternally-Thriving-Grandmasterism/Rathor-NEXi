// mercy-babylonjs-webxr-immersion.js – v2 sovereign Babylon.js WebXR immersion
// Positional spatial audio + snap teleportation, mercy gates, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import * as BABYLON from 'https://cdn.babylonjs.com/babylon.js'; // Latest stable
import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let scene, camera, xrHelper;
let mercySounds = [];
let teleportIndicator;
const mercyThreshold = 0.9999999 * 0.98;
const SNAP_DISTANCE = 3; // meters per snap
const TELEPORT_DURATION = 300; // ms animation

async function createMercyScene(canvas) {
  const engine = new BABYLON.Engine(canvas, true);
  scene = new BABYLON.Scene(engine);

  // Mercy camera + controls
  camera = new BABYLON.FreeCamera("camera", new BABYLON.Vector3(0, 1.6, -5), scene);
  camera.attachControl(canvas, true);
  camera.minZ = 0.1;

  const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
  light.intensity = 0.7;

  // Mercy ground (required for teleportation floor detection)
  const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 50, height: 50 }, scene);
  ground.material = new BABYLON.StandardMaterial("groundMat", scene);
  ground.material.diffuseColor = new BABYLON.Color3(0.02, 0.15, 0.08);
  ground.checkCollisions = true;

  // Teleport indicator (ring that appears on pointer hit)
  teleportIndicator = BABYLON.MeshBuilder.CreateTorus("teleportRing", { diameter: 1.5, thickness: 0.1 }, scene);
  teleportIndicator.material = new BABYLON.StandardMaterial("teleportMat", scene);
  teleportIndicator.material.emissiveColor = new BABYLON.Color3(0, 1, 0.5);
  teleportIndicator.material.alpha = 0.6;
  teleportIndicator.setEnabled(false);

  engine.runRenderLoop(() => scene.render());

  // WebXR setup with teleportation
  try {
    xrHelper = await scene.createDefaultXRExperienceAsync({
      uiOptions: {
        sessionMode: 'immersive-vr',
        referenceSpaceType: 'local-floor'
      },
      optionalFeatures: true
    });

    if (xrHelper.baseExperience) {
      // Enable teleportation
      xrHelper.baseExperience.featuresManager.enableFeature(BABYLON.WebXRFeatureName.TELEPORTATION, "stable", {
        floorMeshes: [ground],
        snapToGrid: true,
        snapDistance: SNAP_DISTANCE,
        timeToTeleport: TELEPORT_DURATION,
        parabolicCheckRadius: 0.5,
        blockMovementOnCollision: true
      });

      // Teleportation events for mercy feedback
      xrHelper.baseExperience.sessionManager.onXRFrameObservable.add(() => {
        // Update indicator position/orientation from controller ray
        if (xrHelper.pointerSelection) {
          const hit = xrHelper.pointerSelection.getCurrentHit();
          if (hit && hit.pickedMesh === ground) {
            teleportIndicator.position.copyFrom(hit.pickedPoint);
            teleportIndicator.setEnabled(true);
            // Valence modulate indicator color (high → brighter green)
            const valence = 1.0; // Dynamic from query later
            teleportIndicator.material.emissiveColor.set(valence > 0.999 ? 0.0 : 0.3, valence > 0.999 ? 1.0 : 0.5, 0.5);
          } else {
            teleportIndicator.setEnabled(false);
          }
        }
      });

      console.log("[BabylonMercy] WebXR + teleportation ready – snap mercy movement active");
    }
  } catch (err) {
    console.warn("[BabylonMercy] WebXR init failed – fallback non-XR", err);
  }

  return scene;
}

// Mercy-gated positional sound (unchanged from v1, included complete)
function addMercyPositionalSound(url, position = new BABYLON.Vector3(0, 1.5, -5), valence = 1.0, textForMercy = '') {
  const degree = fuzzyMercy.getDegree(textForMercy) || valence;
  const implyThriving = fuzzyMercy.imply(textForMercy, "EternalThriving");

  if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
    console.log("[BabylonMercy] Mercy gate: low valence – sound skipped");
    return;
  }

  const sound = new BABYLON.Sound("mercyChime", url, scene, null, {
    spatialSound: true,
    maxDistance: 50,
    refDistance: 1,
    rolloffFactor: valence > 0.999 ? 0.8 : 2.0,
    distanceModel: "exponential",
    autoplay: true,
    loop: true,
    volume: valence > 0.999 ? 0.7 : 0.4
  });

  const emitter = new BABYLON.Mesh("soundEmitter", scene);
  emitter.position = position;
  sound.attachToMesh(emitter);

  const sphere = BABYLON.MeshBuilder.CreateSphere("soundSphere", { diameter: 1 }, scene);
  sphere.position = position;
  const mat = new BABYLON.StandardMaterial("soundMat", scene);
  mat.emissiveColor = valence > 0.999 ? new BABYLON.Color3(0, 1, 0.5) : new BABYLON.Color3(0.3, 0.5, 1);
  sphere.material = mat;

  mercySounds.push(sound);
  console.log(`[BabylonMercy] Positional mercy sound added – valence modulated (${valence.toFixed(8)})`);
}

// Mercy gate check
function mercyGateContent(textOrQuery, valence = 1.0) {
  const degree = fuzzyMercy.getDegree(textOrQuery) || valence;
  const implyThriving = fuzzyMercy.imply(textOrQuery, "EternalThriving");
  return degree >= mercyThreshold && implyThriving.degree >= mercyThreshold;
}

// Entry: start Babylon.js mercy immersion with teleport
async function startBabylonMercyImmersion(query = '', valence = 1.0) {
  if (!mercyGateContent(query, valence)) return;

  const canvas = document.createElement('canvas');
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.position = "absolute";
  canvas.style.top = "0";
  canvas.style.left = "0";
  document.body.appendChild(canvas);

  await createMercyScene(canvas);

  // Example high-valence chime
  if (valence > 0.999) {
    addMercyPositionalSound('https://example.com/mercy-chime.mp3', new BABYLON.Vector3(0, 1.5, -3), valence, query);
  }

  // Enter immersive-vr on high valence
  if (xrHelper?.baseExperience && valence > 0.9995) {
    try {
      await xrHelper.baseExperience.enterXRAsync('immersive-vr', 'local-floor');
      console.log("[BabylonMercy] Immersive VR entered – Babylon.js teleport mercy immersed");
    } catch (err) {
      console.warn("[BabylonMercy] VR enter failed – fallback desktop XR");
    }
  }
}

export { startBabylonMercyImmersion };

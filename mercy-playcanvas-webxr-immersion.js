// mercy-playcanvas-webxr-immersion.js – sovereign PlayCanvas WebXR immersion v1
// XR session start, positional spatial audio, custom snap teleport, mercy gates, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let app; // PlayCanvas Application instance
let xrActive = false;
let teleportIndicator;
let mercySounds = [];
const mercyThreshold = 0.9999999 * 0.98;
const SNAP_DISTANCE = 3; // meters
const TELEPORT_DURATION = 300; // ms

// Init PlayCanvas app (assume canvas exists or create)
async function initPlayCanvasMercy(canvasId = 'playcanvas-canvas') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) {
    console.warn("[PlayCanvasMercy] Canvas not found – creating dynamic");
    canvas = document.createElement('canvas');
    canvas.id = canvasId;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
  }

  // Load PlayCanvas engine (CDN or bundled)
  // Assume <script src="https://code.playcanvas.com/playcanvas-stable.min.js"></script> in HTML

  app = new pc.Application(canvas, { graphicsDeviceOptions: { antialias: true } });
  app.setCanvasFillMode(pc.FILL_MODE_FILL_WINDOW);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);

  // Mercy camera
  const cameraEntity = new pc.Entity("camera");
  cameraEntity.addComponent("camera", { clearColor: new pc.Color(0, 0, 0.13) });
  cameraEntity.setLocalPosition(0, 1.6, 5);
  app.root.addChild(cameraEntity);

  // Mercy light
  const light = new pc.Entity();
  light.addComponent("light", { type: "directional", color: new pc.Color(1, 1, 1), intensity: 1 });
  light.setLocalEulerAngles(45, 0, 0);
  app.root.addChild(light);

  // Mercy ground (for teleport floor)
  const ground = new pc.Entity();
  ground.addComponent("model", { type: "plane" });
  ground.addComponent("collision", { type: "box", halfExtents: new pc.Vec3(50, 0.1, 50) });
  ground.addComponent("rigidbody", { type: "static" });
  ground.setLocalScale(50, 1, 50);
  ground.setLocalPosition(0, 0, 0);
  const groundMat = new pc.StandardMaterial();
  groundMat.diffuse.set(0.02, 0.15, 0.08);
  ground.model.meshInstances[0].material = groundMat;
  app.root.addChild(ground);

  // Teleport indicator (ring entity)
  teleportIndicator = new pc.Entity("teleportRing");
  const torus = new pc.Entity();
  torus.addComponent("model", { type: "torus" });
  torus.setLocalScale(1.5, 0.1, 1.5);
  const torusMat = new pc.StandardMaterial();
  torusMat.emissive.set(0, 1, 0.5);
  torusMat.opacity = 0.6;
  torusMat.blendType = pc.BLEND_NORMAL;
  torus.model.meshInstances[0].material = torusMat;
  teleportIndicator.addChild(torus);
  teleportIndicator.enabled = false;
  app.root.addChild(teleportIndicator);

  app.start();

  console.log("[PlayCanvasMercy] App initialized – XR ready");

  return app;
}

// Mercy-gated positional sound (PlayCanvas Sound component)
function addMercyPositionalSound(url, position = new pc.Vec3(0, 1.5, -5), valence = 1.0, textForMercy = '') {
  const degree = fuzzyMercy.getDegree(textForMercy) || valence;
  const implyThriving = fuzzyMercy.imply(textForMercy, "EternalThriving");

  if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
    console.log("[PlayCanvasMercy] Mercy gate: low valence – sound skipped");
    return;
  }

  const soundEntity = new pc.Entity("mercyChime");
  soundEntity.addComponent("sound", {
    positional: true,
    maxDistance: 50,
    refDistance: 1,
    rolloffFactor: valence > 0.999 ? 0.8 : 2.0,
    distanceModel: "exponential",
    volume: valence > 0.999 ? 0.7 : 0.4
  });
  soundEntity.sound.addSlot("chime", { asset: url, autoPlay: true, loop: true });
  soundEntity.setPosition(position);

  // Visual indicator
  const sphere = new pc.Entity();
  sphere.addComponent("model", { type: "sphere" });
  sphere.setLocalScale(1, 1, 1);
  const mat = new pc.StandardMaterial();
  mat.emissive.set(valence > 0.999 ? 0 : 0.3, 1, 0.5);
  sphere.model.meshInstances[0].material = mat;
  sphere.setPosition(position);
  app.root.addChild(sphere);

  app.root.addChild(soundEntity);
  mercySounds.push(soundEntity.sound);
  console.log(`[PlayCanvasMercy] Positional mercy sound added – valence modulated (${valence.toFixed(8)})`);
}

// Custom snap teleport script (attach to controller entities or root)
function setupSnapTeleport() {
  if (!xrActive) return;

  // Assume controller entities exist (from XR input)
  // Simple raycast from camera forward for demo (expand to controller ray)
  app.on("update", dt => {
    if (app.keyboard.wasPressed(pc.KEY_SPACE) || /* controller trigger */) {
      const from = camera.getPosition();
      const direction = camera.forward;
      const to = from.clone().add(direction.scale(SNAP_DISTANCE));

      // Raycast to ground
      const result = app.systems.rigidbody.raycastFirst(from, to);
      if (result && result.entity.name === "ground") {
        // Teleport with animation
        camera.setPosition(to.x, camera.getPosition().y, to.z);
        teleportIndicator.setPosition(to);
        teleportIndicator.enabled = true;
        setTimeout(() => { teleportIndicator.enabled = false; }, 500);

        console.log("[PlayCanvasMercy] Snap teleport executed – mercy movement");
      }
    }
  });
}

// Mercy gate check
function mercyGateContent(textOrQuery, valence = 1.0) {
  const degree = fuzzyMercy.getDegree(textOrQuery) || valence;
  const implyThriving = fuzzyMercy.imply(textOrQuery, "EternalThriving");
  return degree >= mercyThreshold && implyThriving.degree >= mercyThreshold;
}

// Entry: start PlayCanvas mercy immersion with teleport
async function startPlayCanvasMercyImmersion(query = '', valence = 1.0) {
  if (!mercyGateContent(query, valence)) return;

  const canvas = document.querySelector('#playcanvas-canvas') || document.createElement('canvas');
  if (!canvas.id) canvas.id = 'playcanvas-canvas';
  document.body.appendChild(canvas);

  await initPlayCanvasMercy(canvas.id);

  // Example high-valence chime
  if (valence > 0.999) {
    addMercyPositionalSound('https://example.com/mercy-chime.mp3', new pc.Vec3(0, 1.5, -3), valence, query);
  }

  // Start XR session on high valence
  if (valence > 0.9995 && app.xr.supported) {
    try {
      const cameraEntity = app.root.findByName("camera");
      app.xr.start(cameraEntity.camera, pc.XRTYPE_VR, pc.XRSPACE_LOCALFLOOR).then(() => {
        xrActive = true;
        setupSnapTeleport();
        console.log("[PlayCanvasMercy] XR session started – PlayCanvas teleport mercy immersed");
      });
    } catch (err) {
      console.warn("[PlayCanvasMercy] XR start failed – fallback non-XR");
    }
  }
}

export { startPlayCanvasMercyImmersion };

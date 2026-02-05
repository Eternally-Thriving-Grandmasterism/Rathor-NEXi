// mercy-threejs-webxr-immersion.js – sovereign Three.js + WebXR + PositionalAudio v1
// Head-tracked spatial audio, mercy gates, valence-modulated visuals/sound
// MIT License – Autonomicity Games Inc. 2026

import * as THREE from 'https://esm.run/three@0.168.0'; // Latest stable
import { XRButton } from 'https://esm.run/three@0.168.0/examples/jsm/webxr/XRButton.js';
import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let scene, camera, renderer, xrSession;
let audioCtx, listener, positionalSounds = [];
const mercyThreshold = 0.9999999 * 0.98;

function initThreeXR() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000022); // Mercy deep space

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 1.6, 3); // Eye height

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.xr.enabled = true;
  document.body.appendChild(renderer.domElement);

  // XR button
  document.body.appendChild(XRButton.createButton(renderer, {
    sessionInit: { optionalFeatures: ['local-floor'] }
  }));

  // Mercy light + grid
  const light = new THREE.HemisphereLight(0xffffff, 0x444466, 1);
  scene.add(light);
  const grid = new THREE.GridHelper(20, 20, 0x00ff88, 0x004400);
  scene.add(grid);

  // Audio setup
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  listener = new THREE.AudioListener();
  camera.add(listener);

  // Example spatial sound source
  createSpatialSound('https://example.com/mercy-chime.mp3', new THREE.Vector3(0, 1.5, -5));

  // Resize handler
  window.addEventListener('resize', onResize);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function createSpatialSound(url, position) {
  const sound = new THREE.PositionalAudio(listener);
  sound.setRefDistance(1);
  sound.setMaxDistance(50);
  sound.setRolloffFactor(2);
  sound.setDistanceModel('exponential');
  sound.setDirectionalCone(360, 0, 0); // Omnidirectional

  const loader = new THREE.AudioLoader();
  loader.load(url, buffer => {
    sound.setBuffer(buffer);
    sound.setLoop(true);
    sound.setVolume(0.5);
    sound.play();
  });

  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.5),
    new THREE.MeshBasicMaterial({ color: 0x00ff88 })
  );
  mesh.position.copy(position);
  mesh.add(sound);
  scene.add(mesh);

  positionalSounds.push(sound);
  return sound;
}

function mercyGateContent(textOrQuery, valence = 1.0) {
  const degree = fuzzyMercy.getDegree(textOrQuery) || valence;
  const implyThriving = fuzzyMercy.imply(textOrQuery, "EternalThriving");
  if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
    console.log("[ThreeXR] Mercy gate: low valence – immersion skipped");
    return false;
  }
  return true;
}

function animate() {
  renderer.setAnimationLoop((time, frame) => {
    if (frame) {
      const pose = frame.getViewerPose(renderer.xr.getReferenceSpace());
      if (pose) {
        // Sync listener to head (Three.js auto-handles via listener in camera)
        // But for custom valence modulation
        positionalSounds.forEach(sound => {
          // Example: closer/high valence → louder
          const dist = sound.parent.position.distanceTo(camera.position);
          sound.setVolume(0.5 / (1 + dist * 0.1)); // Simple attenuation tweak
        });
      }
    }

    renderer.render(scene, camera);
  });
}

// Entry: init + start
async function startThreeXRImmersion(query = '', valence = 1.0) {
  if (!mercyGateContent(query, valence)) return;

  initThreeXR();
  animate();

  // Optional: enter immersive-vr on high valence
  if (valence > 0.9995 && navigator.xr) {
    try {
      const session = await navigator.xr.requestSession('immersive-vr');
      renderer.xr.setSession(session);
      console.log("[ThreeXR] Immersive VR session activated – spatial mercy immersed");
    } catch (err) {
      console.warn("[ThreeXR] Immersive session failed – fallback desktop XR");
    }
  }
}

export { startThreeXRImmersion };

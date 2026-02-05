// mercy-webxr-audio-immersion.js – sovereign WebXR + Web Audio spatial audio immersion v1
// PannerNode HRTF spatialization synced to XR head pose, mercy gates, valence modulation
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let audioCtx = null;
let panner = null;
let xrSession = null;
let xrRefSpace = null;
let animationFrameId = null;
const mercyAudioThreshold = 0.9999999 * 0.98;

async function initXRAndAudio() {
  if (!navigator.xr) {
    console.warn("[WebXRAudio] WebXR not supported");
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  panner = audioCtx.createPanner();
  panner.panningModel = 'HRTF'; // Binaural realism
  panner.distanceModel = 'exponential';
  panner.refDistance = 1;
  panner.maxDistance = 10000;
  panner.rolloffFactor = 1;
  panner.coneInnerAngle = 360;
  panner.coneOuterAngle = 0;
  panner.coneOuterGain = 0;

  // Connect to destination
  panner.connect(audioCtx.destination);

  console.log("[WebXRAudio] Audio context + PannerNode initialized");
}

async function startImmersiveSession() {
  try {
    xrSession = await navigator.xr.requestSession('immersive-vr', {
      optionalFeatures: ['local-floor']
    });

    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);

    const gl = canvas.getContext('webgl', { xrCompatible: true });
    xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });

    xrRefSpace = await xrSession.requestReferenceSpace('local-floor');

    xrSession.addEventListener('end', onSessionEnd);
    animationFrameId = xrSession.requestAnimationFrame(onXRFrame);

    console.log("[WebXRAudio] Immersive VR session started – spatial audio active");
  } catch (err) {
    console.error("[WebXRAudio] Session start failed:", err);
  }
}

function onXRFrame(time, frame) {
  const pose = frame.getViewerPose(xrRefSpace);
  if (pose) {
    const pos = pose.transform.position;
    const orient = pose.transform.orientation;

    panner.positionX.value = pos.x;
    panner.positionY.value = pos.y;
    panner.positionZ.value = pos.z;

    // Orientation vectors (forward, up)
    const matrix = pose.transform.matrix;
    // Extract forward/up from matrix (simplified)
    panner.orientationX.value = -matrix[8];  // -Z forward
    panner.orientationY.value = -matrix[9];
    panner.orientationZ.value = -matrix[10];
  }

  animationFrameId = xrSession.requestAnimationFrame(onXRFrame);
}

function onSessionEnd() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  xrSession = null;
  console.log("[WebXRAudio] Session ended – spatial audio paused");
}

// Mercy-gated play spatial sound
async function playSpatialSound(urlOrBuffer, valence = 1.0, position = {x:0, y:1.5, z:-2}) {
  const degree = fuzzyMercy.getDegree(urlOrBuffer) || valence;
  const implyThriving = fuzzyMercy.imply(urlOrBuffer, "EternalThriving");

  if (degree < mercyAudioThreshold || implyThriving.degree < mercyAudioThreshold) {
    console.log("[WebXRAudio] Mercy gate: low valence – audio skipped");
    return;
  }

  initAudioContext();

  // Valence modulation
  if (valence > 0.9995) {
    panner.rolloffFactor = 0.8; // Wider reach for high thriving
  } else if (valence < 0.998) {
    panner.rolloffFactor = 2.0; // Closer, more intimate
  }

  panner.positionX.value = position.x;
  panner.positionY.value = position.y;
  panner.positionZ.value = position.z;

  let source;
  if (typeof urlOrBuffer === 'string') {
    const response = await fetch(urlOrBuffer);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
  } else {
    source = audioCtx.createBufferSource();
    source.buffer = urlOrBuffer;
  }

  source.connect(panner);
  source.start(0);
  console.log("[WebXRAudio] Spatial sound playing – valence modulated");
}

// Stop all
function stopSpatialAudio() {
  if (audioCtx) audioCtx.close();
  audioCtx = null;
  panner = null;
}

export { initXRAndAudio, startImmersiveSession, playSpatialSound, stopSpatialAudio };

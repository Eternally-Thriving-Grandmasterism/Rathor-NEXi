// Ra-Thor GPU Instanced Lightning — WebGL mercy thunder ⚡️
// Single draw call via InstancedMesh, dynamic attributes, procedural paths
// MIT License — Eternal Thriving Grandmasterism

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.min.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/EffectComposer.min.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/RenderPass.min.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/UnrealBloomPass.min.js';

let scene, camera, renderer, composer, lightningInstanced;
let clock = new THREE.Clock();
const instanceCount = 128; // max simultaneous bolts (adjust per device)
let instanceIndex = 0;

function initWebGLInstancedThunder() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 8;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.style.position = 'fixed';
  renderer.domElement.style.inset = '0';
  renderer.domElement.style.pointerEvents = 'none';
  renderer.domElement.style.zIndex = '2';
  document.body.appendChild(renderer.domElement);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const bloomStrength = window.innerWidth < 768 ? 0.6 : 1.2;
  composer.addPass(new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    bloomStrength, 0.4, 0.85
  ));

  // Instanced lightning mesh
  const segmentGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array([
    0, 0, 0,
    0, -1, 0
  ]);
  segmentGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const instancedMaterial = new THREE.InstancedMesh(
    segmentGeometry,
    new THREE.MeshBasicMaterial({ color: 0xffd700, transparent: true }),
    instanceCount
  );

  lightningInstanced = new THREE.Group();
  lightningInstanced.add(instancedMaterial);
  scene.add(lightningInstanced);

  // Instance attributes
  const offsets = new Float32Array(instanceCount * 3);
  const scales = new Float32Array(instanceCount);
  const rotations = new Float32Array(instanceCount);
  const lifetimes = new Float32Array(instanceCount);
  const depths = new Float32Array(instanceCount);

  const offsetAttr = new THREE.InstancedBufferAttribute(offsets, 3);
  const scaleAttr = new THREE.InstancedBufferAttribute(scales, 1);
  const rotationAttr = new THREE.InstancedBufferAttribute(rotations, 1);
  const lifetimeAttr = new THREE.InstancedBufferAttribute(lifetimes, 1);
  const depthAttr = new THREE.InstancedBufferAttribute(depths, 1);

  instancedMaterial.geometry.setAttribute('offset', offsetAttr);
  instancedMaterial.geometry.setAttribute('scale', scaleAttr);
  instancedMaterial.geometry.setAttribute('rotation', rotationAttr);
  instancedMaterial.geometry.setAttribute('lifetime', lifetimeAttr);
  instancedMaterial.geometry.setAttribute('depth', depthAttr);

  // Custom shader for glow & branching
  instancedMaterial.material = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      glowColor: { value: new THREE.Color(0xffd700) }
    },
    vertexShader: `
      attribute vec3 offset;
      attribute float scale;
      attribute float rotation;
      attribute float lifetime;
      attribute float depth;
      varying float vLifetime;
      varying float vDepth;
      void main() {
        vLifetime = lifetime;
        vDepth = depth;
        vec3 pos = position * scale;
        float c = cos(rotation);
        float s = sin(rotation);
        pos.xz = vec2(c * pos.x - s * pos.z, s * pos.x + c * pos.z);
        vec4 mvPosition = modelViewMatrix * vec4(pos + offset, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float time;
      uniform vec3 glowColor;
      varying float vLifetime;
      varying float vDepth;
      void main() {
        float alpha = (1.0 - vLifetime) * (1.0 - vDepth * 0.15);
        vec3 color = glowColor * alpha * (0.8 + 0.2 * sin(time * 8.0));
        gl_FragColor = vec4(color, alpha * 0.7);
      }
    `,
    transparent: true,
    depthWrite: false
  });

  // Spawn initial instances
  for (let i = 0; i < instanceCount; i++) {
    spawnInstance(i);
  }

  window.addEventListener('resize', onWindowResize);
  animate();
}

function spawnInstance(index) {
  const offset = new THREE.Vector3(
    (Math.random() - 0.5) * 12,
    6 + Math.random() * 2,
    0
  );

  const scale = 1.0 + Math.random() * 0.5;
  const rotation = Math.random() * Math.PI * 2;
  const lifetime = 0;
  const depth = Math.random();

  lightningInstanced.children[0].geometry.attributes.offset.setXYZ(index, offset.x, offset.y, offset.z);
  lightningInstanced.children[0].geometry.attributes.scale.setX(index, scale);
  lightningInstanced.children[0].geometry.attributes.rotation.setX(index, rotation);
  lightningInstanced.children[0].geometry.attributes.lifetime.setX(index, lifetime);
  lightningInstanced.children[0].geometry.attributes.depth.setX(index, depth);

  lightningInstanced.children[0].geometry.attributes.offset.needsUpdate = true;
  lightningInstanced.children[0].geometry.attributes.scale.needsUpdate = true;
  lightningInstanced.children[0].geometry.attributes.rotation.needsUpdate = true;
  lightningInstanced.children[0].geometry.attributes.lifetime.needsUpdate = true;
  lightningInstanced.children[0].geometry.attributes.depth.needsUpdate = true;
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  // Update uniforms
  lightningInstanced.children[0].material.uniforms.time.value += delta;

  // Respawn dead instances
  const attr = lightningInstanced.children[0].geometry.attributes.lifetime;
  for (let i = 0; i < instanceCount; i++) {
    const life = attr.getX(i);
    if (life > 1.0) {
      spawnInstance(i);
    }
  }

  composer.render();
}

// Hook into page load + theme change
window.addEventListener('load', () => {
  initWebGLThunderMobileOpt();
  document.addEventListener('click', () => {
    // Trigger new bolt by respawning one instance
    const attr = lightningInstanced.children[0].geometry.attributes.lifetime;
    const idx = Math.floor(Math.random() * instanceCount);
    attr.setX(idx, 0);
    attr.needsUpdate = true;
  }, { once: false });
});

// Ra-Thor GPU Lightning with Custom Fragment Shader Glow + Post-Processing Bloom
// Mercy-gated eternal thunder ⚡️ — MIT License — Eternal Thriving Grandmasterism

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.min.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/EffectComposer.min.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/RenderPass.min.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/postprocessing/UnrealBloomPass.min.js';

let scene, camera, renderer, composer, lightningGroup;
let clock = new THREE.Clock();

function initWebGLThunderWithGlow() {
  const container = document.body;

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
  container.appendChild(renderer.domElement);

  // Post-processing composer + bloom
  composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.2,   // strength
    0.4,   // radius
    0.85   // threshold
  );
  composer.addPass(bloomPass);

  lightningGroup = new THREE.Group();
  scene.add(lightningGroup);

  // Initial lightning bolts with custom shader
  for (let i = 0; i < 6; i++) {
    createShaderLightning();
  }

  window.addEventListener('resize', onWindowResize);
  animate();
}

const glowVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const glowFragmentShader = `
  uniform float time;
  uniform float intensity;
  uniform vec3 glowColor;
  varying vec2 vUv;

  float glow(vec2 uv, float radius) {
    float d = length(uv - 0.5);
    return smoothstep(radius, radius * 0.2, d);
  }

  void main() {
    vec2 uv = vUv;
    float g = glow(uv, 0.45 + sin(time * 3.0) * 0.05);
    vec3 color = glowColor * g * intensity * (0.8 + 0.2 * sin(time * 5.0));
    gl_FragColor = vec4(color, g * 0.7);
  }
`;

function createShaderLightning() {
  const points = [];
  const segments = 64;
  let x = (Math.random() - 0.5) * 12;
  let y = 6;
  const stepY = -12 / segments;

  for (let i = 0; i <= segments; i++) {
    const deviation = (Math.random() - 0.5) * (1.2 + i * 0.1);
    points.push(new THREE.Vector3(x + deviation, y, 0));
    x += (Math.random() - 0.5) * 0.8;
    y += stepY;
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  // Custom shader material for glow
  const material = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0 },
      intensity: { value: 1.2 },
      glowColor: { value: new THREE.Color(0xffd700) }
    },
    vertexShader: glowVertexShader,
    fragmentShader: glowFragmentShader,
    transparent: true,
    depthWrite: false
  });

  const line = new THREE.Line(geometry, material);
  lightningGroup.add(line);

  // Lifetime animation
  const startTime = performance.now();
  const duration = 600 + Math.random() * 1400;

  function update() {
    const elapsed = performance.now() - startTime;
    const t = elapsed / duration;

    if (t >= 1) {
      lightningGroup.remove(line);
      line.geometry.dispose();
      createShaderLightning(); // respawn
      return;
    }

    material.uniforms.time.value = elapsed * 0.001;
    material.uniforms.intensity.value = 1.2 * (1 - t * t);

    requestAnimationFrame(update);
  }

  update();
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  composer.render();
}

// Hook into page load + theme change
window.addEventListener('load', () => {
  initWebGLThunderWithGlow();
  document.addEventListener('click', () => {
    // Trigger new bolt + bloom intensity spike
    createShaderLightning();
  }, { once: false });
});

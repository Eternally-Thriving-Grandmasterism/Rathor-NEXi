// mercy-audio-visualizer.js – sovereign Web Audio API waveform visualizer v1
// Real-time canvas drawing during speech, mercy-gated, valence-colored
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

let audioCtx = null;
let analyser = null;
let canvas = null;
let canvasCtx = null;
let animationId = null;
const mercyVizThreshold = 0.9999999 * 0.98;

function initAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048; // Balance resolution vs perf
    console.log("[AudioViz] Web Audio Context initialized – analyser ready");
  }
}

function createTestOscillator() {
  initAudioContext();
  const oscillator = audioCtx.createOscillator();
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(440, audioCtx.currentTime);
  oscillator.connect(analyser);
  analyser.connect(audioCtx.destination);
  oscillator.start();
  return oscillator;
}

function startVisualization(targetCanvas, text = '', valence = 1.0) {
  if (!targetCanvas) return;
  canvas = targetCanvas;
  canvasCtx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const degree = fuzzyMercy.getDegree(text) || valence;
  if (degree < mercyVizThreshold) {
    console.log("[AudioViz] Mercy gate: low valence – viz skipped");
    return;
  }

  initAudioContext();

  // For TTS: no direct connect, so use test osc for demo viz during speak
  // Real TTS viz needs MediaStream hack (partial support) – here test tone
  const testOsc = createTestOscillator(); // Replace with real if stream possible

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    animationId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgb(20, 20, 40)'; // Dark mercy bg
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    // Valence color: high green, mid blue, low purple
    canvasCtx.strokeStyle = valence > 0.999 ? '#00ff88' : valence > 0.998 ? '#4488ff' : '#aa44ff';
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }

  draw();
}

function stopVisualization() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  if (canvasCtx) {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

export { startVisualization, stopVisualization };

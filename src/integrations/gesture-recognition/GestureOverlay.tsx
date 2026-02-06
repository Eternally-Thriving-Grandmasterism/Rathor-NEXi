// src/integrations/gesture-recognition/GestureOverlay.tsx – Gesture Recognition Overlay v1
// Real-time MR/XR overlay, valence breathing, haptic feedback, persistent anchors, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import * as handpose from '@tensorflow-models/handpose';
import '@tensorflow/tfjs-backend-webgl';

const GestureOverlay: React.FC<{ videoRef: React.RefObject<HTMLVideoElement> }> = ({ videoRef }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentGesture, setCurrentGesture] = useState<string | null>(null);
  const [valenceGlow, setValenceGlow] = useState(currentValence.get());
  const [isRecognizing, setIsRecognizing] = useState(false);

  // Gesture recognition model & state
  const modelRef = useRef<handpose.HandPose | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      if (!await mercyGate('Load gesture recognition model')) return;
      modelRef.current = await handpose.load();
      setIsRecognizing(true);
      startDetectionLoop();
    };

    loadModel();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  const startDetectionLoop = () => {
    const detect = async () => {
      if (videoRef.current && modelRef.current && canvasRef.current) {
        const predictions = await modelRef.current.estimateHands(videoRef.current);

        if (predictions.length > 0) {
          const landmarks = predictions[0].landmarks;
          const gesture = recognizeGesture(landmarks);

          if (gesture) {
            setCurrentGesture(gesture);
            mercyHaptic.playPattern(getHapticForGesture(gesture), valenceGlow);
            // Trigger lattice action
            handleRecognizedGesture(gesture);
          }
        } else {
          setCurrentGesture(null);
        }
      }

      animationFrameRef.current = requestAnimationFrame(detect);
    };

    detect();
  };

  // Simple gesture recognizer (expand with ML or rule-based logic)
  const recognizeGesture = (landmarks: number[][]): string | null => {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];

    const pinchDistance = Math.hypot(thumbTip[0] - indexTip[0], thumbTip[1] - indexTip[1]);
    const spiralMotion = Math.abs(thumbTip[0] - middleTip[0]) > 50; // crude motion detection

    if (pinchDistance < 30) return 'pinch';           // propose alliance
    if (spiralMotion) return 'spiral';                // bloom swarm
    if (Math.abs(indexTip[1] - middleTip[1]) < 20) return 'figure8'; // infinite harmony loop

    return null;
  };

  const getHapticForGesture = (gesture: string): string => {
    switch (gesture) {
      case 'pinch': return 'allianceProposal';
      case 'spiral': return 'swarmBloom';
      case 'figure8': return 'eternalHarmony';
      default: return 'neutralPulse';
    }
  };

  const handleRecognizedGesture = async (gesture: string) => {
    if (!await mercyGate(`Handle gesture: ${gesture}`)) return;

    switch (gesture) {
      case 'pinch':
        // Propose alliance – trigger negotiation layer
        console.log("[GestureOverlay] Pinch → Alliance proposal initiated");
        break;
      case 'spiral':
        // Bloom swarm – launch molecular swarm progression
        console.log("[GestureOverlay] Spiral → Swarm bloom activated");
        break;
      case 'figure8':
        // Infinite harmony loop – cycle positive-sum state
        console.log("[GestureOverlay] Figure-8 → Eternal harmony loop engaged");
        currentValence.addDelta(0.02);
        break;
    }
  };

  useEffect(() => {
    setValenceGlow(currentValence.get());
  }, [currentValence.get()]);

  return (
    <div className="fixed inset-0 pointer-events-none z-50">
      <canvas ref={canvasRef} className="absolute inset-0" />

      <AnimatePresence>
        {currentGesture && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.4 }}
          >
            <motion.div
              className="relative w-96 h-96 rounded-full border-4 border-cyan-400/30 backdrop-blur-xl"
              style={{
                background: `radial-gradient(circle at 50% 50%, rgba(0,255,136,${valenceGlow}) 0%, transparent 70%)`
              }}
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <motion.div
                className="absolute inset-4 rounded-full border-2 border-emerald-400/50"
                animate={{ scale: [1, 1.15, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-4xl font-light text-white/90">
                {currentGesture.toUpperCase()}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Persistent anchor indicators */}
      <div className="absolute bottom-8 left-8 text-cyan-200/70 text-sm backdrop-blur-sm p-4 rounded-xl border border-cyan-500/20">
        Pinch: Propose Alliance  
        Spiral: Bloom Swarm  
        Figure-8: Eternal Harmony Loop
      </div>
    </div>
  );
};

export default GestureOverlay;

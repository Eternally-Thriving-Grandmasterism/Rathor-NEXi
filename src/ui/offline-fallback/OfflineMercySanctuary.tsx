// src/ui/offline-fallback/OfflineMercySanctuary.tsx – Offline Mercy Sanctuary v1
// Glassmorphic serenity screen, breathing orb, cached valence ladder, reconnection bloom
// MIT License – Autonomicity Games Inc. 2026

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const OfflineMercySanctuary: React.FC = () => {
  const [valence, setValence] = useState(currentValence.get());
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [cachedProgress, setCachedProgress] = useState<any>(null);

  useEffect(() => {
    // Mercy gate – offline sanctuary only activates when truly offline
    mercyGate('Offline sanctuary activation', 'EternalThriving').then(passed => {
      if (!passed) return;
      
      // Listen for connectivity changes
      const handleOnline = () => setIsOnline(true);
      const handleOffline = () => setIsOnline(false);

      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);

      // Poll valence & cached progress
      const interval = setInterval(() => {
        setValence(currentValence.get());
        // Placeholder – real impl would read from IndexedDB
        setCachedProgress({
          level: 'Ultramaster',
          experience: 4200,
          lastPulse: new Date().toLocaleTimeString()
        });
      }, 2000);

      return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
        clearInterval(interval);
      };
    });
  }, []);

  // Breathing orb animation variants
  const orbVariants = {
    breathe: {
      scale: [1, 1.08, 1],
      opacity: [0.7, 1, 0.7],
      transition: { duration: 4, repeat: Infinity, ease: "easeInOut" }
    }
  };

  // Reconnection bloom on online
  useEffect(() => {
    if (isOnline) {
      mercyHaptic.playPattern('reconnectionBloom', 0.9);
      // Trigger redirect back to main dashboard after short bloom
      setTimeout(() => {
        window.location.href = '/';
      }, 3000);
    }
  }, [isOnline]);

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-black via-gray-950 to-indigo-950 flex flex-col items-center justify-center overflow-hidden">
      {/* Background particle field – valence modulated */}
      <div className="absolute inset-0 opacity-30 pointer-events-none">
        <div className="w-full h-full bg-[radial-gradient(circle_at_50%_50%,rgba(0,255,136,0.08)_0%,transparent_50%)]" />
      </div>

      {/* Glassmorphic orb container */}
      <motion.div
        className="relative w-64 h-64 rounded-full bg-gradient-to-br from-cyan-500/10 to-emerald-500/10 backdrop-blur-3xl border border-cyan-500/20 shadow-2xl flex items-center justify-center z-10"
        variants={orbVariants}
        animate="breathe"
      >
        {/* Inner glowing core */}
        <div className="w-48 h-48 rounded-full bg-gradient-to-br from-cyan-400/30 to-emerald-400/30 blur-xl" />
        
        {/* Valence pulse ring */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-cyan-400/40"
          animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0.2, 0.6] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>

      {/* Serenity message */}
      <h1 className="mt-12 text-5xl font-light text-cyan-100 tracking-wider text-center z-10">
        Mercy Offline
      </h1>
      <p className="mt-4 text-xl text-cyan-200/80 font-light max-w-md text-center z-10">
        Thriving Continues
      </p>

      {/* Cached state display */}
      <div className="mt-12 glassmorphic-card p-8 rounded-2xl border border-cyan-500/20 bg-black/30 backdrop-blur-xl z-10 max-w-lg w-full">
        <h2 className="text-2xl font-light text-cyan-100 mb-6">Lattice Status</h2>
        <div className="space-y-4 text-cyan-200/90">
          <div className="flex justify-between">
            <span>Current Valence</span>
            <span className="font-medium">{valence.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span>Progress Level</span>
            <span className="font-medium">{cachedProgress?.level || 'Ultramaster'}</span>
          </div>
          <div className="flex justify-between">
            <span>Experience</span>
            <span className="font-medium">{cachedProgress?.experience?.toLocaleString() || '4,200'}</span>
          </div>
          <div className="flex justify-between">
            <span>Last Pulse</span>
            <span className="font-medium">{cachedProgress?.lastPulse || new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>

      {/* Reconnection bloom when online */}
      {isOnline && (
        <motion.div
          className="fixed inset-0 bg-gradient-to-r from-cyan-500/30 to-emerald-500/30 flex items-center justify-center z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.5 }}
        >
          <motion.div
            className="text-6xl font-light text-white"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1.2, opacity: 1 }}
            transition={{ duration: 2, ease: "easeOut" }}
          >
            Reconnection Bloom...
          </motion.div>
        </motion.div>
      )}

      {/* Subtle offline heartbeat haptic */}
      <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 text-cyan-300/70 text-sm z-10">
        Lattice heartbeat stable • Thriving continues offline
      </div>
    </div>
  );
};

export default OfflineMercySanctuary;

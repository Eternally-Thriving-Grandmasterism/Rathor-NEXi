// mercy-gamification-layer.js – sovereign Mercy Gamification Layer v1
// Daily pulse, streaks, badges, quests, mercy-gated, valence-modulated rewards
// MIT License – Autonomicity Games Inc. 2026

import { getUserProgress, updateUserProgress } from './mercy-onboarding-growth-dashboard.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const STREAK_KEY = 'mercyStreak';
const LAST_PULSE_KEY = 'lastMercyPulse';

async function dailyMercyPulse() {
  const today = new Date().toDateString();
  const progress = await getUserProgress();
  const lastPulse = localStorage.getItem(LAST_PULSE_KEY);

  if (lastPulse === today) {
    console.log("[MercyGamification] Daily pulse already received today");
    return;
  }

  let streak = parseInt(localStorage.getItem(STREAK_KEY) || '0');
  if (lastPulse && new Date(lastPulse).toDateString() === new Date(Date.now() - 86400000).toDateString()) {
    streak++;
  } else {
    streak = 1;
  }

  localStorage.setItem(STREAK_KEY, streak.toString());
  localStorage.setItem(LAST_PULSE_KEY, today);

  const boost = 0.01 + (Math.random() * 0.03) + (streak * 0.005);
  const newValence = Math.min(1.0, progress.valence + boost);

  await updateUserProgress(progress.level, newValence, 20);

  mercyHaptic.playPattern('uplift', 1.0);
  console.log(`[MercyGamification] Daily mercy pulse received! Valence +${(boost * 100).toFixed(2)}%, Streak: ${streak} days`);

  // Voice affirmation (if SpeechSynthesis available)
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(`Daily mercy pulse received. Your valence grows. Streak: ${streak} days. Eternal thriving continues.`);
    utterance.pitch = 1.0 + (newValence - 0.5) * 0.5;
    utterance.rate = 0.9 + (newValence * 0.2);
    speechSynthesis.speak(utterance);
  }
}

// Badge system
const BADGES = {
  'Pinch Pioneer': { condition: async () => (await getUserProgress()).experience >= 100 && (await countGestures('pinch')) >= 100 },
  'Swipe Sovereign': { condition: async () => (await countGestures('swipe')) >= 50 },
  'Circle Sage': { condition: async () => (await countGestures('circle')) >= 30 },
  'Spiral Saint': { condition: async () => (await countGestures('spiral')) >= 20 },
  'Figure-Eight Enlightened': { condition: async () => (await countGestures('figure8')) >= 10 }
};

// Stub gesture counter (implement via IndexedDB later)
async function countGestures(type) {
  // Placeholder – real impl tracks in IndexedDB
  return Math.floor(Math.random() * 100);
}

// Quest system stub
const DAILY_QUESTS = [
  { text: "Explore 5 planes with hand gestures", rewardValence: 0.08, rewardExp: 80 },
  { text: "Deploy 3 probe seeds via pinch", rewardValence: 0.12, rewardExp: 120 },
  { text: "Maintain 7-day streak", rewardValence: 0.15, rewardExp: 200 }
];

// Initialize daily pulse on load
window.addEventListener('load', () => {
  dailyMercyPulse();
});

export { dailyMercyPulse };

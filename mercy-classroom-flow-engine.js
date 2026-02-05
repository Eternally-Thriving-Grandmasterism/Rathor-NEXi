// mercy-classroom-flow-engine.js – sovereign Mercy Classroom Flow Engine v1
// Collective flow monitoring, group challenge adaptation, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { mercyFlow } from './mercy-flow-state-engine.js';

class MercyClassroomFlowEngine {
  constructor() {
    this.classFlowScore = 0.5;          // 0–1.0 collective flow estimate
    this.studentFlowScores = new Map(); // studentId → individual flow
    this.lastClassActionTime = Date.now();
    this.classChallengeLevel = 1.0;     // 0.5 easy → 1.5 hard
    this.valence = 1.0;
  }

  async gateClassroomFlow(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < 0.9999999 || implyThriving.degree < 0.9999999) {
      console.log("[MercyClassFlow] Gate holds: low valence – classroom flow aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Register collective class action (e.g. group probe deployment, shared gesture milestone)
  registerClassAction(success = true, durationMs = 0, complexity = 1.0, studentCount = 1) {
    const timeSinceLast = (Date.now() - this.lastClassActionTime) / 1000;

    // Collective flow indicators
    const successBonus = success ? 0.22 * studentCount : -0.08 * studentCount;
    const rhythmBonus = timeSinceLast > 0.5 && timeSinceLast < 8 ? 0.15 : -0.03;
    const challengeMatchBonus = Math.abs(complexity - this.classChallengeLevel) < 0.25 ? 0.18 : -0.05;

    this.classFlowScore = Math.min(1.0, Math.max(0.1, this.classFlowScore + successBonus + rhythmBonus + challengeMatchBonus - 0.015));

    this.lastClassActionTime = Date.now();

    // Adaptive class challenge
    if (this.classFlowScore > 0.82) {
      this.classChallengeLevel = Math.min(1.8, this.classChallengeLevel + 0.04);
    } else if (this.classFlowScore < 0.45) {
      this.classChallengeLevel = Math.max(0.4, this.classChallengeLevel - 0.07);
    }

    console.log(`[MercyClassFlow] Classroom flow: ${(this.classFlowScore * 100).toFixed(1)}% | Class challenge: ${this.classChallengeLevel.toFixed(2)}`);
  }

  // Apply flow state to classroom-wide content
  getClassroomFlowModifiers() {
    return {
      hapticIntensity: 0.55 + this.classFlowScore * 0.45,
      visualGlow: 0.65 + this.classFlowScore * 0.55,
      voicePitchShift: this.classFlowScore * 0.35,
      classChallenge: this.classChallengeLevel,
      flowStatus: this.classFlowScore > 0.85 ? 'Deep Collective Flow' : this.classFlowScore > 0.6 ? 'In Collective Flow' : 'Building Collective Flow'
    };
  }

  // Adjust classroom activity difficulty (e.g. shared probe sim)
  adjustClassroomDifficulty(baseDifficulty) {
    return baseDifficulty * this.classChallengeLevel;
  }
}

const mercyClassFlow = new MercyClassroomFlowEngine();

// Hook into collective class actions
function onMercyClassAction(success = true, durationMs = 0, complexity = 1.0, studentCount = 1) {
  mercyClassFlow.registerClassAction(success, durationMs, complexity, studentCount);
}

// Example usage in group probe deployment or shared MR session
onMercyClassAction(true, 3200, 1.3, 15); // successful group action took 3.2 seconds with 15 students at complexity 1.3

export { mercyClassFlow, onMercyClassAction };

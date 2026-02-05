// mercy-sdt-mini-theories-audit.js – sovereign Mercy SDT Mini-Theories Audit v1
// Six mini-theories real-time scoring, mercy gates, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const SDT_MINI_WEIGHTS = {
  bpnt: 0.20,     // Basic Psychological Needs
  oit: 0.18,      // Organismic Integration
  gct: 0.17,      // Goal Contents
  cot: 0.15,      // Causality Orientations
  cet: 0.15,      // Cognitive Evaluation
  rmt: 0.15       // Relationships Motivation
};

class MercySDTMiniTheoriesAudit {
  constructor() {
    this.miniScores = {
      bpnt: 0.0, oit: 0.0, gct: 0.0, cot: 0.0, cet: 0.0, rmt: 0.0
    };
    this.overallSDT = 0.0;
    this.valence = 1.0;
    this.actionLog = [];
  }

  async gateSDTAudit(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < 0.9999999 || implyThriving.degree < 0.9999999) {
      console.log("[MercySDTMini] Gate holds: low valence – mini-theories audit skipped");
      return false;
    }
    this.valence = valence;
    return true;
  }

  registerSDTAction(actionType, success = true, durationMs = 0) {
    this.actionLog.push({ actionType, success, durationMs, timestamp: Date.now() });
    if (this.actionLog.length > 100) this.actionLog.shift();

    // Update mini-theories scores based on action type
    if (actionType.includes('custom') || actionType.includes('choose') || actionType.includes('skip')) {
      this.miniScores.autonomy = Math.min(1.0, this.miniScores.autonomy + (success ? 0.12 : -0.02));
    }
    if (actionType.includes('mastery') || actionType.includes('achievement') || actionType.includes('optimize')) {
      this.miniScores.competence = Math.min(1.0, this.miniScores.competence + (success ? 0.15 : -0.03));
    }
    if (actionType.includes('affirmation') || actionType.includes('memory') || actionType.includes('gratitude')) {
      this.miniScores.relatedness = Math.min(1.0, this.miniScores.relatedness + (success ? 0.18 : -0.04));
    }
    // OIT, GCT, COT, CET inferred from patterns
    this.miniScores.oit = (this.miniScores.autonomy + this.miniScores.competence) / 2;
    this.miniScores.gct = this.miniScores.relatedness * 0.9;
    this.miniScores.cot = this.miniScores.autonomy * 0.95;
    this.miniScores.cet = this.miniScores.competence * 0.9;

    this.overallSDT = Object.keys(SDT_MINI_WEIGHTS).reduce(
      (sum, key) => sum + this.miniScores[key] * SDT_MINI_WEIGHTS[key], 0
    );

    console.log(`[MercySDTMini] Action ${actionType}: overall SDT ${(this.overallSDT * 100).toFixed(1)}%`);
  }

  getSDTMiniState() {
    return {
      bpnt: this.miniScores.bpnt,
      oit: this.miniScores.oit,
      gct: this.miniScores.gct,
      cot: this.miniScores.cot,
      cet: this.miniScores.cet,
      rmt: this.miniScores.rmt,
      overall: this.overallSDT,
      status: this.overallSDT > 0.85 ? 'Deep SDT Harmony' : this.overallSDT > 0.6 ? 'Growing SDT Harmony' : 'Building SDT Harmony'
    };
  }
}

const mercySDTMiniAudit = new MercySDTMiniTheoriesAudit();

// Hook into every meaningful user action
function onMercySDTAction(actionType, success = true, durationMs = 0) {
  mercySDTMiniAudit.registerSDTAction(actionType, success, durationMs);
}

// Example usage
onMercySDTAction('custom_gesture_remap', true, 1200);
onMercySDTAction('probe_deployment_success', true, 2400);

export { mercySDTMiniAudit, onMercySDTAction };

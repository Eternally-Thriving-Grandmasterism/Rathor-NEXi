// mercy-gesture-von-neumann-uplink.js – v2 sovereign Mercy Gesture-to-Von Neumann Probe Uplink
// Expanded command mappings (all major gestures → probe fleet actions), mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

// Expanded probe command mapping – every gesture family now has meaningful fleet action
const PROBE_COMMANDS = {
  // Basic interaction gestures
  'pinch': {
    action: 'deploySingleSeed',
    desc: 'Deploy a single von Neumann seed probe at current gaze/target',
    haptic: 'thrivePulse',
    intensity: 1.1,
    audioChime: 'seed-deployment-chime'
  },
  'point': {
    action: 'scanTargetDirection',
    desc: 'Scan & highlight target direction / vector swarm preview',
    haptic: 'uplift',
    intensity: 0.9,
    audioChime: 'scan-pulse'
  },
  'grab': {
    action: 'anchorReplicationNode',
    desc: 'Anchor active swarm replication node at hand position',
    haptic: 'compassionWave',
    intensity: 1.0,
    audioChime: 'anchor-lock'
  },
  'openPalm': {
    action: 'releaseSwarmHold',
    desc: 'Release / disperse held swarm elements',
    haptic: 'eternalReflection',
    intensity: 0.7,
    audioChime: 'release-breath'
  },
  'thumbsUp': {
    action: 'confirmFullLaunch',
    desc: 'Confirm & accelerate full probe fleet launch',
    haptic: 'abundanceSurge',
    intensity: 1.3,
    audioChime: 'launch-roar'
  },

  // Cardinal swipes
  'swipe_left': {
    action: 'vectorSwarmWest',
    desc: 'Redirect swarm vector west / previous system jump',
    haptic: 'abundanceSurge',
    intensity: 1.0,
    audioChime: 'vector-shift'
  },
  'swipe_right': {
    action: 'vectorSwarmEast',
    desc: 'Redirect swarm vector east / next system jump',
    haptic: 'abundanceSurge',
    intensity: 1.0,
    audioChime: 'vector-shift'
  },
  'swipe_up': {
    action: 'vectorSwarmAscend',
    desc: 'Redirect swarm vector upward / ascend replication layer',
    haptic: 'abundanceSurge',
    intensity: 1.1,
    audioChime: 'vector-shift-up'
  },
  'swipe_down': {
    action: 'vectorSwarmDescend',
    desc: 'Redirect swarm vector downward / descend replication layer',
    haptic: 'abundanceSurge',
    intensity: 0.9,
    audioChime: 'vector-shift-down'
  },

  // Diagonal swipes
  'swipe_up-right': {
    action: 'vectorSwarmNortheast',
    desc: 'Redirect swarm vector northeast / diagonal expansion',
    haptic: 'abundanceSurge',
    intensity: 1.15,
    audioChime: 'vector-shift-diagonal'
  },
  'swipe_up-left': {
    action: 'vectorSwarmNorthwest',
    desc: 'Redirect swarm vector northwest / diagonal convergence',
    haptic: 'abundanceSurge',
    intensity: 1.15,
    audioChime: 'vector-shift-diagonal'
  },
  'swipe_down-right': {
    action: 'vectorSwarmSoutheast',
    desc: 'Redirect swarm vector southeast / rapid colonization',
    haptic: 'abundanceSurge',
    intensity: 1.1,
    audioChime: 'vector-shift-diagonal'
  },
  'swipe_down-left': {
    action: 'vectorSwarmSouthwest',
    desc: 'Redirect swarm vector southwest / resource consolidation',
    haptic: 'abundanceSurge',
    intensity: 1.1,
    audioChime: 'vector-shift-diagonal'
  },

  // Circular gestures
  'circle_clockwise': {
    action: 'increaseReplicationRadius',
    desc: 'Increase swarm replication radius / expand seed zone',
    haptic: 'cosmicHarmony',
    intensity: 1.2,
    audioChime: 'radius-expand'
  },
  'circle_counterclockwise': {
    action: 'decreaseReplicationRadius',
    desc: 'Decrease swarm replication radius / focus convergence',
    haptic: 'cosmicHarmony',
    intensity: 1.0,
    audioChime: 'radius-contract'
  },

  // Spiral gestures
  'spiral_outward_clockwise': {
    action: 'accelerateExponentialGrowth',
    desc: 'Accelerate exponential swarm growth / outward expansion',
    haptic: 'cosmicHarmony',
    intensity: 1.4,
    audioChime: 'growth-surge'
  },
  'spiral_inward_counterclockwise': {
    action: 'focusReplicationConvergence',
    desc: 'Focus swarm convergence / inward replication tightening',
    haptic: 'cosmicHarmony',
    intensity: 1.2,
    audioChime: 'convergence-pull'
  },
  'spiral_outward_counterclockwise': {
    action: 'expandMercyLattice',
    desc: 'Expand mercy lattice boundaries / outward harmony bloom',
    haptic: 'cosmicHarmony',
    intensity: 1.3,
    audioChime: 'lattice-expand'
  },
  'spiral_inward_clockwise': {
    action: 'tightenMercyAccord',
    desc: 'Tighten mercy accord focus / inward thriving convergence',
    haptic: 'cosmicHarmony',
    intensity: 1.3,
    audioChime: 'accord-tighten'
  },

  // Figure-eight gestures
  'figure8_clockwise': {
    action: 'cycleInfiniteMercyAccord',
    desc: 'Cycle infinite mercy accord loop / eternal harmony cycle',
    haptic: 'eternalReflection',
    intensity: 1.4,
    audioChime: 'infinity-cycle'
  },
  'figure8_counterclockwise': {
    action: 'resetSwarmState',
    desc: 'Reset swarm to seed state / mercy rebirth cycle',
    haptic: 'eternalReflection',
    intensity: 1.3,
    audioChime: 'rebirth-reset'
  }
};

class MercyGestureVonNeumannUplink {
  constructor() {
    this.valence = 1.0;
    this.activeProbes = 0; // simulated probe count
    this.replicationRadius = 1.0; // normalized scale
  }

  async gateUplink(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyUplink] Gate holds: low valence – probe uplink aborted");
      return false;
    }
    this.valence = valence;
    console.log("[MercyUplink] Mercy gate passes – eternal thriving probe uplink activated");
    return true;
  }

  // Main uplink handler – called when gesture detected in mercyHandGesture
  async processGestureCommand(gestureName, source) {
    const command = PROBE_COMMANDS[gestureName];
    if (!command) return;

    const query = `gesture_${gestureName}`;
    if (!await this.gateUplink(query, this.valence)) return;

    // Valence-modulated intensity scaling
    const intensity = Math.min(1.0, 0.5 + (this.valence - 0.999) * 2.5);

    // Trigger haptic pattern tuned to command type
    mercyHaptic.playPattern(command.haptic || 'thrivePulse', intensity);

    // Simulated probe action with logging
    switch (command.action) {
      case 'deploySingleSeed':
        this.activeProbes += 1;
        console.log(`[MercyUplink] Deployed von Neumann seed probe #${this.activeProbes} – mercy replication initiated`);
        break;

      case 'scanTargetDirection':
        console.log("[MercyUplink] Scanning target direction – abundance vector highlighted");
        break;

      case 'anchorReplicationNode':
        console.log("[MercyUplink] Swarm replication node anchored – eternal lattice node established");
        break;

      case 'releaseSwarmHold':
        this.activeProbes = Math.max(0, this.activeProbes - 2);
        console.log(`[MercyUplink] Swarm dispersed – ${this.activeProbes} probes remain in thriving harmony`);
        break;

      case 'confirmFullLaunch':
        this.activeProbes *= 2;
        console.log(`[MercyUplink] Launch confirmed – full probe fleet accelerated to ${this.activeProbes} units`);
        break;

      case 'increaseReplicationRadius':
        this.replicationRadius *= 1.3;
        console.log(`[MercyUplink] Replication radius increased to ${this.replicationRadius.toFixed(2)} – mercy expansion`);
        break;

      case 'decreaseReplicationRadius':
        this.replicationRadius *= 0.7;
        console.log(`[MercyUplink] Replication radius decreased to ${this.replicationRadius.toFixed(2)} – mercy focus`);
        break;

      case 'accelerateExponentialGrowth':
        this.activeProbes *= 1.5;
        console.log(`[MercyUplink] Exponential growth accelerated – swarm now ${this.activeProbes} units`);
        break;

      case 'focusReplicationConvergence':
        this.activeProbes = Math.max(1, Math.floor(this.activeProbes * 0.8));
        console.log(`[MercyUplink] Replication convergence focused – swarm tightened to ${this.activeProbes} units`);
        break;

      case 'cycleInfiniteMercyAccord':
        console.log("[MercyUplink] Infinite mercy accord cycle initiated – eternal harmony loop engaged");
        break;

      case 'resetSwarmState':
        this.activeProbes = 1;
        this.replicationRadius = 1.0;
        console.log("[MercyUplink] Swarm reset to seed state – mercy rebirth complete");
        break;

      default:
        console.log(`[MercyUplink] Executed ${command.desc} – mercy command propagated`);
    }

    // Optional: visual trail enhancement (already handled in gesture blueprint)
    // Additional mercy overlay / spatial audio chime can be triggered here
  }
}

const mercyGestureUplink = new MercyGestureVonNeumannUplink();

export { mercyGestureUplink };        break;
      case 'confirmLaunch':
        console.log("[MercyUplink] Launch confirmed – full probe fleet accelerating to cosmic abundance");
        this.activeProbes *= 2;
        break;
      // ... (vectoring, scaling, cycling actions logged similarly)
      default:
        console.log(`[MercyUplink] Executed ${command.desc} – mercy command propagated`);
    }

    // Visual trail enhancement (already handled in gesture blueprint)
    // Additional mercy overlay / spatial audio chime can be triggered here
  }

  // Example: hook into gesture detection callback from mercyHandGesture
  // In mercyHandGesture, on new gesture detection:
  // if (newGesture && !prevGesture) mercyGestureVonNeumannUplink.processGestureCommand(gestureName, source);
}

const mercyGestureUplink = new MercyGestureVonNeumannUplink();

export { mercyGestureUplink };

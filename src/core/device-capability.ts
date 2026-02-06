// src/core/device-capability.ts – Device Capability Detection & Adaptive Config v1
// Real-time hardware profiling, auto model/backend/fps selection, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

export interface DeviceProfile {
  deviceMemoryGB: number;           // navigator.deviceMemory (or estimated)
  cpuThreads: number;               // navigator.hardwareConcurrency
  hasWebGPU: boolean;
  hasWebNN: boolean;
  isHighEnd: boolean;
  recommendedModel: 'tiny' | 'medium' | 'large';
  recommendedBackend: 'webgpu' | 'webnn' | 'webgl' | 'wasm';
  recommendedFPS: number;           // target inference fps cap
  visualQuality: 'low' | 'medium' | 'high'; // particle field / orb detail
  batteryLevel?: number;            // if available
  isCharging?: boolean;
}

let cachedProfile: DeviceProfile | null = null;

export class DeviceCapability {
  static async detect(): Promise<DeviceProfile> {
    const actionName = 'Detect device capability';
    if (!await mercyGate(actionName)) {
      return this.getFallbackProfile();
    }

    if (cachedProfile) return cachedProfile;

    try {
      // 1. Basic signals
      const deviceMemoryGB = navigator.deviceMemory || 4; // fallback to 4 GB
      const cpuThreads = navigator.hardwareConcurrency || 4;

      // 2. WebGPU support
      let hasWebGPU = false;
      try {
        hasWebGPU = 'gpu' in navigator && !!(await navigator.gpu?.requestAdapter());
      } catch {}

      // 3. WebNN support
      let hasWebNN = 'ml' in navigator && 'createContext' in (navigator as any).ml;

      // 4. Battery status (optional)
      let batteryLevel: number | undefined;
      let isCharging: boolean | undefined;
      try {
        const battery = await (navigator as any).getBattery?.();
        if (battery) {
          batteryLevel = battery.level;
          isCharging = battery.charging;
        }
      } catch {}

      // 5. Heuristics for "high-end" device
      const isHighEnd = deviceMemoryGB >= 8 && cpuThreads >= 8 && hasWebGPU;

      // 6. Adaptive recommendations
      let model: 'tiny' | 'medium' | 'large' = 'tiny';
      let backend: 'webgpu' | 'webnn' | 'webgl' | 'wasm' = 'webgl';
      let fps = 15;
      let visual = 'low';

      if (isHighEnd && hasWebGPU) {
        model = 'medium';
        backend = hasWebNN ? 'webnn' : 'webgpu';
        fps = 60;
        visual = 'high';
      } else if (deviceMemoryGB >= 6 && cpuThreads >= 6) {
        model = 'tiny';
        backend = hasWebGPU ? 'webgpu' : 'webgl';
        fps = 30;
        visual = 'medium';
      } else {
        // Low-end / survival mode
        model = 'tiny';
        backend = 'webgl';
        fps = 15;
        visual = 'low';
      }

      // Battery-aware throttling
      if (batteryLevel !== undefined && batteryLevel < 0.3 && !isCharging) {
        fps = Math.max(10, fps - 10);
        visual = visual === 'high' ? 'medium' : 'low';
      }

      const profile: DeviceProfile = {
        deviceMemoryGB,
        cpuThreads,
        hasWebGPU,
        hasWebNN,
        isHighEnd,
        recommendedModel: model,
        recommendedBackend: backend,
        recommendedFPS: fps,
        visualQuality: visual,
        batteryLevel,
        isCharging
      };

      cachedProfile = profile;
      console.log("[DeviceCapability] Detected profile:", profile);
      return profile;
    } catch (e) {
      console.error("[DeviceCapability] Detection failed", e);
      return this.getFallbackProfile();
    }
  }

  private static getFallbackProfile(): DeviceProfile {
    return {
      deviceMemoryGB: 4,
      cpuThreads: 4,
      hasWebGPU: false,
      hasWebNN: false,
      isHighEnd: false,
      recommendedModel: 'tiny',
      recommendedBackend: 'webgl',
      recommendedFPS: 15,
      visualQuality: 'low'
    };
  }

  static async applyToEngine() {
    const profile = await this.detect();
    // Example usage in WebLLMEngine / ONNX / tfjs loaders
    await WebLLMEngine.switchModel(profile.recommendedModel);
    // Similar calls for tfjs backend preference, MediaPipe complexity, etc.
  }
}

export default DeviceCapability;

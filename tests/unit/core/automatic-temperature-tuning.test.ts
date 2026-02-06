// tests/unit/core/automatic-temperature-tuning.test.ts
// Vitest unit tests for SAC-style auto-α tuning logic
// MIT License – Autonomicity Games Inc. 2026

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getTemperature,
  updateTemperature,
  entropyBonus,
  trainingStepWithAutoTemp
} from '@/core/automatic-temperature-tuning';
import { currentValence } from '@/core/valence-tracker';

vi.mock('@/core/valence-tracker', () => ({
  currentValence: {
    get: vi.fn().mockReturnValue(0.95)
  }
}));

describe('automatic-temperature-tuning', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset internal logAlpha state between tests if needed
    // (implementation detail – may need manual reset if module stateful)
  });

  it('returns initial temperature when not updated', () => {
    expect(getTemperature()).toBeCloseTo(Math.exp(Math.log(0.2)), 5);
  });

  it('updates temperature toward target entropy with valence modulation', async () => {
    const batchEntropy = -Math.log(0.3); // \~1.2 nats
    const newAlpha = await updateTemperature(batchEntropy);

    expect(newAlpha).toBeGreaterThan(0);
    expect(newAlpha).toBeLessThanOrEqual(10);
    expect(currentValence.get).toHaveBeenCalled();
  });

  it('applies entropy bonus correctly', () => {
    const logProbs = [-0.5, -0.6, -0.4];
    const bonus = entropyBonus(logProbs);

    expect(bonus).toBeGreaterThan(0);
    expect(bonus).toBeCloseTo(0.5 * getTemperature() * 0.01, 5); // rough check
  });

  it('respects mercy gate in training step', async () => {
    const mercyGateMock = vi.mocked(mercyGate);
    mercyGateMock.mockResolvedValueOnce(false);

    const result = await trainingStepWithAutoTemp(-Math.log(0.3));
    expect(result).toBe(getTemperature());
  });
});

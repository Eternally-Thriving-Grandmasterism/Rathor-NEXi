// tests/unit/core/valence-projection.test.ts
// Vitest unit tests for future valence trajectory projection
// MIT License – Autonomicity Games Inc. 2026

import { describe, it, expect, vi, beforeEach } from 'vitest';
import ValenceProjection from '@/core/valence-projection';
import { currentValence } from '@/core/valence-tracker';

vi.mock('@/core/valence-tracker', () => ({
  currentValence: {
    get: vi.fn().mockReturnValue(0.95)
  }
}));

describe('ValenceProjection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('projects safe trajectory for high current valence', async () => {
    const result = await ValenceProjection.project(0.96);
    expect(result.isSafe).toBe(true);
    expect(result.projectedValence).toBeGreaterThanOrEqual(0.85);
    expect(result.trajectory.length).toBe(30);
  });

  it('rejects unsafe trajectory when projected drop is large', async () => {
    currentValence.get.mockReturnValueOnce(0.4);
    const result = await ValenceProjection.project();
    expect(result.isSafe).toBe(false);
    expect(result.dropFromBaseline).toBeGreaterThan(0.05);
  });

  it('uses provided baseline when specified', async () => {
    const result = await ValenceProjection.project(0.98);
    expect(result.projectedValence).toBeCloseTo(0.98, 2);
  });

  it('caches results for repeated calls', async () => {
    const first = await ValenceProjection.project();
    const second = await ValenceProjection.project();
    expect(second).toBe(first);
  });
});

// tests/unit/setup.ts – Vitest global setup for all unit tests
// Mocks common dependencies, sets up test environment
// MIT License – Autonomicity Games Inc. 2026

import { vi } from 'vitest';

// Mock mercyHaptic (no side effects in tests)
vi.mock('@/utils/haptic-utils', () => ({
  default: {
    playPattern: vi.fn()
  }
}));

// Mock mercyGate to always pass in unit tests (override in specific tests if needed)
vi.mock('@/core/mercy-gate', async () => {
  const actual = await vi.importActual('@/core/mercy-gate');
  return {
    ...actual,
    mercyGate: vi.fn().mockResolvedValue(true)
  };
});

// Mock currentValence for controlled testing
vi.mock('@/core/valence-tracker', () => ({
  currentValence: {
    get: vi.fn().mockReturnValue(0.95),
    subscribe: vi.fn()
  }
}));

// Global beforeEach / afterEach if needed
beforeEach(() => {
  vi.clearAllMocks();
});

export {};

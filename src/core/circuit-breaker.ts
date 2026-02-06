// src/core/circuit-breaker.ts – Circuit Breaker Pattern v1.0
// Prevents cascading failures, mercy-gated, valence-aware retry aggression
// States: CLOSED (normal), OPEN (failed, fast-fail), HALF_OPEN (testing recovery)
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const DEFAULT_FAILURE_THRESHOLD = 5;
const DEFAULT_RESET_TIMEOUT_MS = 30000;     // 30s before trying again
const DEFAULT_HALF_OPEN_MAX_ATTEMPTS = 3;
const VALENCE_AGGRESSION_BOOST = 1.8;       // high valence → faster recovery attempts
const VALENCE_AGGRESSION_DAMP = 0.6;        // low valence → slower recovery

type CircuitState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

interface CircuitBreakerConfig {
  name?: string;
  failureThreshold?: number;
  resetTimeoutMs?: number;
  halfOpenMaxAttempts?: number;
  onOpen?: () => void;
  onClose?: () => void;
  onHalfOpen?: () => void;
}

export class CircuitBreaker {
  private name: string;
  private state: CircuitState = 'CLOSED';
  private failureCount = 0;
  private successCountInHalfOpen = 0;
  private lastFailureTime = 0;
  private config: Required<CircuitBreakerConfig>;

  constructor(config: CircuitBreakerConfig = {}) {
    this.config = {
      name: config.name || 'unnamed-breaker',
      failureThreshold: config.failureThreshold ?? DEFAULT_FAILURE_THRESHOLD,
      resetTimeoutMs: config.resetTimeoutMs ?? DEFAULT_RESET_TIMEOUT_MS,
      halfOpenMaxAttempts: config.halfOpenMaxAttempts ?? DEFAULT_HALF_OPEN_MAX_ATTEMPTS,
      onOpen: config.onOpen || (() => {}),
      onClose: config.onClose || (() => {}),
      onHalfOpen: config.onHalfOpen || (() => {})
    };
  }

  private getValenceFactor(): number {
    const valence = currentValence.get();
    return VALENCE_AGGRESSION_DAMP + (VALENCE_AGGRESSION_BOOST - VALENCE_AGGRESSION_DAMP) * valence;
  }

  private getAdjustedResetTimeout(): number {
    return Math.round(this.config.resetTimeoutMs * this.getValenceFactor());
  }

  private getAdjustedFailureThreshold(): number {
    return Math.round(this.config.failureThreshold * this.getValenceFactor());
  }

  private transitionToOpen() {
    this.state = 'OPEN';
    this.lastFailureTime = Date.now();
    this.failureCount = this.getAdjustedFailureThreshold();
    this.config.onOpen();
    mercyHaptic.playPattern('warningPulse', currentValence.get() * 0.7);
    console.warn(`[CircuitBreaker:${this.config.name}] OPEN – failures: ${this.failureCount}`);
  }

  private transitionToHalfOpen() {
    this.state = 'HALF_OPEN';
    this.successCountInHalfOpen = 0;
    this.config.onHalfOpen();
    mercyHaptic.playPattern('neutralPulse', currentValence.get());
    console.log(`[CircuitBreaker:${this.config.name}] HALF_OPEN – testing recovery`);
  }

  private transitionToClosed() {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.successCountInHalfOpen = 0;
    this.config.onClose();
    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log(`[CircuitBreaker:${this.config.name}] CLOSED – recovered`);
  }

  /**
   * Execute operation with circuit breaker protection
   * @param operation async function to run
   * @returns result or throws if OPEN or final failure
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    const now = Date.now();

    // Fast-fail when OPEN
    if (this.state === 'OPEN') {
      const timeSinceFailure = now - this.lastFailureTime;
      if (timeSinceFailure < this.getAdjustedResetTimeout()) {
        throw new Error(`Circuit OPEN for ${this.config.name} – wait ${Math.ceil((this.getAdjustedResetTimeout() - timeSinceFailure)/1000)}s`);
      }
      this.transitionToHalfOpen();
    }

    try {
      const result = await operation();

      // Success handling
      if (this.state === 'HALF_OPEN') {
        this.successCountInHalfOpen++;
        if (this.successCountInHalfOpen >= this.config.halfOpenMaxAttempts) {
          this.transitionToClosed();
        }
      } else if (this.state === 'CLOSED') {
        this.failureCount = Math.max(0, this.failureCount - 1);
      }

      return result;
    } catch (error) {
      // Failure handling
      this.failureCount++;

      if (this.state === 'CLOSED' && this.failureCount >= this.getAdjustedFailureThreshold()) {
        this.transitionToOpen();
      }

      throw error;
    }
  }

  /**
   * Convenience wrapper for fetch-like operations
   */
  async fetchWithBreaker<T>(
    input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<Response> {
    return this.execute(async () => {
      const response = await fetch(input, init);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response;
    });
  }

  /**
   * Get current circuit state (for monitoring/dashboard)
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get health metrics
   */
  getMetrics() {
    return {
      name: this.config.name,
      state: this.state,
      failureCount: this.failureCount,
      successCountInHalfOpen: this.successCountInHalfOpen,
      currentAlpha: getTemperature(),
      currentValence: currentValence.get()
    };
  }
}

/**
 * Global registry of circuit breakers (per service/operation)
 */
const circuitBreakers = new Map<string, CircuitBreaker>();

export function getCircuitBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
  if (!circuitBreakers.has(name)) {
    circuitBreakers.set(name, new CircuitBreaker({ name, ...config }));
  }
  return circuitBreakers.get(name)!;
}

// Example usage in fetch-heavy code
export async function fetchModel(url: string) {
  const breaker = getCircuitBreaker('model-fetch', {
    failureThreshold: 4,
    resetTimeoutMs: 15000,
    onOpen: () => console.warn(`Circuit OPEN for model-fetch – protecting downstream`)
  });

  return breaker.fetchWithBreaker(url);
}    console.log(`[CircuitBreaker:${this.config.name}] CLOSED – recovered`);
  }

  /**
   * Execute operation with circuit breaker protection
   * @param operation async function to run
   * @returns result or throws if OPEN or final failure
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    const now = Date.now();

    // Fast-fail when OPEN
    if (this.state === 'OPEN') {
      const timeSinceFailure = now - this.lastFailureTime;
      if (timeSinceFailure < this.getAdjustedResetTimeout()) {
        throw new Error(`Circuit OPEN for ${this.config.name} – wait ${Math.ceil((this.getAdjustedResetTimeout() - timeSinceFailure)/1000)}s`);
      }
      this.transitionToHalfOpen();
    }

    try {
      const result = await operation();

      // Success handling
      if (this.state === 'HALF_OPEN') {
        this.successCountInHalfOpen++;
        if (this.successCountInHalfOpen >= this.config.halfOpenMaxAttempts) {
          this.transitionToClosed();
        }
      } else if (this.state === 'CLOSED') {
        this.failureCount = Math.max(0, this.failureCount - 1);
      }

      return result;
    } catch (error) {
      // Failure handling
      this.failureCount++;

      if (this.state === 'CLOSED' && this.failureCount >= this.getAdjustedFailureThreshold()) {
        this.transitionToOpen();
      }

      throw error;
    }
  }

  /**
   * Convenience wrapper for fetch-like operations
   */
  async fetchWithBreaker<T>(
    input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<Response> {
    return this.execute(async () => {
      const response = await fetch(input, init);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response;
    });
  }

  /**
   * Get current circuit state (for monitoring/dashboard)
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get health metrics
   */
  getMetrics() {
    return {
      name: this.config.name,
      state: this.state,
      failureCount: this.failureCount,
      successCountInHalfOpen: this.successCountInHalfOpen,
      currentAlpha: getTemperature(),
      currentValence: currentValence.get()
    };
  }
}

/**
 * Global registry of circuit breakers (per service/operation)
 */
const circuitBreakers = new Map<string, CircuitBreaker>();

export function getCircuitBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
  if (!circuitBreakers.has(name)) {
    circuitBreakers.set(name, new CircuitBreaker({ name, ...config }));
  }
  return circuitBreakers.get(name)!;
}

// Example usage in fetch-heavy code
export async function fetchModel(url: string) {
  const breaker = getCircuitBreaker('model-fetch', {
    failureThreshold: 4,
    resetTimeoutMs: 15000,
    onOpen: () => console.warn(`Circuit OPEN for model-fetch – protecting downstream`)
  });

  return breaker.fetchWithBreaker(url);
}

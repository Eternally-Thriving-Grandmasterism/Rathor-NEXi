// src/core/error-retry.ts – Error Retry Logic Layer v1.0
// Exponential backoff with jitter, max attempts, mercy gating, valence-modulated aggression
// Works for network fetches, chunk loading, ML init, sync operations, etc.
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const BASE_RETRY_DELAY_MS = 500;
const MAX_RETRY_DELAY_MS = 10000;
const MAX_RETRIES_DEFAULT = 4;
const JITTER_FACTOR = 0.3;                    // random jitter ±30%
const VALENCE_RETRY_BOOST = 1.8;              // high valence → faster/more retries
const VALENCE_RETRY_DAMP = 0.6;               // low valence → slower/fewer retries

interface RetryConfig {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  jitter?: number;
  onRetry?: (attempt: number, error: any) => void;
  onSuccess?: () => void;
  onFinalFailure?: (error: any) => void;
}

export async function withRetry<T>(
  operation: () => Promise<T>,
  config: RetryConfig = {}
): Promise<T> {
  const actionName = 'Retry operation with mercy gating';
  if (!await mercyGate(actionName)) {
    throw new Error('Mercy gate blocked retry operation');
  }

  const {
    maxRetries = MAX_RETRIES_DEFAULT,
    baseDelay = BASE_RETRY_DELAY_MS,
    maxDelay = MAX_RETRY_DELAY_MS,
    jitter = JITTER_FACTOR,
    onRetry = () => {},
    onSuccess = () => {},
    onFinalFailure = () => {}
  } = config;

  const valence = currentValence.get();

  // Valence modulates retry aggression
  const valenceFactor = VALENCE_RETRY_DAMP + (VALENCE_RETRY_BOOST - VALENCE_RETRY_DAMP) * valence;
  const adjustedMaxRetries = Math.round(maxRetries * valenceFactor);
  const adjustedBaseDelay = Math.round(baseDelay * valenceFactor);

  let lastError: any = null;

  for (let attempt = 0; attempt <= adjustedMaxRetries; attempt++) {
    try {
      const result = await operation();
      onSuccess();
      mercyHaptic.playPattern('cosmicHarmony', valence);
      return result;
    } catch (error) {
      lastError = error;
      console.warn(`[Retry] Attempt \( {attempt + 1}/ \){adjustedMaxRetries + 1} failed:`, error);

      if (attempt === adjustedMaxRetries) {
        onFinalFailure(error);
        throw error;
      }

      onRetry(attempt + 1, error);

      // Exponential backoff with jitter
      const delay = Math.min(
        adjustedBaseDelay * Math.pow(2, attempt),
        maxDelay
      ) * (1 + (Math.random() * 2 - 1) * jitter);

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  // Should never reach here
  throw lastError;
}

/**
 * Convenience wrapper for fetch-like operations
 */
export async function fetchWithRetry<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
  config?: RetryConfig
): Promise<Response> {
  return withRetry(async () => {
    const response = await fetch(input, init);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  }, {
    ...config,
    maxRetries: config?.maxRetries ?? 5,
    baseDelay: config?.baseDelay ?? 800
  });
}

/**
 * Example usage in chunk loading / ML init
 */
export async function loadCriticalChunk(url: string): Promise<any> {
  return withRetry(async () => {
    const module = await import(/* @vite-ignore */ url);
    return module;
  }, {
    maxRetries: 6,
    baseDelay: 600,
    onRetry: (attempt, err) => {
      console.warn(`Chunk retry ${attempt}: ${err.message}`);
      mercyHaptic.playPattern('warningPulse', currentValence.get() * 0.7);
    },
    onFinalFailure: (err) => {
      console.error('Critical chunk failed all retries:', err);
    }
  });
}

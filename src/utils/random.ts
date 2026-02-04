/**
 * This file provides deterministic random number generation utilities.
 *
 * Role in system:
 * - Enables reproducible stochastic optimizers (e.g., CMA-ES) via explicit seeds
 * - Provides uniform and standard normal sampling without external dependencies
 * - Keeps browser compatibility (no Node-specific APIs required)
 *
 * For first-time readers:
 * - Use createSeededRandom(seed) to get a generator
 * - Call nextUniform() for U[0,1), nextStandardNormal() for N(0,1)
 * - Seed behavior: seed > 0 is deterministic; seed is auto-generated if 0/undefined
 */

export type SeededRandom = {
  seed: number;
  nextUniform: () => number;
  nextStandardNormal: () => number;
};

const UINT32_MAX_PLUS_ONE = 2 ** 32;
const AUTO_SEED_MULTIPLIER = 1664525;
const AUTO_SEED_INCREMENT = 1013904223;
const TWO_PI = 2.0 * Math.PI;
const MINIMUM_POSITIVE_UNIFORM = 1e-12; // Guard against log(0) in Box–Muller

function coerceToUint32(value: number): number {
  // WHY: We use uint32 arithmetic for deterministic, portable PRNG behavior.
  return value >>> 0;
}

function computeAutoSeed(): number {
  // WHY: When seed is not forced, we want a different stream each run.
  // We mix time and Math.random() so that environments with coarse timers still vary.
  const timeSeed = coerceToUint32(Date.now());
  const randomSeed = coerceToUint32(Math.floor(Math.random() * UINT32_MAX_PLUS_ONE));
  const mixed = coerceToUint32(timeSeed ^ randomSeed);
  // One LCG step to further diffuse obvious patterns like time-only seeds.
  return coerceToUint32(mixed * AUTO_SEED_MULTIPLIER + AUTO_SEED_INCREMENT);
}

function mulberry32(nextState: { value: number }): number {
  // Reference: Mulberry32 PRNG (fast, decent quality for optimization sampling).
  // State is kept in an object so callers can share and advance it.
  let t = (nextState.value += 0x6d2b79f5);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  const result = (t ^ (t >>> 14)) >>> 0;
  return result / UINT32_MAX_PLUS_ONE;
}

function computeUniformOpenInterval(nextUniform: () => number): number {
  const u = nextUniform();
  // Guard: avoid exactly 0 which would break Box–Muller.
  return u <= 0.0 ? MINIMUM_POSITIVE_UNIFORM : u;
}

export function createSeededRandom(seed: number | undefined): SeededRandom {
  const resolvedSeed = seed && seed > 0 ? coerceToUint32(seed) : computeAutoSeed();
  const state = { value: resolvedSeed };
  let cachedNormal: number | undefined;

  function nextUniform(): number {
    return mulberry32(state);
  }

  function nextStandardNormal(): number {
    if (cachedNormal !== undefined) {
      const value = cachedNormal;
      cachedNormal = undefined;
      return value;
    }

    // Box–Muller transform (polar form would require rejection; classic is fine here).
    const u1 = computeUniformOpenInterval(nextUniform);
    const u2 = nextUniform();
    const radius = Math.sqrt(-2.0 * Math.log(u1));
    const angle = TWO_PI * u2;
    const z0 = radius * Math.cos(angle);
    const z1 = radius * Math.sin(angle);
    cachedNormal = z1;
    return z0;
  }

  return { seed: resolvedSeed, nextUniform, nextStandardNormal };
}


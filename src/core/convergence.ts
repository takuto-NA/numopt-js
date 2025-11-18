/**
 * This file provides helper functions for convergence checking across
 * different optimization algorithms.
 * 
 * Role in system:
 * - Centralizes convergence checking logic (DRY principle)
 * - Used by gradient descent, Gauss-Newton, and Levenberg-Marquardt
 * - Provides consistent convergence criteria
 * 
 * For first-time readers:
 * - These are utility functions used by optimization algorithms
 * - Each function checks a specific convergence criterion
 */

import type { OptimizationResult } from './types';

/**
 * Creates a convergence result object with consistent structure.
 * Used to avoid code duplication across optimization algorithms.
 */
export function createConvergenceResult(
  parameters: Float64Array,
  iteration: number,
  converged: boolean,
  finalCost: number,
  finalGradientNorm?: number
): OptimizationResult {
  return {
    parameters,
    iterations: iteration + 1,
    converged,
    finalCost,
    finalGradientNorm
  };
}

/**
 * Checks if gradient norm indicates convergence.
 * Returns true if gradient is small enough (algorithm has found a stationary point).
 */
export function checkGradientConvergence(
  gradientNorm: number,
  tolerance: number,
  iteration: number
): boolean {
  // Skip convergence check on first iteration (no step taken yet)
  return iteration > 0 && gradientNorm < tolerance;
}

/**
 * Checks if step size indicates convergence.
 * Returns true if step is small enough (algorithm is making minimal progress).
 */
export function checkStepSizeConvergence(
  stepNorm: number,
  tolerance: number,
  iteration: number
): boolean {
  // Skip convergence check on first iteration (no step taken yet)
  return iteration > 0 && stepNorm < tolerance;
}

/**
 * Checks if residual norm indicates convergence.
 * Returns true if residual is small enough (problem is solved to desired accuracy).
 */
export function checkResidualConvergence(
  residualNorm: number,
  tolerance: number,
  iteration: number
): boolean {
  // Skip convergence check on first iteration (no step taken yet)
  return iteration > 0 && residualNorm < tolerance;
}


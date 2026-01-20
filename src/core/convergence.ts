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

import type { OptimizationResult } from './types.js';

/**
 * Creates a convergence result object with consistent structure.
 * Used to avoid code duplication across optimization algorithms.
 */
export function createConvergenceResult(
  finalParameters: Float64Array,
  iteration: number,
  converged: boolean,
  finalCost: number,
  finalGradientNorm?: number
): OptimizationResult {
  return {
    finalParameters,
    parameters: finalParameters,
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
  // NOTE:
  // If the initial point is already a stationary point (||∇f(x0)|| ≈ 0),
  // optimizers should report convergence immediately. Skipping iteration 0
  // can incorrectly trigger line-search failures (e.g., zero search direction).
  void iteration; // kept for backward-compatible signature
  return gradientNorm < tolerance;
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
  // A tiny step on the first iteration is still a valid convergence signal
  // (e.g., already near the optimum).
  void iteration; // kept for backward-compatible signature
  return stepNorm < tolerance;
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
  // If the initial residual is already small, we should converge immediately.
  void iteration; // kept for backward-compatible signature
  return residualNorm < tolerance;
}


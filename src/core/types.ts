/**
 * This file defines the core type definitions for the numopt-js library.
 * 
 * Role in system:
 * - Provides type contracts for all optimization algorithms
 * - Defines function signatures for cost functions, gradients, and Jacobians
 * - Establishes option and result interfaces for consistent API design
 * 
 * For first-time readers:
 * - Start with ResidualFn and CostFn to understand function signatures
 * - Check option interfaces to see what can be configured
 * - Review result interfaces to understand what each algorithm returns
 */

import { Matrix } from 'ml-matrix';

/**
 * Function that computes the residual vector for nonlinear least squares problems.
 * Takes parameter vector and returns residual vector.
 * 
 * Note: Uses Float64Array for performance and type safety in numerical computations.
 */
export type ResidualFn = (parameters: Float64Array) => Float64Array;

/**
 * Function that computes the Jacobian matrix for nonlinear least squares problems.
 * Takes parameter vector and returns Jacobian matrix (ml-matrix Matrix type).
 * 
 * Note: Uses Matrix from ml-matrix package for efficient matrix operations.
 * The Matrix type provides optimized linear algebra operations and is browser-compatible.
 */
export type JacobianFn = (parameters: Float64Array) => Matrix;

/**
 * Function that computes the cost (objective function value) for general optimization.
 * Takes parameter vector and returns scalar cost value.
 * 
 * Note: Uses Float64Array for parameter vector to ensure 64-bit floating-point precision
 * and better performance in numerical computations.
 */
export type CostFn = (parameters: Float64Array) => number;

/**
 * Function that computes the gradient vector for general optimization.
 * Takes parameter vector and returns gradient vector.
 * 
 * Note: Uses Float64Array for performance and memory efficiency in numerical computations.
 */
export type GradientFn = (parameters: Float64Array) => Float64Array;

/**
 * Common options shared across optimization algorithms.
 */
export interface CommonOptimizationOptions {
  /**
   * Maximum number of iterations before stopping.
   * Default: 1000
   */
  maxIterations?: number;

  /**
   * Tolerance for convergence check (gradient norm, step size, etc.).
   * Default: 1e-6
   */
  tolerance?: number;

  /**
   * Callback function called at each iteration for progress monitoring.
   * Useful for debugging and monitoring convergence.
   */
  onIteration?: (iteration: number, cost: number, parameters: Float64Array) => void;

  /**
   * Enable verbose logging for debugging.
   * When true, detailed information is logged to console.
   * Default: false
   */
  verbose?: boolean;
}

/**
 * Options specific to gradient descent algorithm.
 */
export interface GradientDescentOptions extends CommonOptimizationOptions {
  /**
   * Step size (learning rate) for gradient descent.
   * If not provided, line search will be used to determine step size.
   * Default: undefined (use line search)
   */
  stepSize?: number;

  /**
   * Use line search to determine optimal step size.
   * Default: true
   */
  useLineSearch?: boolean;
}

/**
 * Options for line search algorithm.
 */
export interface LineSearchOptions {
  /**
   * Initial step size to try.
   * Default: 1.0
   */
  initialStepSize?: number;

  /**
   * Contraction factor for backtracking line search.
   * Step size is multiplied by this factor when condition is not met.
   * Default: 0.5
   */
  contractionFactor?: number;

  /**
   * Armijo condition parameter (sufficient decrease).
   * Default: 0.1
   */
  armijoParameter?: number;

  /**
   * Maximum number of line search iterations.
   * Default: 50
   */
  maxIterations?: number;
}

/**
 * Options for numerical differentiation.
 */
export interface NumericalDifferentiationOptions {
  /**
   * Step size for finite difference approximation.
   * Default: 1e-6
   */
  stepSize?: number;
}

/**
 * Options for Gauss-Newton method.
 */
export interface GaussNewtonOptions extends CommonOptimizationOptions {
  /**
   * Analytical Jacobian function. If provided, this will be used instead of numerical differentiation.
   * If not provided, numerical Jacobian will be used (if useNumericJacobian is true).
   */
  jacobian?: JacobianFn;

  /**
   * Use numerical differentiation to compute Jacobian if user doesn't provide it.
   * Default: true
   */
  useNumericJacobian?: boolean;

  /**
   * Step size for numerical Jacobian computation.
   * Default: 1e-6
   */
  jacobianStep?: number;
}

/**
 * Options for Levenberg-Marquardt algorithm.
 */
export interface LevenbergMarquardtOptions extends GaussNewtonOptions {
  /**
   * Initial value of damping parameter lambda.
   * Default: 1e-3
   */
  lambdaInitial?: number;

  /**
   * Factor for updating lambda (success: divide, failure: multiply).
   * Default: 10.0
   */
  lambdaFactor?: number;

  /**
   * Tolerance for gradient norm convergence check.
   * Default: 1e-6
   */
  tolGradient?: number;

  /**
   * Tolerance for step size convergence check.
   * Default: 1e-6
   */
  tolStep?: number;

  /**
   * Tolerance for residual norm convergence check.
   * Default: 1e-6
   */
  tolResidual?: number;
}

/**
 * Result returned by optimization algorithms.
 */
export interface OptimizationResult {
  /**
   * Optimized parameter vector.
   */
  parameters: Float64Array;

  /**
   * Number of iterations performed.
   */
  iterations: number;

  /**
   * Whether the algorithm converged successfully.
   */
  converged: boolean;

  /**
   * Final cost (objective function value).
   */
  finalCost: number;

  /**
   * Final gradient norm (if applicable).
   */
  finalGradientNorm?: number;

  /**
   * Final residual norm (for least squares problems).
   * Available for algorithms that work with residual functions.
   */
  finalResidualNorm?: number;
}

/**
 * Result returned by Levenberg-Marquardt algorithm.
 */
export interface LevenbergMarquardtResult extends OptimizationResult {
  /**
   * Final residual norm.
   */
  finalResidualNorm: number;

  /**
   * Final lambda (damping parameter) value.
   */
  finalLambda: number;
}

/**
 * Result returned by gradient descent algorithm.
 */
export interface GradientDescentResult extends OptimizationResult {
  /**
   * Whether line search was used.
   */
  usedLineSearch: boolean;
}


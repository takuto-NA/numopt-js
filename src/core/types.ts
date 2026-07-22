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
 * Function that computes the constraint vector for constrained optimization problems.
 * Takes parameter vector and state vector, returns constraint vector.
 * The constraint c(p, x) = 0 must be satisfied.
 * 
 * Note: The constraint vector length and state vector length can differ.
 * The adjoint method supports both square and non-square constraint Jacobians.
 */
export type ConstraintFn = (parameters: Float64Array, states: Float64Array) => Float64Array;

/**
 * Function that computes the cost (objective function value) for constrained optimization.
 * Takes parameter vector and state vector, returns scalar cost value.
 * 
 * Note: The state vector x must satisfy c(p, x) = 0.
 */
export type ConstrainedCostFn = (parameters: Float64Array, states: Float64Array) => number;

/**
 * Function that computes the residual vector for constrained nonlinear least squares problems.
 * Takes parameter vector and state vector, returns residual vector.
 * The cost function is f(p, x) = 1/2 r(p, x)^T r(p, x).
 * 
 * Note: The state vector x must satisfy c(p, x) = 0.
 */
export type ConstrainedResidualFn = (parameters: Float64Array, states: Float64Array) => Float64Array;

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
   * 
   * @deprecated Use logLevel instead for more fine-grained control.
   * If both logLevel and verbose are specified, logLevel takes precedence.
   */
  verbose?: boolean;

  /**
   * Log level for detailed logging output.
   * Controls which log messages are displayed:
   * - DEBUG: Detailed progress information (cost, gradient norm, step size, etc.)
   * - INFO: Convergence messages and important state changes
   * - WARN: Warnings (singular matrix, max iterations reached, line search failure, etc.)
   * - ERROR: Fatal errors (currently not used, reserved for future extensions)
   * 
   * If verbose is true and logLevel is not specified, logLevel defaults to INFO.
   * If both logLevel and verbose are specified, logLevel takes precedence.
   * Default: undefined (no logging)
   */
  logLevel?: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
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
   * If not provided, the initial step size is automatically scaled by the gradient norm:
   * α₀ = 1.0 / ||∇f(x)||
   * This prevents steps from being too large when gradients are large, improving
   * convergence performance. If the gradient norm is very small (< 1e-10) or the
   * computed step size is not finite, the default value of 1.0 is used.
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
 * Options for Strong Wolfe line search (Nocedal & Wright, 2nd ed., Algorithm 3.5).
 *
 * This line search aims to satisfy both:
 * - sufficient decrease (Armijo) and
 * - curvature condition (Strong Wolfe)
 *
 * Strong Wolfe is commonly used with quasi-Newton methods (BFGS / L-BFGS) because
 * it tends to produce steps that satisfy the curvature condition \(s^T y > 0\),
 * which helps keep Hessian approximations well-behaved.
 */
export interface StrongWolfeLineSearchOptions {
  /**
   * Initial step size to try.
   * If not provided, the initial step size is scaled by the gradient norm:
   * α₀ = 1.0 / ||∇f(x)||
   */
  initialStepSize?: number;

  /**
   * Armijo parameter c1 for sufficient decrease condition.
   * Typical value: 1e-4.
   */
  wolfeC1?: number;

  /**
   * Curvature parameter c2 for Strong Wolfe condition.
   * Typical value: 0.9.
   */
  wolfeC2?: number;

  /**
   * Maximum number of outer iterations (bracketing phase).
   * Default: 25
   */
  maxIterations?: number;

  /**
   * Maximum number of zoom iterations (within a bracket).
   * Default: 25
   */
  maxZoomIterations?: number;

  /**
   * Growth factor for expanding the trial step size when still in the bracketing phase.
   * Default: 2.0
   */
  stepSizeGrowthFactor?: number;
}

/**
 * Options for L-BFGS (limited-memory BFGS).
 */
export interface LbfgsOptions extends CommonOptimizationOptions {
  /**
   * Maximum number of correction pairs to store.
   * Default: 10
   */
  historySize?: number;

  /**
   * Whether to use Strong Wolfe line search.
   * Default: true
   */
  useLineSearch?: boolean;

  /**
   * Options passed to Strong Wolfe line search.
   */
  lineSearchOptions?: StrongWolfeLineSearchOptions;

  /**
   * Fixed step size used only when useLineSearch is false.
   * Default: 1.0
   */
  stepSize?: number;
}

/**
 * Options for dense BFGS (stores a full inverse Hessian approximation).
 */
export interface BfgsOptions extends CommonOptimizationOptions {
  /**
   * Whether to use Strong Wolfe line search.
   * Default: true
   */
  useLineSearch?: boolean;

  /**
   * Options passed to Strong Wolfe line search.
   */
  lineSearchOptions?: StrongWolfeLineSearchOptions;

  /**
   * Fixed step size used only when useLineSearch is false.
   * Default: 1.0
   */
  stepSize?: number;
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
   * Final optimized parameter vector.
   */
  finalParameters: Float64Array;

  /**
   * Final optimized parameter vector.
   *
   * @deprecated Use finalParameters instead. This alias will be removed in a future release.
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

/**
 * Options for adjoint gradient descent algorithm.
 */
export interface AdjointGradientDescentOptions extends GradientDescentOptions {
  /**
   * Analytical partial derivative of cost function with respect to parameters.
   * If provided, this will be used instead of numerical differentiation.
   * Function signature: (p: Float64Array, x: Float64Array) => Float64Array
   */
  dfdp?: (parameters: Float64Array, states: Float64Array) => Float64Array;

  /**
   * Analytical partial derivative of cost function with respect to states.
   * If provided, this will be used instead of numerical differentiation.
   * Function signature: (p: Float64Array, x: Float64Array) => Float64Array
   */
  dfdx?: (parameters: Float64Array, states: Float64Array) => Float64Array;

  /**
   * Analytical partial derivative of constraint function with respect to parameters.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (constraintCount × parameterCount).
   */
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Analytical partial derivative of constraint function with respect to states.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (constraintCount × stateCount).
   *
   * The adjoint method supports both square and non-square constraint Jacobians:
   * - If square, it solves (∂c/∂x)^T λ = rhs directly.
   * - If non-square, it solves the system in a least-squares sense.
   *
   * Note: Non-square (or ill-conditioned) Jacobians can be numerically sensitive.
   * Consider scaling/normalizing your states and constraints if you see instability.
   */
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Step size for numerical differentiation with respect to parameters.
   * Default: 1e-6
   */
  stepSizeP?: number;

  /**
   * Step size for numerical differentiation with respect to states.
   * Default: 1e-6
   */
  stepSizeX?: number;

  /**
   * Tolerance for checking constraint satisfaction c(p, x) = 0.
   * If ||c(p, x)|| exceeds this value, a warning will be issued.
   * Default: 1e-6
   */
  constraintTolerance?: number;

  /**
   * Base Tikhonov regularization for adjoint / least-squares solves involving ∂c/∂x.
   * Increase for ill-conditioned constraint Jacobians (e.g. densely sampled curve constraints).
   * Default: 0 (an automatic floor may still apply when the Jacobian is numerically singular)
   */
  regularization?: number;
}

/**
 * Result returned by adjoint gradient descent algorithm.
 */
export interface AdjointGradientDescentResult extends GradientDescentResult {
  /**
   * Final state vector (satisfies constraint c(p, x) = 0).
   */
  finalStates: Float64Array;

  /**
   * Final constraint violation norm ||c(p, x)||.
   */
  finalConstraintNorm?: number;
}

/**
 * Options for constrained Gauss-Newton method.
 */
export interface ConstrainedGaussNewtonOptions extends CommonOptimizationOptions {
  /**
   * Analytical partial derivative of residual function with respect to parameters.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (residualCount × parameterCount).
   */
  drdp?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Analytical partial derivative of residual function with respect to states.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (residualCount × stateCount).
   */
  drdx?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Analytical partial derivative of constraint function with respect to parameters.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (constraintCount × parameterCount).
   */
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Analytical partial derivative of constraint function with respect to states.
   * If provided, this will be used instead of numerical differentiation.
   * Returns a Matrix of size (constraintCount × stateCount).
   * Supports both square (constraintCount == stateCount) and non-square matrices.
   * For non-square matrices, the adjoint method uses normal equations with Cholesky decomposition.
   */
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix;

  /**
   * Step size for numerical differentiation with respect to parameters.
   * Default: 1e-6
   */
  stepSizeP?: number;

  /**
   * Step size for numerical differentiation with respect to states.
   * Default: 1e-6
   */
  stepSizeX?: number;

  /**
   * Tolerance for checking constraint satisfaction c(p, x) = 0.
   * If ||c(p, x)|| exceeds this value, a warning will be issued.
   * Default: 1e-6
   */
  constraintTolerance?: number;
}

/**
 * Result returned by constrained Gauss-Newton algorithm.
 */
export interface ConstrainedGaussNewtonResult extends OptimizationResult {
  /**
   * Final state vector (satisfies constraint c(p, x) = 0).
   */
  finalStates: Float64Array;

  /**
   * Final constraint violation norm ||c(p, x)||.
   */
  finalConstraintNorm?: number;
}

/**
 * Options for constrained Levenberg-Marquardt algorithm.
 */
export interface ConstrainedLevenbergMarquardtOptions extends ConstrainedGaussNewtonOptions {
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
 * Result returned by constrained Levenberg-Marquardt algorithm.
 */
export interface ConstrainedLevenbergMarquardtResult extends ConstrainedGaussNewtonResult {
  /**
   * Final lambda (damping parameter) value.
   */
  finalLambda: number;
}

/**
 * Options for CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * This implementation follows the vanilla CMA-ES default parameter formulas and
 * core termination criteria used by libcmaes (CMAES_DEFAULT).
 */
export interface CmaEsOptions extends CommonOptimizationOptions {
  /**
   * Population size (λ), number of candidate solutions sampled per generation.
   *
   * If not provided, uses libcmaes default:
   * λ = 4 + floor(3 * log(dim))
   */
  populationSize?: number;

  /**
   * Initial distribution step size σ₀ (sigma0 in libcmaes).
   *
   * This strongly affects performance; the optimum is ideally within
   * [x0 - σ0, x0 + σ0] (per dimension) in a reasonable scaling of parameters.
   *
   * If not provided (or non-positive), falls back to 1 / dim (libcmaes behavior).
   */
  initialStepSize?: number;

  /**
   * Random seed for reproducible results.
   *
   * libcmaes semantics:
   * - seed > 0: deterministic stream
   * - seed == 0 or unspecified: auto-generated seed (non-deterministic)
   */
  randomSeed?: number;

  /**
   * Maximum number of objective function evaluations (MAXFEVALS in libcmaes).
   * If specified, CMA-ES stops when the evaluation budget is reached.
   */
  maxFunctionEvaluations?: number;

  /**
   * Objective function target value (FTARGET in libcmaes).
   * If specified, CMA-ES stops successfully when best cost <= targetCost.
   */
  targetCost?: number;

  /**
   * Function value tolerance for TolHistFun termination criterion.
   * Stops successfully when the range of recent best costs becomes small enough.
   *
   * Default: 1e-12 (libcmaes)
   */
  functionTolerance?: number;

  /**
   * Parameter tolerance for TolX termination criterion.
   * Stops (partial success) when the distribution becomes sufficiently narrow.
   *
   * Default: 1e-12 (libcmaes)
   */
  parameterTolerance?: number;

  /**
   * Initial diagonal regularization applied when covariance Cholesky fails.
   * Increase this if you observe repeated numerical failures.
   *
   * Default: 1e-12
   */
  covarianceRegularization?: number;

  /**
   * Maximum size of the best-cost history buffer used by termination criteria.
   * If not provided (or non-positive), defaults to:
   * 10 + ceil(30 * dim / λ) (libcmaes)
   */
  maxHistorySize?: number;

  /**
   * Restart strategy for CMA-ES.
   * - "none": single run (default)
   * - "ipop": increasing population size restarts (λ doubles each restart)
   */
  restartStrategy?: 'none' | 'ipop';

  /**
   * Maximum number of restarts for IPOP.
   * If not provided, defaults to 9 (libcmaes).
   */
  maxRestarts?: number;

  /**
   * Enable lightweight profiling of CMA-ES internal timings.
   * When true, timing breakdown is returned in the result.
   */
  profiling?: boolean;
}

/**
 * Result returned by CMA-ES optimization.
 */
export interface CmaEsResult extends OptimizationResult {
  /**
   * Population size (λ) used during optimization.
   */
  populationSize: number;

  /**
   * Total number of objective function evaluations performed.
   */
  functionEvaluations: number;

  /**
   * Final step size σ.
   */
  finalStepSize: number;

  /**
   * Maximum standard deviation across coordinates: max_i sqrt(C_ii) * σ.
   * Useful for assessing the final search radius.
   */
  finalMaxStdDev: number;

  /**
   * Stop reason for the overall optimization run.
   */
  stopReason?: 'MAXITER' | 'MAXFEVALS' | 'FTARGET' | 'TOLHISTFUN' | 'TOLX' | 'IPOP_MAX_RESTARTS';

  /**
   * Optional profiling breakdown (milliseconds).
   */
  profiling?: {
    totalMs: number;
    costMs: number;
    choleskyMs: number;
    samplingMs: number;
    updateMs: number;
  };
}


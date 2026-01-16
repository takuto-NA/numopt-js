/**
 * This file implements the adjoint method for constrained optimization problems.
 * 
 * The adjoint method efficiently computes gradients for constrained optimization
 * by solving for an adjoint variable λ instead of explicitly inverting matrices.
 * 
 * Mathematical background:
 * - For constraint c(p, x) = 0, the implicit function theorem gives:
 *   df/dp = ∂f/∂p - ∂f/∂x (∂c/∂x)^-1 ∂c/∂p
 * - Instead of computing (∂c/∂x)^-1 ∂c/∂p explicitly, we solve:
 *   (∂c/∂x)^T λ = (∂f/∂x)^T
 *   Then: df/dp = ∂f/∂p - λ^T ∂c/∂p
 * - This requires solving only one linear system per iteration instead of
 *   paramCount systems, making it much more efficient.
 * 
 * For residual functions r(p, x) where f = 1/2 r^T r:
 * - Solve: (∂c/∂x)^T λ = r^T ∂r/∂x
 * - Then: df/dp = r^T ∂r/∂p - λ^T ∂c/∂p
 * 
 * References:
 * - Nocedal & Wright, "Numerical Optimization" (2nd ed.), Chapter 12 (constrained optimization)
 * - Adjoint method is widely used in optimal control and shape optimization
 * 
 * Role in system:
 * - Provides efficient constrained optimization using adjoint method
 * - Supports both cost functions and residual functions
 * - Uses finite differences or analytical derivatives
 * - For residual functions r(p, x), can compute dr/dp (Jacobian matrix) efficiently
 *   by reusing ∂c/∂x decomposition for all residual components. This is more efficient
 *   than BFGS or Lagrange multiplier methods. The Jacobian enables Gauss-Newton or
 *   Levenberg-Marquardt methods for quadratic convergence in constrained optimization
 * 
 * For first-time readers:
 * - Start with adjointGradientDescent function
 * - Understand how adjoint variable λ is computed
 * - Check how states x are updated using linear approximation
 */

import { Matrix } from 'ml-matrix';
// @ts-ignore - SingularValueDecomposition exists in ml-matrix but may not be in type definitions
import { SingularValueDecomposition } from 'ml-matrix';
import type {
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn,
  AdjointGradientDescentOptions,
  AdjointGradientDescentResult
} from './types.js';
import {
  finiteDiffPartialP,
  finiteDiffPartialX,
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX,
  finiteDiffResidualPartialP,
  finiteDiffResidualPartialX
} from './finiteDiff.js';
import { backtrackingLineSearch } from './lineSearch.js';
import { vectorNorm, scaleVector, addVectors, subtractVectors } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence, createConvergenceResult } from './convergence.js';
import { Logger } from './logger.js';
import { float64ArrayToMatrix, matrixToFloat64Array } from '../utils/matrix.js';
import { solveAdjointEquation as solveAdjointEquationShared, solveLeastSquares as solveLeastSquaresShared } from './constrainedUtils.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 0.01;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_CONSTRAINT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE_P = 1e-6;
const DEFAULT_STEP_SIZE_X = 1e-6;
const ZERO_STEP_SIZE = 0.0;
const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const RESIDUAL_COST_COEFFICIENT = 0.5; // Coefficient for residual cost: f = 1/2 r^T r
const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negating vectors
const MAX_DIMENSION_FOR_DETAILED_LOGGING = 3; // Maximum dimension for detailed parameter/state logging
const DEFAULT_REGULARIZATION = 0.0; // Optional Tikhonov regularization for adjoint solve
const MAX_SVD_DIAGNOSTIC_DIMENSION = 50; // Limit for expensive SVD diagnostics in logs
const CONDITION_WARNING_THRESHOLD = 1e10; // Threshold to warn about ill-conditioning
const AUTO_REGULARIZATION = 1e-8; // Floor regularization injected when Jacobian is singular/ill-conditioned
const MAX_REGULARIZATION_RETRY_ATTEMPTS = 20; // Maximum number of retry attempts with increasing regularization
const REGULARIZATION_MULTIPLIER = 10; // Multiplier for increasing regularization on each retry
const FALLBACK_REGULARIZATION = 1e-6; // Fallback regularization value when current regularization is zero
const MAX_MATRIX_DIMENSION_FOR_SVD_DIAGNOSTICS = 300; // Maximum matrix dimension for expensive SVD diagnostics
const FLOATING_POINT_EQUALITY_TOLERANCE = 1e-15; // Tolerance for floating point equality comparisons
const INITIAL_ATTEMPT_NUMBER = 1; // Initial attempt number for logging

/**
 * Checks if a function is a residual function by calling it and checking return type.
 */
function isResidualFunction(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  parameters: Float64Array,
  states: Float64Array
): costFunction is ConstrainedResidualFn {
  const result = costFunction(parameters, states);
  return result instanceof Float64Array;
}

/**
 * Computes cost from either a cost function or residual function.
 * For residual functions r(p,x), computes f = 1/2 r^T r.
 */
function computeCost(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  parameters: Float64Array,
  states: Float64Array
): number {
  if (isResidualFunction(costFunction, parameters, states)) {
    const residual = costFunction(parameters, states);
    const residualNorm = vectorNorm(residual);
    return RESIDUAL_COST_COEFFICIENT * residualNorm * residualNorm;
  }
  
  return costFunction(parameters, states);
}

/**
 * Computes gradient from residual function: df/dp = r^T ∂r/∂p
 * This formula comes from the chain rule applied to the residual cost function f = 1/2 r^T r.
 */
function computeGradientFromResidual(
  residual: Float64Array,
  derivativeMatrix: Matrix
): Float64Array {
  const residualMatrix = float64ArrayToMatrix(residual);
  const gradientMatrix = residualMatrix.transpose().mmul(derivativeMatrix);
  return rowVectorToFloat64Array(gradientMatrix);
}

/**
 * Computes ∂f/∂p or ∂r/∂p using analytical functions or finite differences.
 */
function computeDfdp(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  options: AdjointGradientDescentOptions
): Float64Array {
  const stepSizeP = options.stepSizeP ?? DEFAULT_STEP_SIZE_P;
  
  if (options.dfdp) {
    return options.dfdp(parameters, states);
  }
  
  const isResidual = isResidualFunction(costFunction, parameters, states);
  if (isResidual) {
    const derivativeResidualPartialP = finiteDiffResidualPartialP(parameters, states, costFunction, { stepSize: stepSizeP });
    const residual = costFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialP);
  }
  
  return finiteDiffPartialP(parameters, states, costFunction, { stepSize: stepSizeP });
}

/**
 * Computes ∂f/∂x or ∂r/∂x using analytical functions or finite differences.
 */
function computeDfdx(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  options: AdjointGradientDescentOptions
): Float64Array {
  const stepSizeX = options.stepSizeX ?? DEFAULT_STEP_SIZE_X;
  
  if (options.dfdx) {
    return options.dfdx(parameters, states);
  }
  
  const isResidual = isResidualFunction(costFunction, parameters, states);
  if (isResidual) {
    const derivativeResidualPartialX = finiteDiffResidualPartialX(parameters, states, costFunction, { stepSize: stepSizeX });
    const residual = costFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialX);
  }
  
  return finiteDiffPartialX(parameters, states, costFunction, { stepSize: stepSizeX });
}

/**
 * Computes partial derivatives using analytical functions or finite differences.
 */
function computePartialDerivatives(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions
): {
  dfdp: Float64Array;
  dfdx: Float64Array;
  dcdp: Matrix;
  dcdx: Matrix;
} {
  const stepSizeP = options.stepSizeP ?? DEFAULT_STEP_SIZE_P;
  const stepSizeX = options.stepSizeX ?? DEFAULT_STEP_SIZE_X;
  
  const dfdp = computeDfdp(parameters, states, costFunction, options);
  const dfdx = computeDfdx(parameters, states, costFunction, options);

  // Compute ∂c/∂p: needed for adjoint gradient computation (df/dp = ∂f/∂p - λ^T ∂c/∂p)
  const dcdp = options.dcdp
    ? options.dcdp(parameters, states)
    : finiteDiffConstraintPartialP(parameters, states, constraintFunction, { stepSize: stepSizeP });

  // Compute ∂c/∂x: needed to solve adjoint equation (∂c/∂x)^T λ = (∂f/∂x)^T
  const dcdx = options.dcdx
    ? options.dcdx(parameters, states)
    : finiteDiffConstraintPartialX(parameters, states, constraintFunction, { stepSize: stepSizeX });

  return { dfdp, dfdx, dcdp, dcdx };
}


function computeMatrixDiagnostics(matrix: Matrix): { frobenius: number; maxAbs: number; minAbs: number } {
  let sumSquares = 0;
  let maxAbs = 0;
  let minAbs = Number.POSITIVE_INFINITY;
  for (let r = 0; r < matrix.rows; r++) {
    for (let c = 0; c < matrix.columns; c++) {
      const value = matrix.get(r, c);
      const abs = Math.abs(value);
      sumSquares += value * value;
      maxAbs = Math.max(maxAbs, abs);
      minAbs = Math.min(minAbs, abs);
    }
  }
  return {
    frobenius: Math.sqrt(sumSquares),
    maxAbs,
    minAbs: minAbs === Number.POSITIVE_INFINITY ? 0 : minAbs
  };
}

function rowVectorToFloat64Array(matrix: Matrix): Float64Array {
  if (matrix.rows !== 1) {
    throw new Error('Expected row vector (1 x n)');
  }
  const result = new Float64Array(matrix.columns);
  for (let c = 0; c < matrix.columns; c++) {
    result[c] = matrix.get(0, c);
  }
  return result;
}

function computeVectorDiagnostics(vector: Float64Array): { norm: number; maxAbs: number } {
  let sumSquares = 0;
  let maxAbs = 0;
  for (let i = 0; i < vector.length; i++) {
    const value = vector[i];
    const abs = Math.abs(value);
    sumSquares += value * value;
    maxAbs = Math.max(maxAbs, abs);
  }
  return { norm: Math.sqrt(sumSquares), maxAbs };
}

/**
 * Updates regularization value by multiplying with the multiplier, or sets to fallback if current is zero.
 * This exponential backoff strategy helps stabilize ill-conditioned linear systems.
 */
function updateRegularizationWithMultiplier(currentRegularization: number): number {
  return currentRegularization > 0 ? currentRegularization * REGULARIZATION_MULTIPLIER : FALLBACK_REGULARIZATION;
}

function computeSvdDiagnostics(
  matrix: Matrix
): { sigmaMax: number; sigmaMin: number; condEst: number; rankEst: number } | undefined {
  if (matrix.rows > MAX_SVD_DIAGNOSTIC_DIMENSION || matrix.columns > MAX_SVD_DIAGNOSTIC_DIMENSION) {
    return undefined;
  }
  try {
    const svd = new SingularValueDecomposition(matrix);
    const singularValues = svd.diagonal;
    const sigmaMax = Math.max(...singularValues);
    const sigmaMin = Math.min(...singularValues);
    const condEst = sigmaMin > 0 ? sigmaMax / sigmaMin : Infinity;
    const threshold = sigmaMax * Number.EPSILON * Math.max(matrix.rows, matrix.columns);
    const rankEst = singularValues.filter((s: number) => s > threshold).length;
    return { sigmaMax, sigmaMin, condEst, rankEst };
  } catch {
    return undefined;
  }
}

function logAdjointDiagnostics(
  dcdx: Matrix,
  rightHandSide: Float64Array,
  logger: Logger,
  message: string,
  level: 'debug' | 'warn',
  attempt: number,
  regularization: number,
  error?: unknown,
  svdDiagnostics?: { sigmaMax: number; sigmaMin: number; condEst: number; rankEst: number }
): void {
  const matrixStats = computeMatrixDiagnostics(dcdx);
  const rightHandSideStats = computeVectorDiagnostics(rightHandSide);
  const details: Array<{ key: string; value: number | string }> = [
    { key: 'attempt', value: attempt },
    { key: 'reg', value: regularization },
    { key: 'rows', value: dcdx.rows },
    { key: 'columns', value: dcdx.columns },
    { key: 'frobenius', value: matrixStats.frobenius },
    { key: 'max_abs', value: matrixStats.maxAbs },
    { key: 'min_abs', value: matrixStats.minAbs },
    { key: 'rhs_norm', value: rightHandSideStats.norm },
    { key: 'rhs_max_abs', value: rightHandSideStats.maxAbs }
  ];

  const svd = svdDiagnostics ?? computeSvdDiagnostics(dcdx);
  if (svd) {
    details.push(
      { key: 'sigma_max', value: svd.sigmaMax },
      { key: 'sigma_min', value: svd.sigmaMin },
      { key: 'cond_est', value: svd.condEst },
      { key: 'rank_est', value: svd.rankEst }
    );
  }

   // For tiny systems, include raw entries to quickly spot zero/duplicate rows/cols.
   if (dcdx.rows <= MAX_DIMENSION_FOR_DETAILED_LOGGING && dcdx.columns <= MAX_DIMENSION_FOR_DETAILED_LOGGING) {
     for (let r = 0; r < dcdx.rows; r++) {
       for (let c = 0; c < dcdx.columns; c++) {
         details.push({ key: `A[${r},${c}]`, value: dcdx.get(r, c) });
       }
     }
   }

  if (error !== undefined) {
    details.push({ key: 'error', value: String(error) });
  }

  // Filter out string values for logger (logger only accepts numbers)
  const numericDetails = details.filter(d => typeof d.value === 'number') as Array<{ key: string; value: number }>;
  if (level === 'warn') {
    logger.warn('adjointGradientDescent', undefined, message, numericDetails);
  } else {
    logger.debug('adjointGradientDescent', undefined, message, numericDetails);
  }
}


/**
 * Logs SVD diagnostics for small matrices to help diagnose numerical issues.
 * Only computed for small matrices to avoid performance overhead.
 */
function logSvdDiagnosticsForSmallMatrix(matrix: Matrix, logger: Logger): void {
  if (matrix.rows > MAX_MATRIX_DIMENSION_FOR_SVD_DIAGNOSTICS || matrix.columns > MAX_MATRIX_DIMENSION_FOR_SVD_DIAGNOSTICS) {
    return;
  }
  try {
    const svd = new SingularValueDecomposition(matrix);
    const singularValues = svd.diagonal;
    const maxSingularValue = Math.max(...singularValues);
    const minSingularValue = Math.min(...singularValues);
    logger.debug('adjointGradientDescent', undefined, 'SVD diagnostics', [
      { key: 'sigma_max', value: maxSingularValue },
      { key: 'sigma_min', value: minSingularValue },
      { key: 'cond_est', value: minSingularValue > 0 ? maxSingularValue / minSingularValue : Infinity }
    ]);
  } catch (svdError) {
    // Error message is string, so we log it separately without details
    logger.debug('adjointGradientDescent', undefined, `SVD diagnostics failed: ${String(svdError)}`);
  }
}

// Override shared solver to add retry logic with exponentially increasing regularization.
// This handles ill-conditioned matrices that fail on first attempt but succeed with regularization.
function solveLeastSquares(
  A: Matrix,
  b: Matrix,
  logger: Logger
): Float64Array {
  const baseRegularization = (globalThis as any).__ADJOINT_REGULARIZATION__ ?? DEFAULT_REGULARIZATION;
  let regularization = baseRegularization;
  let lastError: unknown;
  for (let attempt = 0; attempt < MAX_REGULARIZATION_RETRY_ATTEMPTS; attempt++) {
    try {
      return solveLeastSquaresShared(A, b, logger, 'adjointGradientDescent', regularization);
    } catch (error) {
      lastError = error;
      regularization = updateRegularizationWithMultiplier(regularization);
      if (!logger) {
        continue;
      }
      logger.warn('adjointGradientDescent', undefined, 'solveLeastSquares failed, retrying with higher regularization', [
        { key: 'attempt', value: attempt + 1 },
        { key: 'rows', value: A.rows },
        { key: 'columns', value: A.columns },
        { key: 'reg', value: regularization },
        { key: 'error', value: String(error) }
      ]);
      logSvdDiagnosticsForSmallMatrix(A, logger);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

function solveAdjointEquation(
  dcdx: Matrix,
  dfdx: Float64Array,
  logger: Logger
): Float64Array {
  const baseRegularization = (globalThis as any).__ADJOINT_REGULARIZATION__ ?? DEFAULT_REGULARIZATION;
  let regularization = baseRegularization;
  let lastError: unknown;
  const svdDiagnostics = computeSvdDiagnostics(dcdx);
  const warnAboutCondition =
    dcdx.rows === dcdx.columns &&
    svdDiagnostics !== undefined &&
    svdDiagnostics.condEst > CONDITION_WARNING_THRESHOLD;

  // If Jacobian is numerically singular, seed a tiny regularization up front.
  if (svdDiagnostics && (!isFinite(svdDiagnostics.condEst) || svdDiagnostics.rankEst === 0 || svdDiagnostics.sigmaMin === 0)) {
    regularization = Math.max(regularization, AUTO_REGULARIZATION);
  }

  logAdjointDiagnostics(
    dcdx,
    dfdx,
    logger,
    warnAboutCondition
      ? 'Adjoint Jacobian appears ill-conditioned before solve'
      : 'Adjoint Jacobian diagnostics before solve',
    warnAboutCondition ? 'warn' : 'debug',
    INITIAL_ATTEMPT_NUMBER,
    regularization,
    undefined,
    svdDiagnostics
  );

  for (let attempt = 0; attempt < MAX_REGULARIZATION_RETRY_ATTEMPTS; attempt++) {
    try {
      return solveAdjointEquationShared(dcdx, dfdx, logger, 'adjointGradientDescent', regularization);
    } catch (error) {
      lastError = error;
      regularization = updateRegularizationWithMultiplier(regularization);
      logAdjointDiagnostics(
        dcdx,
        dfdx,
        logger,
        'solveAdjointEquation failed, retrying with higher regularization',
        'warn',
        attempt + 1,
        regularization,
        error,
        svdDiagnostics
      );
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/**
 * Computes the adjoint gradient: df/dp = ∂f/∂p - λ^T ∂c/∂p
 */
function computeAdjointGradient(
  dfdp: Float64Array,
  lambda: Float64Array,
  dcdp: Matrix
): Float64Array {
  // Compute λ^T ∂c/∂p: this term represents how constraint violations affect the gradient.
  // Matrix dimensions: λ is constraintCount × 1, λ^T is 1 × constraintCount,
  // ∂c/∂p is constraintCount × parameterCount, so λ^T ∂c/∂p is 1 × parameterCount (row vector).
  const lambdaMatrix = float64ArrayToMatrix(lambda);
  const lambdaTranspose = lambdaMatrix.transpose();
  const lambdaTdcdp = lambdaTranspose.mmul(dcdp);
  
  // Convert row vector to Float64Array for vector subtraction.
  // The matrix multiplication produces a 1 × parameterCount matrix, so we extract the first (and only) row.
  const parameterCount = dfdp.length;
  const lambdaTdcdpVector = new Float64Array(parameterCount);
  for (let i = 0; i < parameterCount; i++) {
    lambdaTdcdpVector[i] = lambdaTdcdp.get(0, i);
  }

  // df/dp = ∂f/∂p - λ^T ∂c/∂p
  return subtractVectors(dfdp, lambdaTdcdpVector);
}

/**
 * Updates states using linear approximation: x_new = x_old + dx
 * where dx solves (∂c/∂x) dx = -∂c/∂p · Δp
 * Supports both square and non-square constraint Jacobians.
 */
function updateStates(
  currentStates: Float64Array,
  dcdx: Matrix,
  dcdp: Matrix,
  deltaP: Float64Array,
  logger: Logger
): Float64Array {
  // Compute ∂c/∂p · Δp: this represents how parameter changes affect constraint values.
  const deltaPMatrix = float64ArrayToMatrix(deltaP);
  const dcdpDeltaP = dcdp.mmul(deltaPMatrix);
  const dcdpDeltaPVector = matrixToFloat64Array(dcdpDeltaP);

  // Solve: (∂c/∂x) dx = -∂c/∂p · Δp to find state update that maintains constraint satisfaction.
  // The negative sign ensures states adjust to compensate for constraint changes from parameter updates.
  const negativeDcdpDeltaP = scaleVector(dcdpDeltaPVector, NEGATIVE_COEFFICIENT);
  const negativeDcdpDeltaPMatrix = float64ArrayToMatrix(negativeDcdpDeltaP);
  
  // Use hierarchical solver that handles both square and non-square matrices
  const dx = solveLeastSquares(dcdx, negativeDcdpDeltaPMatrix, logger);

  // Update states using linear approximation to maintain constraint satisfaction.
  // This is more efficient than solving the full nonlinear constraint system each iteration.
  return addVectors(currentStates, dx);
}

/**
 * Creates a cost function wrapper for line search that updates states using linear approximation.
 * Partial derivatives are pre-computed and cached to avoid recomputation during line search.
 */
function createCostFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions,
  logger: Logger,
  cachedPartials?: { dcdx: Matrix; dcdp: Matrix }
): (params: Float64Array) => number {
  // Pre-compute partial derivatives once if not provided
  const partials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options
  );
  const { dcdx, dcdp } = partials;

  return (params: Float64Array): number => {
    // Update states during line search to maintain constraint satisfaction.
    // We use linear approximation (x_new = x_old + dx) where dx solves (∂c/∂x) dx = -∂c/∂p · Δp
    // because solving the full nonlinear constraint system for each line search step would be too expensive.
    const deltaP = subtractVectors(params, currentParameters);
    const newStates = updateStates(currentStates, dcdx, dcdp, deltaP, logger);
    return computeCost(costFunction, params, newStates);
  };
}

/**
 * Checks if two parameter arrays are equal within floating point tolerance.
 * Used to avoid redundant gradient computation when line search evaluates at the starting point.
 */
function areParametersEqual(
  parameters1: Float64Array,
  parameters2: Float64Array
): boolean {
  if (parameters1.length !== parameters2.length) {
    return false;
  }
  for (let i = 0; i < parameters1.length; i++) {
    if (Math.abs(parameters1[i] - parameters2[i]) > FLOATING_POINT_EQUALITY_TOLERANCE) {
      return false;
    }
  }
  return true;
}

/**
 * Creates a gradient function wrapper for line search.
 * For each trial parameter, updates states and computes gradient at that point.
 * Uses pre-computed currentGradient for current point to ensure consistency.
 */
function createGradientFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentGradient: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions,
  logger: Logger,
  cachedPartials?: { dfdp: Float64Array; dfdx: Float64Array; dcdp: Matrix; dcdx: Matrix }
): (_params: Float64Array) => Float64Array {
  // Pre-compute partial derivatives once for state updates
  const currentPartials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options
  );
  const { dcdx: currentDcdx, dcdp: currentDcdp } = currentPartials;

  return (trialParams: Float64Array): Float64Array => {
    // Return pre-computed gradient if parameters haven't changed to avoid redundant computation.
    // Line search evaluates gradient at the starting point for direction derivative calculation.
    if (areParametersEqual(trialParams, currentParameters)) {
      return new Float64Array(currentGradient);
    }
    
    // For different trial parameters, update states to maintain constraints and compute gradient.
    // We use linear approximation for efficiency: solving full nonlinear constraints for each trial would be too slow.
    const deltaP = subtractVectors(trialParams, currentParameters);
    const trialStates = updateStates(currentStates, currentDcdx, currentDcdp, deltaP, logger);
    
    // Compute gradient at trial point to evaluate search direction quality in line search.
    const trialPartials = computePartialDerivatives(
      trialParams,
      trialStates,
      costFunction,
      constraintFunction,
      options
    );
    const lambda = solveAdjointEquation(trialPartials.dcdx, trialPartials.dfdx, logger);
    return computeAdjointGradient(trialPartials.dfdp, lambda, trialPartials.dcdp);
  };
}

/**
 * Determines the step size for gradient descent iteration.
 */
function determineStepSize(
  currentGradient: Float64Array,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  options: AdjointGradientDescentOptions,
  logger: Logger,
  cachedPartials?: { dfdp: Float64Array; dfdx: Float64Array; dcdp: Matrix; dcdx: Matrix }
): { stepSize: number; usedLineSearch: boolean } {
  if (!useLineSearch || fixedStepSize !== undefined) {
    return { stepSize: fixedStepSize ?? DEFAULT_STEP_SIZE, usedLineSearch: false };
  }

  // Pre-compute partial derivatives once and reuse in both wrappers
  const partials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options
  );

  const costFnWrapper = createCostFunctionWrapper(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger,
    { dcdx: partials.dcdx, dcdp: partials.dcdp }
  );
  const gradientFnWrapper = createGradientFunctionWrapper(
    currentParameters,
    currentStates,
    currentGradient,
    costFunction,
    constraintFunction,
    options,
    logger,
    partials
  );

  const searchDirection = scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
  const stepSize = backtrackingLineSearch(
    costFnWrapper,
    gradientFnWrapper,
    currentParameters,
    searchDirection
  );

  return { stepSize, usedLineSearch: true };
}

/**
 * Checks constraint violation and logs warning if needed.
 */
function checkConstraintViolation(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  iteration: number,
  logger: Logger
): { constraint: Float64Array; constraintNorm: number } {
  const constraint = constraintFunction(currentParameters, currentStates);
  const constraintNorm = vectorNorm(constraint);
  if (constraintNorm > constraintTolerance) {
    logger.warn('adjointGradientDescent', iteration, 'Constraint violation detected', [
      { key: '||c(p,x)||:', value: constraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }
  return { constraint, constraintNorm };
}

/**
 * Computes the adjoint gradient and its norm.
 * Returns both the gradient and partial derivatives for reuse.
 */
function computeAdjointGradientAndNorm(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions,
  logger: Logger
): {
  adjointGradient: Float64Array;
  gradientNorm: number;
  partials: { dfdp: Float64Array; dfdx: Float64Array; dcdp: Matrix; dcdx: Matrix };
} {
  const partials = computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options
  );
  const lambda = solveAdjointEquation(partials.dcdx, partials.dfdx, logger);
  const adjointGradient = computeAdjointGradient(partials.dfdp, lambda, partials.dcdp);
  const gradientNorm = vectorNorm(adjointGradient);
  return { adjointGradient, gradientNorm, partials };
}

/**
 * Checks gradient convergence and returns result if converged.
 */
function checkGradientConvergenceAndReturn(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  constraintNorm: number,
  constraintTolerance: number,
  tolerance: number,
  usedLineSearchFlag: boolean,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  if (constraintNorm <= constraintTolerance && checkGradientConvergence(gradientNorm, tolerance, iteration)) {
    logger.info('adjointGradientDescent', iteration, 'Converged', [
      { key: 'Cost:', value: currentCost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Constraint norm:', value: constraintNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return {
      converged: true,
      result: {
        ...result,
        usedLineSearch: usedLineSearchFlag,
        finalStates: currentStates,
        finalConstraintNorm: constraintNorm
      }
    };
  }
  return { converged: false };
}

/**
 * Handles line search failure case.
 */
function handleLineSearchFailure(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  constraintNorm: number,
  logger: Logger
): { converged: boolean; result: AdjointGradientDescentResult } {
  logger.warn('adjointGradientDescent', iteration, 'Line search failed', [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm }
  ]);
  return {
    converged: true,
    result: {
      finalParameters: currentParameters,
      parameters: currentParameters,
      iterations: iteration,
      converged: false,
      finalCost: currentCost,
      finalGradientNorm: gradientNorm,
      usedLineSearch: true,
      finalStates: currentStates,
      finalConstraintNorm: constraintNorm
    }
  };
}

/**
 * Updates parameters and states, then computes new cost.
 */
function updateParametersAndStates(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  adjointGradient: Float64Array,
  stepSize: number,
  partials: { dcdx: Matrix; dcdp: Matrix },
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  logger: Logger
): { newParameters: Float64Array; newStates: Float64Array; newCost: number } {
  const negativeStepSize = NEGATIVE_GRADIENT_DIRECTION * stepSize;
  const step = scaleVector(adjointGradient, negativeStepSize);
  const newParameters = addVectors(currentParameters, step);
  const deltaP = subtractVectors(newParameters, currentParameters);
  const newStates = updateStates(currentStates, partials.dcdx, partials.dcdp, deltaP, logger);
  const newCost = computeCost(costFunction, newParameters, newStates);

  return { newParameters, newStates, newCost };
}

/**
 * Checks step size convergence and returns result if converged.
 */
function checkStepSizeConvergenceAndReturn(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  stepNorm: number,
  constraintNorm: number,
  constraintTolerance: number,
  tolerance: number,
  newUsedLineSearch: boolean,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  if (constraintNorm <= constraintTolerance && checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
    logger.info('adjointGradientDescent', iteration, 'Converged', [
      { key: 'Cost:', value: currentCost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Step size:', value: stepNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return {
      converged: true,
      result: {
        ...result,
        usedLineSearch: newUsedLineSearch,
        finalStates: currentStates,
        finalConstraintNorm: constraintNorm
      }
    };
  }
  return { converged: false };
}

/**
 * Creates detailed log information for progress logging.
 */
/**
 * Adds array elements to log details with a prefix for readability.
 * This helps create consistent logging format across different array types.
 */
function addArrayToLogDetails(
  details: Array<{ key: string; value: number }>,
  array: Float64Array,
  prefix: string
): void {
  for (let i = 0; i < array.length; i++) {
    details.push({ key: `${prefix}[${i}]:`, value: array[i] });
  }
}

function createProgressLogDetails(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraint: Float64Array,
  currentCost: number,
  gradientNorm: number,
  stepSize: number,
  constraintNorm: number
): Array<{ key: string; value: number }> {
  const logDetails: Array<{ key: string; value: number }> = [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm },
    { key: 'Step size:', value: stepSize },
    { key: 'Constraint norm:', value: constraintNorm }
  ];

  // Add parameter and state information for small dimensions (for readability)
  if (currentParameters.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING && currentStates.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING) {
    addArrayToLogDetails(logDetails, currentParameters, 'p');
    addArrayToLogDetails(logDetails, currentStates, 'x');
    // Add constraint values for small dimensions
    if (constraint.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING) {
      addArrayToLogDetails(logDetails, constraint, 'c');
    }
  }

  return logDetails;
}

/**
 * Handles callback and checks gradient convergence.
 */
function checkConvergenceAndHandleCallback(
  iteration: number,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentCost: number,
  gradientNorm: number,
  constraintNorm: number,
  constraintTolerance: number,
  tolerance: number,
  usedLineSearchFlag: boolean,
  onIteration: ((iteration: number, cost: number, parameters: Float64Array) => void) | undefined,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  // Handle callback
  if (onIteration) {
    const callbackIteration = iteration;
    onIteration(callbackIteration, currentCost, currentParameters);
  }

  // Check gradient convergence
  const gradientConvergenceResult = checkGradientConvergenceAndReturn(
    currentParameters,
    currentStates,
    iteration,
    currentCost,
    gradientNorm,
    constraintNorm,
    constraintTolerance,
    tolerance,
    usedLineSearchFlag,
    logger
  );
  if (gradientConvergenceResult.converged && gradientConvergenceResult.result) {
    return gradientConvergenceResult;
  }
  return { converged: false };
}

/**
 * Handles step size determination, parameter update, and step size convergence check.
 */
function handleStepSizeAndUpdate(
  adjointGradient: Float64Array,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraint: Float64Array,
  currentCost: number,
  gradientNorm: number,
  constraintNorm: number,
  iteration: number,
  constraintTolerance: number,
  tolerance: number,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  usedLineSearchFlag: boolean,
  partials: { dfdp: Float64Array; dfdx: Float64Array; dcdx: Matrix; dcdp: Matrix },
  options: AdjointGradientDescentOptions,
  logger: Logger
): {
  converged: boolean;
  result?: AdjointGradientDescentResult;
  newParameters?: Float64Array;
  newStates?: Float64Array;
  newCost?: number;
  newUsedLineSearch?: boolean;
} {
  // Determine step size (reuse pre-computed partials to avoid recomputation in line search)
  const stepSizeResult = determineStepSize(
    adjointGradient,
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    useLineSearch,
    fixedStepSize,
    options,
    logger,
    partials
  );

  if (stepSizeResult.stepSize === ZERO_STEP_SIZE) {
    return handleLineSearchFailure(
      currentParameters,
      currentStates,
      iteration,
      currentCost,
      gradientNorm,
      constraintNorm,
      logger
    );
  }

  const newUsedLineSearch = usedLineSearchFlag || stepSizeResult.usedLineSearch;

  // Update parameters and states, then compute new cost to evaluate the step quality.
  // States are updated to maintain constraint satisfaction after parameter changes.
  const { newParameters, newStates, newCost } = updateParametersAndStates(
    currentParameters,
    currentStates,
    adjointGradient,
    stepSizeResult.stepSize,
    partials,
    costFunction,
    logger
  );

  // Check step size convergence to detect when optimization has stalled.
  // Log progress to help diagnose convergence issues.
  if (!newParameters) {
    return {
      converged: false,
      newParameters,
      newStates,
      newCost,
      newUsedLineSearch
    };
  }
  const stepSizeConvergenceResult = checkStepSizeConvergenceAndLog(
    currentParameters,
    currentStates,
    constraint,
    currentCost,
    gradientNorm,
    stepSizeResult.stepSize,
    constraintNorm,
    iteration,
    constraintTolerance,
    tolerance,
    newUsedLineSearch,
    newParameters,
    logger
  );
  if (stepSizeConvergenceResult.converged && stepSizeConvergenceResult.result) {
    return stepSizeConvergenceResult;
  }

  return {
    converged: false,
    newParameters,
    newStates,
    newCost,
    newUsedLineSearch
  };
}

/**
 * Checks step size convergence and logs progress.
 */
function checkStepSizeConvergenceAndLog(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraint: Float64Array,
  currentCost: number,
  gradientNorm: number,
  stepSize: number,
  constraintNorm: number,
  iteration: number,
  constraintTolerance: number,
  tolerance: number,
  newUsedLineSearch: boolean,
  newParameters: Float64Array,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  // Check step size convergence: if step is too small, optimization has likely converged or stalled.
  const step = subtractVectors(newParameters, currentParameters);
  const stepNorm = vectorNorm(step);
  const stepSizeConvergenceResult = checkStepSizeConvergenceAndReturn(
    currentParameters,
    currentStates,
    iteration,
    currentCost,
    gradientNorm,
    stepNorm,
    constraintNorm,
    constraintTolerance,
    tolerance,
    newUsedLineSearch,
    logger
  );
  if (stepSizeConvergenceResult.converged && stepSizeConvergenceResult.result) {
    return stepSizeConvergenceResult;
  }

  // Log progress with detailed information to help diagnose optimization behavior and convergence issues.
  const logDetails = createProgressLogDetails(
    currentParameters,
    currentStates,
    constraint,
    currentCost,
    gradientNorm,
    stepSize,
    constraintNorm
  );
  logger.debug('adjointGradientDescent', iteration, 'Progress', logDetails);

  return { converged: false };
}

/**
 * Performs a single adjoint gradient descent iteration.
 */
function performAdjointGradientDescentIteration(
  iteration: number,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentCost: number,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  tolerance: number,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  constraintTolerance: number,
  onIteration: ((iteration: number, cost: number, parameters: Float64Array) => void) | undefined,
  logger: Logger,
  usedLineSearchFlag: boolean,
  options: AdjointGradientDescentOptions
): {
  converged: boolean;
  result?: AdjointGradientDescentResult;
  newParameters?: Float64Array;
  newStates?: Float64Array;
  newCost?: number;
  newUsedLineSearch?: boolean;
} {
  // Check constraint satisfaction each iteration to detect and warn about constraint violations.
  // This helps identify when the linear approximation for state updates is breaking down.
  const { constraint, constraintNorm } = checkConstraintViolation(
    currentParameters,
    currentStates,
    constraintFunction,
    constraintTolerance,
    iteration,
    logger
  );

  // Compute adjoint gradient and norm: the gradient drives parameter updates, norm indicates convergence.
  const { adjointGradient, gradientNorm, partials } = computeAdjointGradientAndNorm(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger
  );

  // Handle callback to allow user monitoring, then check gradient convergence to detect optimality.
  const convergenceResult = checkConvergenceAndHandleCallback(
    iteration,
    currentParameters,
    currentStates,
    currentCost,
    gradientNorm,
    constraintNorm,
    constraintTolerance,
    tolerance,
    usedLineSearchFlag,
    onIteration,
    logger
  );
  if (convergenceResult.converged && convergenceResult.result) {
    return convergenceResult;
  }

  // Handle step size determination and parameter update: this is the core optimization step.
  const updateResult = handleStepSizeAndUpdate(
    adjointGradient,
    currentParameters,
    currentStates,
    constraint,
    currentCost,
    gradientNorm,
    constraintNorm,
    iteration,
    constraintTolerance,
    tolerance,
    costFunction,
    constraintFunction,
    useLineSearch,
    fixedStepSize,
    usedLineSearchFlag,
    partials,
    options,
    logger
  );
  if (updateResult.converged && updateResult.result) {
    return updateResult;
  }

  return updateResult;
}

/**
 * Validates initial conditions including constraint satisfaction and dimensions.
 */
function validateInitialConditions(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  logger: Logger
): void {
  const initialConstraint = constraintFunction(initialParameters, initialStates);
  const initialConstraintNorm = vectorNorm(initialConstraint);
  if (initialConstraintNorm > constraintTolerance) {
    logger.warn('adjointGradientDescent', undefined, 'Initial constraint violation', [
      { key: '||c(p0,x0)||:', value: initialConstraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }

  // Note: Constraint count and state count no longer need to match.
  // The adjoint method now supports non-square constraint Jacobians.
}

/**
 * Computes initial cost from initial parameters and states.
 */
function computeInitialCost(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  initialParameters: Float64Array,
  initialStates: Float64Array
): number {
  return computeCost(costFunction, initialParameters, initialStates);
}

/**
 * Creates result for maximum iterations reached case.
 */
function createMaxIterationsResult(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentCost: number,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  maxIterations: number,
  usedLineSearchFlag: boolean,
  options: AdjointGradientDescentOptions,
  logger: Logger
): AdjointGradientDescentResult {
  const partials = computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options
  );
  const lambda = solveAdjointEquation(partials.dcdx, partials.dfdx, logger);
  const finalGradient = computeAdjointGradient(partials.dfdp, lambda, partials.dcdp);
  const finalGradientNorm = vectorNorm(finalGradient);
  const finalConstraint = constraintFunction(currentParameters, currentStates);
  const finalConstraintNorm = vectorNorm(finalConstraint);

  logger.warn('adjointGradientDescent', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: currentCost },
    { key: 'Final gradient norm:', value: finalGradientNorm },
    { key: 'Final constraint norm:', value: finalConstraintNorm }
  ]);

  return {
    finalParameters: currentParameters,
    parameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: currentCost,
    finalGradientNorm: finalGradientNorm,
    usedLineSearch: usedLineSearchFlag,
    finalStates: currentStates,
    finalConstraintNorm: finalConstraintNorm
  };
}

/**
 * Performs adjoint gradient descent optimization to minimize a constrained cost function.
 * 
 * Algorithm:
 * 1. Start with initial parameters p0 and states x0 (satisfying c(p0, x0) = 0)
 * 2. Compute partial derivatives ∂f/∂p, ∂f/∂x, ∂c/∂p, ∂c/∂x
 * 3. Solve adjoint equation: (∂c/∂x)^T λ = (∂f/∂x)^T
 * 4. Compute gradient: df/dp = ∂f/∂p - λ^T ∂c/∂p
 * 5. Update parameters: p_new = p_old - stepSize * df/dp
 * 6. Update states: x_new = x_old - (∂c/∂x)^-1 ∂c/∂p · Δp (linear approximation)
 * 7. Repeat until convergence or max iterations
 * 
 * Supports both cost functions f(p,x) and residual functions r(p,x) where f = 1/2 r^T r.
 * 
 * @param initialParameters - Initial parameter vector p0
 * @param initialStates - Initial state vector x0 (should satisfy c(p0, x0) = 0)
 * @param costFunction - Cost function f(p, x) or residual function r(p, x)
 * @param constraintFunction - Constraint function c(p, x) = 0
 * @param options - Optimization options
 * @returns Optimization result with final parameters, states, and constraint norm
 */
export function adjointGradientDescent(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions = {}
): AdjointGradientDescentResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const stepSize = options.stepSize;
  const useLineSearch = options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH;
  const constraintTolerance = options.constraintTolerance ?? DEFAULT_CONSTRAINT_TOLERANCE;
  const onIteration = options.onIteration;
  const logger = new Logger(options.logLevel, options.verbose);

  validateInitialConditions(initialParameters, initialStates, constraintFunction, constraintTolerance, logger);

  let currentParameters = new Float64Array(initialParameters);
  let currentStates = new Float64Array(initialStates);
  let currentCost = computeInitialCost(costFunction, currentParameters, currentStates);
  let usedLineSearchFlag = false;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const iterationResult = performAdjointGradientDescentIteration(
      iteration,
      currentParameters,
      currentStates,
      currentCost,
      costFunction,
      constraintFunction,
      tolerance,
      useLineSearch,
      stepSize,
      constraintTolerance,
      onIteration,
      logger,
      usedLineSearchFlag,
      options
    );

    if (iterationResult.converged && iterationResult.result) {
      return iterationResult.result;
    }

    if (!iterationResult.newParameters || !iterationResult.newStates || iterationResult.newCost === undefined) {
      continue;
    }

    currentParameters = new Float64Array(iterationResult.newParameters);
    currentStates = new Float64Array(iterationResult.newStates);
    currentCost = iterationResult.newCost;
    if (iterationResult.newUsedLineSearch !== undefined) {
      usedLineSearchFlag = iterationResult.newUsedLineSearch;
    }
  }

  return createMaxIterationsResult(
    currentParameters,
    currentStates,
    currentCost,
    costFunction,
    constraintFunction,
    maxIterations,
    usedLineSearchFlag,
    options,
    logger
  );
}


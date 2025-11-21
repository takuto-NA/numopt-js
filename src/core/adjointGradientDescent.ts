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

import { Matrix, solve, CholeskyDecomposition, QR, pseudoInverse } from 'ml-matrix';
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
    const drdp = finiteDiffResidualPartialP(parameters, states, costFunction, { stepSize: stepSizeP });
    const residual = costFunction(parameters, states);
    // df/dp = r^T ∂r/∂p
    const residualMatrix = float64ArrayToMatrix(residual);
    const dfdpMatrix = residualMatrix.transpose().mmul(drdp);
    return matrixToFloat64Array(dfdpMatrix);
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
    const drdx = finiteDiffResidualPartialX(parameters, states, costFunction, { stepSize: stepSizeX });
    const residual = costFunction(parameters, states);
    // df/dx = r^T ∂r/∂x
    const residualMatrix = float64ArrayToMatrix(residual);
    const dfdxMatrix = residualMatrix.transpose().mmul(drdx);
    return matrixToFloat64Array(dfdxMatrix);
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

  // Compute ∂c/∂p
  const dcdp = options.dcdp
    ? options.dcdp(parameters, states)
    : finiteDiffConstraintPartialP(parameters, states, constraintFunction, { stepSize: stepSizeP });

  // Compute ∂c/∂x
  const dcdx = options.dcdx
    ? options.dcdx(parameters, states)
    : finiteDiffConstraintPartialX(parameters, states, constraintFunction, { stepSize: stepSizeX });

  return { dfdp, dfdx, dcdp, dcdx };
}

/**
 * Solves a least squares problem Ax = b using hierarchical approach.
 * For square matrices, uses existing Cholesky/LU decomposition (backward compatibility).
 * For non-square matrices:
 * - Overdetermined (rows > columns): QR decomposition → normal equations → pseudoInverse
 * - Underdetermined (rows < columns): pseudoInverse directly (QR fails with rank deficient error)
 * 
 * @param A - Coefficient matrix
 * @param b - Right-hand side vector (as Matrix column vector)
 * @param logger - Logger for error messages
 * @returns Solution vector x as Float64Array
 */
function solveLeastSquares(
  A: Matrix,
  b: Matrix,
  logger: Logger
): Float64Array {
  // Square matrix: use existing fast methods (backward compatibility)
  if (A.rows === A.columns) {
    try {
      // Try Cholesky decomposition first for efficiency
      const cholesky = new CholeskyDecomposition(A);
      if (cholesky.isPositiveDefinite()) {
        return matrixToFloat64Array(cholesky.solve(b));
      } else {
        // Fallback to LU decomposition
        return matrixToFloat64Array(solve(A, b));
      }
    } catch (error) {
      // Fallback to LU decomposition if Cholesky fails
      try {
        return matrixToFloat64Array(solve(A, b));
      } catch (solveError) {
        logger.warn('adjointGradientDescent', undefined, `Failed to solve square system: ${solveError}`);
        throw new Error(
          `Failed to solve square system Ax = b. ` +
          `The matrix A may be singular or ill-conditioned. ` +
          `Original error: ${solveError}`
        );
      }
    }
  }

  // Non-square matrix: hierarchical approach
  const isOverdetermined = A.rows > A.columns;
  
  if (isOverdetermined) {
    // Overdetermined system (rows > columns)
    // Strategy: QR decomposition (most stable) → normal equations → pseudoInverse
    
    // Try QR decomposition first (most numerically stable)
    try {
      const qr = new QR(A);
      return matrixToFloat64Array(qr.solve(b));
    } catch (qrError) {
      // QR failed, try normal equations
      try {
        const AT = A.transpose();
        const ATA = AT.mmul(A);
        const ATb = AT.mmul(b);
        return matrixToFloat64Array(solve(ATA, ATb));
      } catch (normalError) {
        // Normal equations failed, use pseudoInverse as last resort
        try {
          const pinv = pseudoInverse(A);
          return matrixToFloat64Array(pinv.mmul(b));
        } catch (pinvError) {
          logger.warn('adjointGradientDescent', undefined, `All methods failed for overdetermined system: QR=${qrError}, Normal=${normalError}, PseudoInv=${pinvError}`);
          throw new Error(
            `Failed to solve overdetermined system Ax = b. ` +
            `All methods (QR, normal equations, pseudoInverse) failed. ` +
            `The matrix A may be rank deficient or ill-conditioned.`
          );
        }
      }
    }
  } else {
    // Underdetermined system (rows < columns)
    // Strategy: pseudoInverse directly (QR fails with rank deficient error)
    try {
      const pinv = pseudoInverse(A);
      return matrixToFloat64Array(pinv.mmul(b));
    } catch (pinvError) {
      logger.warn('adjointGradientDescent', undefined, `Failed to solve underdetermined system with pseudoInverse: ${pinvError}`);
      throw new Error(
        `Failed to solve underdetermined system Ax = b. ` +
        `PseudoInverse computation failed. ` +
        `The matrix A may be ill-conditioned. ` +
        `Original error: ${pinvError}`
      );
    }
  }
}

/**
 * Solves the adjoint equation: (∂c/∂x)^T λ = (∂f/∂x)^T
 * Returns the adjoint variable λ.
 * Supports both square and non-square constraint Jacobians.
 */
function solveAdjointEquation(
  dcdx: Matrix,
  dfdx: Float64Array,
  logger: Logger
): Float64Array {
  // Transpose dcdx: (∂c/∂x)^T
  const dcdxTranspose = dcdx.transpose();

  // Right-hand side: (∂f/∂x)^T (as column vector)
  const dfdxMatrix = float64ArrayToMatrix(dfdx);

  // Solve: (∂c/∂x)^T λ = (∂f/∂x)^T
  // Uses hierarchical solver that handles both square and non-square matrices
  try {
    return solveLeastSquares(dcdxTranspose, dfdxMatrix, logger);
  } catch (error) {
    logger.warn('adjointGradientDescent', undefined, `Failed to solve adjoint equation: ${error}`);
    throw new Error(
      `Failed to solve adjoint equation (∂c/∂x)^T λ = (∂f/∂x)^T. ` +
      `The constraint Jacobian ∂c/∂x may be singular or ill-conditioned. ` +
      `Matrix size: ${dcdx.rows} × ${dcdx.columns}. ` +
      `Original error: ${error}`
    );
  }
}

/**
 * Computes the adjoint gradient: df/dp = ∂f/∂p - λ^T ∂c/∂p
 */
function computeAdjointGradient(
  dfdp: Float64Array,
  lambda: Float64Array,
  dcdp: Matrix
): Float64Array {
  // λ^T ∂c/∂p
  // λ is constraintCount × 1, λ^T is 1 × constraintCount
  // ∂c/∂p is constraintCount × parameterCount
  // λ^T ∂c/∂p is 1 × parameterCount (row vector)
  const lambdaMatrix = float64ArrayToMatrix(lambda);
  const lambdaTranspose = lambdaMatrix.transpose();
  const lambdaTdcdp = lambdaTranspose.mmul(dcdp);
  
  // Convert row vector (1 × parameterCount) to Float64Array
  // lambdaTdcdp is 1 × parameterCount, so we read the first (and only) row
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
  // Compute ∂c/∂p · Δp
  const deltaPMatrix = float64ArrayToMatrix(deltaP);
  const dcdpDeltaP = dcdp.mmul(deltaPMatrix);
  const dcdpDeltaPVector = matrixToFloat64Array(dcdpDeltaP);

  // Solve: (∂c/∂x) dx = -∂c/∂p · Δp
  const negativeDcdpDeltaP = scaleVector(dcdpDeltaPVector, NEGATIVE_COEFFICIENT);
  const negativeDcdpDeltaPMatrix = float64ArrayToMatrix(negativeDcdpDeltaP);
  
  // Use hierarchical solver that handles both square and non-square matrices
  const dx = solveLeastSquares(dcdx, negativeDcdpDeltaPMatrix, logger);

  // x_new = x_old + dx
  return addVectors(currentStates, dx);
}

/**
 * Creates a cost function wrapper for line search that updates states using linear approximation.
 */
function createCostFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions,
  logger: Logger
): (params: Float64Array) => number {
  return (params: Float64Array): number => {
    // For line search, we need to update states as well
    // Use linear approximation: x_new = x_old + dx where dx solves (∂c/∂x) dx = -∂c/∂p · Δp
    const deltaP = subtractVectors(params, currentParameters);
    
    // Compute partial derivatives for state update
    const { dcdx, dcdp } = computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      options
    );

    const newStates = updateStates(currentStates, dcdx, dcdp, deltaP, logger);
    return computeCost(costFunction, params, newStates);
  };
}

/**
 * Creates a gradient function wrapper for line search.
 */
function createGradientFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions,
  logger: Logger
): (_params: Float64Array) => Float64Array {
  return (_params: Float64Array): Float64Array => {
    const partials = computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      options
    );

    const lambda = solveAdjointEquation(partials.dcdx, partials.dfdx, logger);
    return computeAdjointGradient(partials.dfdp, lambda, partials.dcdp);
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
  logger: Logger
): { stepSize: number; usedLineSearch: boolean } {
  if (!useLineSearch || fixedStepSize !== undefined) {
    return { stepSize: fixedStepSize ?? DEFAULT_STEP_SIZE, usedLineSearch: false };
  }

  const costFnWrapper = createCostFunctionWrapper(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger
  );
  const gradientFnWrapper = createGradientFunctionWrapper(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger
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
  tolerance: number,
  usedLineSearchFlag: boolean,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  if (checkGradientConvergence(gradientNorm, tolerance, iteration)) {
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
  tolerance: number,
  newUsedLineSearch: boolean,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  if (checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
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
    for (let i = 0; i < currentParameters.length; i++) {
      logDetails.push({ key: `p[${i}]:`, value: currentParameters[i] });
    }
    for (let i = 0; i < currentStates.length; i++) {
      logDetails.push({ key: `x[${i}]:`, value: currentStates[i] });
    }
    // Add constraint values for small dimensions
    if (constraint.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING) {
      for (let i = 0; i < constraint.length; i++) {
        logDetails.push({ key: `c[${i}]:`, value: constraint[i] });
      }
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
  tolerance: number,
  usedLineSearchFlag: boolean,
  onIteration: ((iteration: number, cost: number, parameters: Float64Array) => void) | undefined,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  // Handle callback
  if (onIteration) {
    const callbackIteration = iteration === 0 ? 0 : iteration;
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
  tolerance: number,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  usedLineSearchFlag: boolean,
  partials: { dcdx: Matrix; dcdp: Matrix },
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
  // Determine step size
  const stepSizeResult = determineStepSize(
    adjointGradient,
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    useLineSearch,
    fixedStepSize,
    options,
    logger
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

  // Update parameters and states, compute new cost
  const { newParameters, newStates, newCost } = updateParametersAndStates(
    currentParameters,
    currentStates,
    adjointGradient,
    stepSizeResult.stepSize,
    partials,
    costFunction,
    logger
  );

  // Check step size convergence and log progress
  if (newParameters) {
    const stepSizeConvergenceResult = checkStepSizeConvergenceAndLog(
      currentParameters,
      currentStates,
      constraint,
      currentCost,
      gradientNorm,
      stepSizeResult.stepSize,
      constraintNorm,
      iteration,
      tolerance,
      newUsedLineSearch,
      newParameters,
      logger
    );
    if (stepSizeConvergenceResult.converged && stepSizeConvergenceResult.result) {
      return stepSizeConvergenceResult;
    }
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
  tolerance: number,
  newUsedLineSearch: boolean,
  newParameters: Float64Array,
  logger: Logger
): { converged: boolean; result?: AdjointGradientDescentResult } {
  // Check step size convergence
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
    tolerance,
    newUsedLineSearch,
    logger
  );
  if (stepSizeConvergenceResult.converged && stepSizeConvergenceResult.result) {
    return stepSizeConvergenceResult;
  }

  // Log progress with detailed information
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
  // Check constraint satisfaction
  const { constraint, constraintNorm } = checkConstraintViolation(
    currentParameters,
    currentStates,
    constraintFunction,
    constraintTolerance,
    iteration,
    logger
  );

  // Compute adjoint gradient and norm
  const { adjointGradient, gradientNorm, partials } = computeAdjointGradientAndNorm(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger
  );

  // Handle callback and check gradient convergence
  const convergenceResult = checkConvergenceAndHandleCallback(
    iteration,
    currentParameters,
    currentStates,
    currentCost,
    gradientNorm,
    constraintNorm,
    tolerance,
    usedLineSearchFlag,
    onIteration,
    logger
  );
  if (convergenceResult.converged && convergenceResult.result) {
    return convergenceResult;
  }

  // Handle step size and update
  const updateResult = handleStepSizeAndUpdate(
    adjointGradient,
    currentParameters,
    currentStates,
    constraint,
    currentCost,
    gradientNorm,
    constraintNorm,
    iteration,
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


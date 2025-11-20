/**
 * This file provides shared utility functions for constrained optimization algorithms
 * using the adjoint method.
 * 
 * Role in system:
 * - Eliminates code duplication between adjointGradientDescent, constrainedGaussNewton, and constrainedLevenbergMarquardt
 * - Centralizes adjoint method computation logic (DRY principle)
 * - Provides reusable functions for state updates and constraint handling
 * 
 * For first-time readers:
 * - These are utility functions used internally by constrained optimization algorithms
 * - solveAdjointEquation: Solves the adjoint equation for computing gradients
 * - updateStates: Updates states using linear approximation to maintain constraint satisfaction
 * - validateInitialConditions: Validates initial states and constraints
 * 
 * Extracted from adjointGradientDescent.ts to enable code reuse.
 */

import { Matrix, solve, CholeskyDecomposition } from 'ml-matrix';
import type { ConstraintFn } from './types.js';
import { vectorNorm, scaleVector, addVectors } from '../utils/matrix.js';
import { float64ArrayToMatrix, matrixToFloat64Array } from '../utils/matrix.js';
import { Logger } from './logger.js';

const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negating vectors

/**
 * Solves the adjoint equation: (∂c/∂x)^T λ = rhs
 * Returns the adjoint variable λ.
 * 
 * This is the core of the adjoint method, used for efficient gradient computation
 * without explicitly inverting matrices.
 * 
 * @param dcdx - Constraint Jacobian ∂c/∂x (must be square)
 * @param rhs - Right-hand side vector (e.g., (∂f/∂x)^T or (r_x^T r))
 * @param logger - Logger instance for error reporting
 * @param algorithmName - Name of calling algorithm (for error messages)
 * @returns Adjoint variable λ
 */
export function solveAdjointEquation(
  dcdx: Matrix,
  rhs: Float64Array,
  logger: Logger,
  algorithmName: string = 'constrainedOptimization'
): Float64Array {
  // Check if dcdx is square
  if (dcdx.rows !== dcdx.columns) {
    throw new Error(
      `Constraint Jacobian ∂c/∂x must be square (constraintCount == stateCount) for adjoint method. ` +
      `Got ${dcdx.rows} × ${dcdx.columns} matrix.`
    );
  }

  // Transpose dcdx: (∂c/∂x)^T
  const dcdxTranspose = dcdx.transpose();

  // Right-hand side as column vector
  const rhsMatrix = float64ArrayToMatrix(rhs);

  // Solve: (∂c/∂x)^T λ = rhs
  let lambdaMatrix: Matrix;
  try {
    // Try Cholesky decomposition first for efficiency
    const cholesky = new CholeskyDecomposition(dcdxTranspose);
    if (cholesky.isPositiveDefinite()) {
      lambdaMatrix = cholesky.solve(rhsMatrix);
    } else {
      // Fallback to LU decomposition
      lambdaMatrix = solve(dcdxTranspose, rhsMatrix);
    }
  } catch (error) {
    // Fallback to LU decomposition if Cholesky fails
    try {
      lambdaMatrix = solve(dcdxTranspose, rhsMatrix);
    } catch (solveError) {
      logger.warn(algorithmName, undefined, `Failed to solve adjoint equation: ${solveError}`);
      throw new Error(
        `Failed to solve adjoint equation (∂c/∂x)^T λ = rhs. ` +
        `The constraint Jacobian ∂c/∂x may be singular or ill-conditioned. ` +
        `Original error: ${solveError}`
      );
    }
  }

  return matrixToFloat64Array(lambdaMatrix);
}

/**
 * Updates states using linear approximation: x_new = x_old + dx
 * where dx solves (∂c/∂x) dx = -∂c/∂p · Δp
 * 
 * This maintains constraint satisfaction approximately using first-order Taylor expansion.
 * For large steps, constraints may be violated slightly, but the algorithm will correct
 * this in subsequent iterations.
 * 
 * @param currentStates - Current state vector x
 * @param dcdx - Constraint Jacobian ∂c/∂x
 * @param dcdp - Constraint Jacobian ∂c/∂p
 * @param deltaP - Parameter change Δp
 * @returns Updated state vector x_new
 */
export function updateStates(
  currentStates: Float64Array,
  dcdx: Matrix,
  dcdp: Matrix,
  deltaP: Float64Array
): Float64Array {
  // Compute ∂c/∂p · Δp
  const deltaPMatrix = float64ArrayToMatrix(deltaP);
  const dcdpDeltaP = dcdp.mmul(deltaPMatrix);
  const dcdpDeltaPVector = matrixToFloat64Array(dcdpDeltaP);

  // Solve: (∂c/∂x) dx = -∂c/∂p · Δp
  const negativeDcdpDeltaP = scaleVector(dcdpDeltaPVector, NEGATIVE_COEFFICIENT);
  const negativeDcdpDeltaPMatrix = float64ArrayToMatrix(negativeDcdpDeltaP);
  
  let dxMatrix: Matrix;
  try {
    const cholesky = new CholeskyDecomposition(dcdx);
    if (cholesky.isPositiveDefinite()) {
      dxMatrix = cholesky.solve(negativeDcdpDeltaPMatrix);
    } else {
      dxMatrix = solve(dcdx, negativeDcdpDeltaPMatrix);
    }
  } catch (error) {
    dxMatrix = solve(dcdx, negativeDcdpDeltaPMatrix);
  }

  const dx = matrixToFloat64Array(dxMatrix);

  // x_new = x_old + dx
  return addVectors(currentStates, dx);
}

/**
 * Validates initial conditions including constraint satisfaction and dimensions.
 * 
 * Checks that:
 * 1. Constraint count equals state count (required for adjoint method)
 * 2. Initial constraint violation is within tolerance (warns if not)
 * 
 * @param initialParameters - Initial parameter vector p0
 * @param initialStates - Initial state vector x0
 * @param constraintFunction - Constraint function c(p, x) = 0
 * @param constraintTolerance - Tolerance for constraint violation
 * @param logger - Logger instance for warnings
 * @param algorithmName - Name of calling algorithm (for error messages)
 * @throws Error if constraint count != state count
 */
export function validateInitialConditions(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  logger: Logger,
  algorithmName: string = 'constrainedOptimization'
): void {
  const initialConstraint = constraintFunction(initialParameters, initialStates);
  const initialConstraintNorm = vectorNorm(initialConstraint);
  if (initialConstraintNorm > constraintTolerance) {
    logger.warn(algorithmName, undefined, 'Initial constraint violation', [
      { key: '||c(p0,x0)||:', value: initialConstraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }

  // Validate constraint dimensions
  const constraintCount = initialConstraint.length;
  const stateCount = initialStates.length;
  if (constraintCount !== stateCount) {
    throw new Error(
      `Constraint count (${constraintCount}) must equal state count (${stateCount}) ` +
      `for adjoint method. The constraint Jacobian ∂c/∂x must be square. ` +
      `Algorithm: ${algorithmName}`
    );
  }
}


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

import { Matrix, solve, CholeskyDecomposition, QR, pseudoInverse } from 'ml-matrix';
import type { ConstraintFn } from './types.js';
import { vectorNorm, scaleVector, addVectors } from '../utils/matrix.js';
import { float64ArrayToMatrix, matrixToFloat64Array } from '../utils/matrix.js';
import { Logger } from './logger.js';

const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negating vectors

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
 * @param algorithmName - Name of calling algorithm (for error messages)
 * @returns Solution vector x as Float64Array
 */
export function solveLeastSquares(
  A: Matrix,
  b: Matrix,
  logger: Logger,
  algorithmName: string = 'constrainedOptimization'
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
        logger.warn(algorithmName, undefined, `Failed to solve square system: ${solveError}`);
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
          logger.warn(algorithmName, undefined, `All methods failed for overdetermined system: QR=${qrError}, Normal=${normalError}, PseudoInv=${pinvError}`);
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
      logger.warn(algorithmName, undefined, `Failed to solve underdetermined system with pseudoInverse: ${pinvError}`);
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
 * Solves the adjoint equation: (∂c/∂x)^T λ = rhs
 * Returns the adjoint variable λ.
 * Supports both square and non-square constraint Jacobians.
 * 
 * This is the core of the adjoint method, used for efficient gradient computation
 * without explicitly inverting matrices.
 * 
 * @param dcdx - Constraint Jacobian ∂c/∂x
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
  // Transpose dcdx: (∂c/∂x)^T
  const dcdxTranspose = dcdx.transpose();

  // Right-hand side as column vector
  const rhsMatrix = float64ArrayToMatrix(rhs);

  // Solve: (∂c/∂x)^T λ = rhs
  // Uses hierarchical solver that handles both square and non-square matrices
  try {
    return solveLeastSquares(dcdxTranspose, rhsMatrix, logger, algorithmName);
  } catch (error) {
    logger.warn(algorithmName, undefined, `Failed to solve adjoint equation: ${error}`);
    throw new Error(
      `Failed to solve adjoint equation (∂c/∂x)^T λ = rhs. ` +
      `The constraint Jacobian ∂c/∂x may be singular or ill-conditioned. ` +
      `Matrix size: ${dcdx.rows} × ${dcdx.columns}. ` +
      `Original error: ${error}`
    );
  }
}

/**
 * Updates states using linear approximation: x_new = x_old + dx
 * where dx solves (∂c/∂x) dx = -∂c/∂p · Δp
 * Supports both square and non-square constraint Jacobians.
 * 
 * This maintains constraint satisfaction approximately using first-order Taylor expansion.
 * For large steps, constraints may be violated slightly, but the algorithm will correct
 * this in subsequent iterations.
 * 
 * @param currentStates - Current state vector x
 * @param dcdx - Constraint Jacobian ∂c/∂x
 * @param dcdp - Constraint Jacobian ∂c/∂p
 * @param deltaP - Parameter change Δp
 * @param logger - Logger instance for error reporting
 * @param algorithmName - Name of calling algorithm (for error messages)
 * @returns Updated state vector x_new
 */
export function updateStates(
  currentStates: Float64Array,
  dcdx: Matrix,
  dcdp: Matrix,
  deltaP: Float64Array,
  logger: Logger,
  algorithmName: string = 'constrainedOptimization'
): Float64Array {
  // Compute ∂c/∂p · Δp
  const deltaPMatrix = float64ArrayToMatrix(deltaP);
  const dcdpDeltaP = dcdp.mmul(deltaPMatrix);
  const dcdpDeltaPVector = matrixToFloat64Array(dcdpDeltaP);

  // Solve: (∂c/∂x) dx = -∂c/∂p · Δp
  const negativeDcdpDeltaP = scaleVector(dcdpDeltaPVector, NEGATIVE_COEFFICIENT);
  const negativeDcdpDeltaPMatrix = float64ArrayToMatrix(negativeDcdpDeltaP);
  
  // Use hierarchical solver that handles both square and non-square matrices
  const dx = solveLeastSquares(dcdx, negativeDcdpDeltaPMatrix, logger, algorithmName);

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

  // Note: Constraint count and state count no longer need to match.
  // The adjoint method now supports non-square constraint Jacobians.
}


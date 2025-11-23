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
import { finiteDiffConstraintPartialX } from './finiteDiff.js';

const NEGATIVE_COEFFICIENT = -1.0; // Coefficient for negating vectors
const MAX_DIAG_LOG_DIM = 40; // upper bound to log row/col diagnostics
const MAX_REGULARIZATION_ATTEMPTS = 8; // Maximum number of regularization attempts when solving linear systems
const REGULARIZATION_BASE = 10; // Base for exponential regularization scaling
const REGULARIZATION_INITIAL_EXPONENT = -8; // Initial exponent for regularization: 10^(-8)
const REGULARIZATION_MAX_EXPONENT = 7; // Maximum exponent for regularization: 10^7
const REGULARIZATION_FALLBACK_EXPONENT = -1; // Fallback exponent when baseReg is 0: 10^(-1)
const MAX_DIAGNOSTIC_ENTRIES = 5; // Maximum number of smallest rows/columns to include in diagnostics

function checkFiniteMatrix(mat: Matrix): { ok: boolean; firstBad?: { row: number; col: number; value: number } } {
  for (let r = 0; r < mat.rows; r++) {
    for (let c = 0; c < mat.columns; c++) {
      const v = mat.get(r, c);
      if (!Number.isFinite(v)) {
        return { ok: false, firstBad: { row: r, col: c, value: v } };
      }
    }
  }
  return { ok: true };
}

function computeVectorDiagnostics(b: Matrix): { norm: number; minAbs: number; maxAbs: number } {
  let sum = 0;
  let minAbs = Number.POSITIVE_INFINITY;
  let maxAbs = 0;
  for (let r = 0; r < b.rows; r++) {
    for (let c = 0; c < b.columns; c++) {
      const v = b.get(r, c);
      const abs = Math.abs(v);
      sum += v * v;
      minAbs = Math.min(minAbs, abs);
      maxAbs = Math.max(maxAbs, abs);
    }
  }
  return { norm: Math.sqrt(sum), minAbs: minAbs === Number.POSITIVE_INFINITY ? 0 : minAbs, maxAbs };
}

function computeRowColDiagnostics(A: Matrix): {
  minRowNorm: number;
  maxRowNorm: number;
  minColNorm: number;
  maxColNorm: number;
  smallestRows: Array<{ index: number; norm: number }>;
  smallestCols: Array<{ index: number; norm: number }>;
} {
  const rowNorms: number[] = [];
  const colNorms: number[] = [];

  for (let r = 0; r < A.rows; r++) {
    let sum = 0;
    for (let c = 0; c < A.columns; c++) {
      const v = A.get(r, c);
      sum += v * v;
    }
    rowNorms.push(Math.sqrt(sum));
  }

  for (let c = 0; c < A.columns; c++) {
    let sum = 0;
    for (let r = 0; r < A.rows; r++) {
      const v = A.get(r, c);
      sum += v * v;
    }
    colNorms.push(Math.sqrt(sum));
  }

  const rowPairs = rowNorms.map((norm, index) => ({ index, norm })).sort((a, b) => a.norm - b.norm).slice(0, MAX_DIAGNOSTIC_ENTRIES);
  const colPairs = colNorms.map((norm, index) => ({ index, norm })).sort((a, b) => a.norm - b.norm).slice(0, MAX_DIAGNOSTIC_ENTRIES);

  return {
    minRowNorm: Math.min(...rowNorms),
    maxRowNorm: Math.max(...rowNorms),
    minColNorm: Math.min(...colNorms),
    maxColNorm: Math.max(...colNorms),
    smallestRows: rowPairs,
    smallestCols: colPairs
  };
}

/**
 * Computes regularization lambda for a given attempt.
 * Uses exponential scaling to gradually increase regularization strength.
 */
function computeRegularizationLambda(
  baseReg: number,
  attempt: number
): number {
  return baseReg > 0
    ? baseReg * Math.pow(REGULARIZATION_BASE, attempt)
    : Math.pow(REGULARIZATION_BASE, REGULARIZATION_INITIAL_EXPONENT + attempt);
}

/**
 * Attempts to solve linear system with regularization retries.
 * Cholesky decomposition is preferred for efficiency, falls back to general solver.
 */
function trySolveWithRegularization(
  A: Matrix,
  b: Matrix,
  baseReg: number,
  logger: Logger,
  algorithmName: string
): Float64Array {
  let lastError: unknown;
  for (let attempt = 0; attempt < MAX_REGULARIZATION_ATTEMPTS; attempt++) {
    const lambda = computeRegularizationLambda(baseReg, attempt);
    const AwithReg = A.add(Matrix.eye(A.rows, A.columns).mul(lambda));
    try {
      const chol = new CholeskyDecomposition(AwithReg);
      if (chol.isPositiveDefinite()) {
        return matrixToFloat64Array(chol.solve(b));
      }
    } catch (err) {
      lastError = err;
    }
    try {
      return matrixToFloat64Array(solve(AwithReg, b));
    } catch (err) {
      lastError = err;
      continue;
    }
  }
  logger.warn(algorithmName, undefined, `Failed to solve system with regularization: ${lastError}`);
  throw new Error(
    `Failed to solve linear system even with Tikhonov regularization. ` +
    `Matrix may be singular or ill-conditioned. Last error: ${lastError}`
  );
}

/**
 * Solves square system Ax = b with validation and regularization.
 * Fast path for square matrices using direct Cholesky/LU decomposition.
 */
function solveSquareSystem(
  Areg: Matrix,
  b: Matrix,
  regularization: number,
  logger: Logger,
  algorithmName: string
): Float64Array {
  const baseReg = regularization > 0 ? regularization : 0;
  const diagnostics =
    Areg.rows <= MAX_DIAG_LOG_DIM && Areg.columns <= MAX_DIAG_LOG_DIM
      ? computeRowColDiagnostics(Areg)
      : undefined;
  const rhsDiagnostics = computeVectorDiagnostics(b);

  const dimsOk = Areg.rows === b.rows;
  const Afinite = checkFiniteMatrix(Areg);
  const bFinite = checkFiniteMatrix(b);
  if (!dimsOk || !Afinite.ok || !bFinite.ok) {
    const detailRows: Array<{ key: string; value: number | string }> = [
      { key: 'A_rows', value: Areg.rows },
      { key: 'A_cols', value: Areg.columns },
      { key: 'b_rows', value: b.rows },
      { key: 'b_cols', value: b.columns }
    ];
    if (!Afinite.ok && Afinite.firstBad) {
      detailRows.push({ key: 'A_bad_row', value: Afinite.firstBad.row });
      detailRows.push({ key: 'A_bad_col', value: Afinite.firstBad.col });
      detailRows.push({ key: 'A_bad_val', value: Afinite.firstBad.value });
    }
    if (!bFinite.ok && bFinite.firstBad) {
      detailRows.push({ key: 'b_bad_row', value: bFinite.firstBad.row });
      detailRows.push({ key: 'b_bad_col', value: bFinite.firstBad.col });
      detailRows.push({ key: 'b_bad_val', value: bFinite.firstBad.value });
    }
    const numericDetails = detailRows.filter(d => typeof d.value === 'number') as Array<{ key: string; value: number }>;
    logger.warn(algorithmName, undefined, 'Invalid dimensions or NaN/Inf detected before solve', numericDetails);
    throw new Error('Invalid dimensions or NaN/Inf in inputs for square solve');
  }

  try {
    return trySolveWithRegularization(Areg, b, baseReg, logger, algorithmName);
  } catch (error) {
    const detailRows: Array<{ key: string; value: number | string }> = [
      { key: 'rows', value: Areg.rows },
      { key: 'cols', value: Areg.columns },
      { key: 'reg_final', value: baseReg > 0 ? baseReg * Math.pow(REGULARIZATION_BASE, REGULARIZATION_MAX_EXPONENT) : Math.pow(REGULARIZATION_BASE, REGULARIZATION_FALLBACK_EXPONENT) }
    ];
    detailRows.push(
      { key: 'rhs_norm', value: rhsDiagnostics.norm },
      { key: 'rhs_min_abs', value: rhsDiagnostics.minAbs },
      { key: 'rhs_max_abs', value: rhsDiagnostics.maxAbs }
    );
    if (diagnostics) {
      detailRows.push(
        { key: 'minRowNorm', value: diagnostics.minRowNorm },
        { key: 'minColNorm', value: diagnostics.minColNorm },
        { key: 'maxRowNorm', value: diagnostics.maxRowNorm },
        { key: 'maxColNorm', value: diagnostics.maxColNorm }
      );
      diagnostics.smallestRows.forEach((row, idx) => {
        detailRows.push({ key: `row_${idx}`, value: row.index });
        detailRows.push({ key: `row_${idx}_norm`, value: row.norm });
      });
      diagnostics.smallestCols.forEach((col, idx) => {
        detailRows.push({ key: `col_${idx}`, value: col.index });
        detailRows.push({ key: `col_${idx}_norm`, value: col.norm });
      });
    }
    const numericDetails = detailRows.filter(d => typeof d.value === 'number') as Array<{ key: string; value: number }>;
    logger.warn(algorithmName, undefined, `Failed to solve square system with regularization up to ~1e0: ${error}`, numericDetails);
    throw error;
  }
}

/**
 * Solves overdetermined system (rows > columns) using normal equations.
 * Converts to square system A^T A x = A^T b for efficient Cholesky solution.
 */
function solveOverdeterminedSystem(
  A: Matrix,
  b: Matrix,
  regularization: number,
  logger: Logger,
  algorithmName: string
): Float64Array {
  const AT = A.transpose();
  const ATA = AT.mmul(A);
  const ATb = AT.mmul(b);
  const baseReg = regularization > 0 ? regularization : 0;
  
  try {
    return trySolveWithRegularization(ATA, ATb, baseReg, logger, algorithmName);
  } catch (error) {
    logger.warn(algorithmName, undefined, `Failed to solve overdetermined system with normal equations: ${error}`);
    throw new Error(
      `Failed to solve overdetermined system Ax = b using normal equations A^T A x = A^T b. ` +
      `Matrix A^T A may be singular or ill-conditioned. Last error: ${error}`
    );
  }
}

/**
 * Solves underdetermined system (rows < columns) using normal equations.
 * Strategy: solve A A^T y = b, then x = A^T y for minimum-norm solution.
 */
function solveUnderdeterminedSystem(
  A: Matrix,
  b: Matrix,
  regularization: number,
  logger: Logger,
  algorithmName: string
): Float64Array {
  const AT = A.transpose();
  const AAT = A.mmul(AT);
  const baseReg = regularization > 0 ? regularization : 0;
  
  try {
    const y = trySolveWithRegularization(AAT, b, baseReg, logger, algorithmName);
    const x = AT.mmul(float64ArrayToMatrix(y));
    return matrixToFloat64Array(x);
  } catch (error) {
    logger.warn(algorithmName, undefined, `Failed to solve underdetermined system with normal equations: ${error}`);
    throw new Error(
      `Failed to solve underdetermined system Ax = b using normal equations A A^T y = b, x = A^T y. ` +
      `Matrix A A^T may be singular or ill-conditioned. Last error: ${error}`
    );
  }
}

/**
 * Solves a least squares problem Ax = b using Cholesky decomposition.
 * For square matrices, uses Cholesky/LU decomposition directly.
 * For non-square matrices, uses normal equations to convert to square system:
 * - Overdetermined (rows > columns): A^T A x = A^T b (Cholesky on A^T A)
 * - Underdetermined (rows < columns): A A^T y = b, x = A^T y (Cholesky on A A^T)
 * This approach avoids SVD/pseudoInverse entirely for better performance.
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
  algorithmName: string = 'constrainedOptimization',
  regularization: number = 0
): Float64Array {
  const Areg =
    regularization > 0 && A.rows === A.columns
      ? A.add(Matrix.eye(A.rows, A.columns).mul(regularization))
      : A;

  if (Areg.rows === Areg.columns) {
    return solveSquareSystem(Areg, b, regularization, logger, algorithmName);
  }

  const isOverdetermined = A.rows > A.columns;
  if (isOverdetermined) {
    return solveOverdeterminedSystem(A, b, regularization, logger, algorithmName);
  }

  return solveUnderdeterminedSystem(A, b, regularization, logger, algorithmName);
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
  algorithmName: string = 'constrainedOptimization',
  regularization: number = 0
): Float64Array {
  // Transpose needed to form adjoint equation: (∂c/∂x)^T λ = rhs (standard form for linear solve)
  const dcdxTranspose = dcdx.transpose();

  const rhsMatrix = float64ArrayToMatrix(rhs);

  // Hierarchical solver handles both square and non-square constraint Jacobians efficiently
  try {
    return solveLeastSquares(dcdxTranspose, rhsMatrix, logger, algorithmName, regularization);
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
  // Compute how parameter changes affect constraints: needed to determine state updates
  const deltaPMatrix = float64ArrayToMatrix(deltaP);
  const dcdpDeltaP = dcdp.mmul(deltaPMatrix);
  const dcdpDeltaPVector = matrixToFloat64Array(dcdpDeltaP);

  // Linear approximation maintains constraint satisfaction: (∂c/∂x) dx = -∂c/∂p · Δp
  const negativeDcdpDeltaP = scaleVector(dcdpDeltaPVector, NEGATIVE_COEFFICIENT);
  const negativeDcdpDeltaPMatrix = float64ArrayToMatrix(negativeDcdpDeltaP);
  
  // Hierarchical solver efficiently handles both square and non-square constraint Jacobians
  const dx = solveLeastSquares(dcdx, negativeDcdpDeltaPMatrix, logger, algorithmName);

  return addVectors(currentStates, dx);
}

/**
 * Projects states onto the constraint manifold for fixed parameters using
 * a few Newton correction steps: (∂c/∂x) Δx = -c(p, x).
 * This is a standard feasibility-restoration step consistent with the
 * implicit function theorem (solving c(p, x) = 0 locally).
 */
export function projectStatesToConstraints(
  parameters: Float64Array,
  states: Float64Array,
  constraintFunction: ConstraintFn,
  stepSizeX: number,
  constraintTolerance: number,
  logger: Logger,
  algorithmName: string = 'constrainedOptimization',
  maxIterations: number = 3
): Float64Array {
  let projectedStates = new Float64Array(states);

  for (let i = 0; i < maxIterations; i++) {
    const constraint = constraintFunction(parameters, projectedStates);
    const constraintNorm = vectorNorm(constraint);
    if (constraintNorm <= constraintTolerance) {
      break;
    }

    const dcdx = finiteDiffConstraintPartialX(parameters, projectedStates, constraintFunction, { stepSize: stepSizeX });
    const negativeConstraint = scaleVector(constraint, NEGATIVE_COEFFICIENT);
    const negativeConstraintMatrix = float64ArrayToMatrix(negativeConstraint);

    try {
      const deltaX = solveLeastSquares(dcdx, negativeConstraintMatrix, logger, algorithmName);
      const updatedStates = addVectors(projectedStates, deltaX);
      projectedStates = new Float64Array(updatedStates);
    } catch (error) {
      logger.warn(algorithmName, undefined, `Failed to project onto constraints: ${error}`);
      break;
    }
  }

  return projectedStates;
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


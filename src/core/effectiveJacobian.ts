/**
 * This file implements the effective Jacobian computation for constrained optimization.
 * 
 * The effective Jacobian J_eff = dr/dp = r_p - r_x C_x^+ C_p captures all constraint
 * effects, allowing constrained least squares problems to be solved using the same
 * structure as unconstrained problems.
 * 
 * Mathematical background:
 * - For constrained residual r(p, x) where c(p, x) = 0, the implicit function
 *   theorem gives: dr/dp = r_p - r_x C_x^+ C_p
 * - We compute this efficiently by solving C_x dx = (C_p)_j for each parameter j,
 *   then (J_eff)_j = (r_p)_j - r_x dx
 * - This reuses the C_x decomposition across all columns for efficiency.
 * 
 * Role in system:
 * - Core component for constrained Gauss-Newton and Levenberg-Marquardt methods
 * - Enables efficient constrained least squares optimization
 * - Uses direct linear solve to avoid explicit matrix inversion
 * 
 * For first-time readers:
 * - Start with computeEffectiveJacobian function
 * - Understand how each column is computed efficiently
 * - Note the reuse of C_x decomposition for performance
 */

import { Matrix, solve, CholeskyDecomposition } from 'ml-matrix';
import type {
  ConstrainedResidualFn,
  ConstraintFn
} from './types.js';
import {
  finiteDiffResidualPartialP,
  finiteDiffResidualPartialX,
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX
} from './finiteDiff.js';
import { Logger } from './logger.js';
import { float64ArrayToMatrix, matrixToFloat64Array } from '../utils/matrix.js';

const DEFAULT_STEP_SIZE_P = 1e-6;
const DEFAULT_STEP_SIZE_X = 1e-6;

/**
 * Options for effective Jacobian computation.
 */
export interface EffectiveJacobianOptions {
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
}

/**
 * Computes the effective Jacobian J_eff = r_p - r_x C_x^+ C_p.
 * 
 * Algorithm:
 * 1. Compute partial derivatives: r_p, r_x, C_p, C_x
 * 2. For each column j (parameter j):
 *    a. Extract j-th column of C_p: (C_p)_j
 *    b. Solve: C_x dx = (C_p)_j to get dx = C_x^-1 (C_p)_j
 *    c. Compute column: (J_eff)_j = (r_p)_j - r_x dx
 * 3. Return effective Jacobian matrix
 * 
 * The C_x decomposition is reused across all columns for efficiency.
 * 
 * @param parameters - Parameter vector p
 * @param states - State vector x (satisfies c(p, x) = 0)
 * @param residualFunction - Residual function r(p, x)
 * @param constraintFunction - Constraint function c(p, x) = 0
 * @param options - Options for Jacobian computation
 * @param logger - Logger instance for error reporting
 * @param algorithmName - Name of calling algorithm (for error messages)
 * @returns Effective Jacobian matrix J_eff (residualCount × parameterCount)
 */
export function computeEffectiveJacobian(
  parameters: Float64Array,
  states: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: EffectiveJacobianOptions = {},
  logger: Logger,
  algorithmName: string = 'constrainedOptimization'
): Matrix {
  const stepSizeP = options.stepSizeP ?? DEFAULT_STEP_SIZE_P;
  const stepSizeX = options.stepSizeX ?? DEFAULT_STEP_SIZE_X;

  // Compute partial derivatives
  const r_p = options.drdp
    ? options.drdp(parameters, states)
    : finiteDiffResidualPartialP(parameters, states, residualFunction, { stepSize: stepSizeP });

  const r_x = options.drdx
    ? options.drdx(parameters, states)
    : finiteDiffResidualPartialX(parameters, states, residualFunction, { stepSize: stepSizeX });

  const c_p = options.dcdp
    ? options.dcdp(parameters, states)
    : finiteDiffConstraintPartialP(parameters, states, constraintFunction, { stepSize: stepSizeP });

  const c_x = options.dcdx
    ? options.dcdx(parameters, states)
    : finiteDiffConstraintPartialX(parameters, states, constraintFunction, { stepSize: stepSizeX });

  // Get dimensions
  const residualCount = r_p.rows;
  const parameterCount = r_p.columns;
  const stateCount = c_x.columns;

  // Validate dimensions
  if (c_x.rows !== stateCount) {
    const errorMsg = `Constraint Jacobian ∂c/∂x must be square (constraintCount == stateCount) for adjoint method. ` +
      `Got ${c_x.rows} × ${stateCount} matrix. Algorithm: ${algorithmName}`;
    logger.warn(algorithmName, undefined, errorMsg);
    throw new Error(errorMsg);
  }

  if (r_x.columns !== stateCount) {
    const errorMsg = `Residual Jacobian ∂r/∂x must have stateCount columns. ` +
      `Got ${r_x.rows} × ${r_x.columns}, expected ${r_x.rows} × ${stateCount}. Algorithm: ${algorithmName}`;
    logger.warn(algorithmName, undefined, errorMsg);
    throw new Error(errorMsg);
  }

  if (c_p.columns !== parameterCount) {
    const errorMsg = `Constraint Jacobian ∂c/∂p must have parameterCount columns. ` +
      `Got ${c_p.rows} × ${c_p.columns}, expected ${c_p.rows} × ${parameterCount}. Algorithm: ${algorithmName}`;
    logger.warn(algorithmName, undefined, errorMsg);
    throw new Error(errorMsg);
  }

  // Initialize effective Jacobian matrix (residualCount × parameterCount)
  const effectiveJacobianData: number[][] = [];
  for (let i = 0; i < residualCount; i++) {
    effectiveJacobianData.push(new Array(parameterCount).fill(0));
  }

  // Compute each column of effective Jacobian
  for (let paramIndex = 0; paramIndex < parameterCount; paramIndex++) {
    const column = computeEffectiveJacobianColumn(
      paramIndex,
      c_p,
      c_x,
      r_p,
      r_x,
      stateCount,
      residualCount
    );
    for (let i = 0; i < residualCount; i++) {
      effectiveJacobianData[i][paramIndex] = column[i];
    }
  }

  return new Matrix(effectiveJacobianData);
}

/**
 * Computes a single column of the effective Jacobian.
 * For parameter j: (J_eff)_j = (r_p)_j - r_x dx, where dx solves C_x dx = (C_p)_j
 */
function computeEffectiveJacobianColumn(
  paramIndex: number,
  constraintJacobianP: Matrix,
  constraintJacobianX: Matrix,
  residualJacobianP: Matrix,
  residualJacobianX: Matrix,
  stateCount: number,
  residualCount: number
): Float64Array {
  // Extract j-th column of C_p: (C_p)_j
  const constraintJacobianPColumn = new Float64Array(stateCount);
  for (let k = 0; k < stateCount; k++) {
    constraintJacobianPColumn[k] = constraintJacobianP.get(k, paramIndex);
  }

  // Solve: C_x dx = (C_p)_j to get dx = C_x^-1 (C_p)_j
  const constraintJacobianPColumnMatrix = float64ArrayToMatrix(constraintJacobianPColumn);
  let stateSensitivityMatrix: Matrix;
  try {
    const cholesky = new CholeskyDecomposition(constraintJacobianX);
    if (cholesky.isPositiveDefinite()) {
      stateSensitivityMatrix = cholesky.solve(constraintJacobianPColumnMatrix);
    } else {
      stateSensitivityMatrix = solve(constraintJacobianX, constraintJacobianPColumnMatrix);
    }
  } catch (error) {
    stateSensitivityMatrix = solve(constraintJacobianX, constraintJacobianPColumnMatrix);
  }

  const stateSensitivity = matrixToFloat64Array(stateSensitivityMatrix);

  // Compute: (J_eff)_j = (r_p)_j - r_x dx
  const effectiveJacobianColumn = new Float64Array(residualCount);
  for (let i = 0; i < residualCount; i++) {
    let residualJacobianXTimesStateSensitivity = 0;
    for (let k = 0; k < stateCount; k++) {
      residualJacobianXTimesStateSensitivity += residualJacobianX.get(i, k) * stateSensitivity[k];
    }
    effectiveJacobianColumn[i] = residualJacobianP.get(i, paramIndex) - residualJacobianXTimesStateSensitivity;
  }

  return effectiveJacobianColumn;
}

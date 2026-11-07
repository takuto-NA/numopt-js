/**
 * Effective residual Jacobian J_eff = r_p - r_x C_x^+ C_p for constrained NLS.
 * Reuses one C_x factorization across parameter columns.
 */

import { Matrix } from 'ml-matrix';
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
import { solveLeastSquaresMultiple } from './leastSquaresMultiple.js';

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
 * 2. Solve C_x X = C_p with all parameter columns as right-hand sides.
 * 3. Return r_p - r_x X.
 * 
 * Regular square/tall systems reuse one LU/QR decomposition across columns.
 * This is a reduced derivative only where a feasible, locally unique x(p) exists:
 * C_x must have full column rank and C_p must lie in its column space.
 * Otherwise rectangular least-squares/regularization gives a linearization, not proof of x(p).
 * Inputs must already be feasible; this function does not solve or project states.
 * For f=1/2 ||r||², the reduced gradient is J_eff^T r.
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
  logger: Logger = new Logger(undefined, undefined),
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

  const residualCount = residualFunction(parameters, states).length;
  const constraintCount = constraintFunction(parameters, states).length;
  assertShape(r_p, residualCount, parameters.length, 'Residual Jacobian ∂r/∂p');
  assertShape(r_x, residualCount, states.length, 'Residual Jacobian ∂r/∂x');
  assertShape(c_p, constraintCount, parameters.length, 'Constraint Jacobian ∂c/∂p');
  assertShape(c_x, constraintCount, states.length, 'Constraint Jacobian ∂c/∂x');
  const sensitivity = solveLeastSquaresMultiple(c_x, c_p, logger, algorithmName);
  const effectiveJacobian = r_p.clone().sub(r_x.mmul(sensitivity));
  if (!effectiveJacobian.to1DArray().every(Number.isFinite)) throw new Error('Effective residual Jacobian contains NaN or Infinity');
  return effectiveJacobian;
}

function assertShape(matrix: Matrix, rows: number, columns: number, label: string): void {
  if (matrix.rows !== rows || matrix.columns !== columns) {
    throw new Error(`${label} must be ${rows} × ${columns}; got ${matrix.rows} × ${matrix.columns}`);
  }
}

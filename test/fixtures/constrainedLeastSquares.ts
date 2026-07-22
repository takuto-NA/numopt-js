/**
 * Shared equality-constrained least-squares toy problem for solver tests.
 *
 * Minimize  f(p, x) = 1/2 ((p - 0.5)² + (x - 0.5)²)
 * Subject to  c(p, x) = p + x - 1 = 0
 * Optimum: p = 0.5, x = 0.5, f = 0
 */

import { Matrix } from 'ml-matrix';
import type { ConstrainedResidualFn, ConstraintFn } from '../../src/core/types';

export const CONSTRAINED_LS_TARGET_PARAMETER = 0.5;
export const CONSTRAINED_LS_TARGET_STATE = 0.5;
export const CONSTRAINED_LS_PARAMETER_TOLERANCE = 1e-3;
export const CONSTRAINED_LS_COST_TOLERANCE = 1e-5;
export const CONSTRAINED_LS_CONSTRAINT_TOLERANCE = 1e-3;

export const constrainedLeastSquaresResidual: ConstrainedResidualFn = (parameters, states) => {
  return new Float64Array([
    parameters[0] - CONSTRAINED_LS_TARGET_PARAMETER,
    states[0] - CONSTRAINED_LS_TARGET_STATE
  ]);
};

export const constrainedLeastSquaresConstraint: ConstraintFn = (parameters, states) => {
  return new Float64Array([parameters[0] + states[0] - 1.0]);
};

export function createConstrainedLeastSquaresInitial(): {
  parameters: Float64Array;
  states: Float64Array;
} {
  return {
    parameters: new Float64Array([2.0]),
    states: new Float64Array([-1.0])
  };
}

export function createConstrainedLeastSquaresAnalyticalDerivatives() {
  return {
    drdp: () => new Matrix([[1], [0]]),
    drdx: () => new Matrix([[0], [1]]),
    dcdp: () => new Matrix([[1]]),
    dcdx: () => new Matrix([[1]])
  };
}

export function halfSquaredResidualNorm(parameters: Float64Array, states: Float64Array): number {
  const residual = constrainedLeastSquaresResidual(parameters, states);
  return 0.5 * (residual[0] * residual[0] + residual[1] * residual[1]);
}

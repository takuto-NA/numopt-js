/**
 * Newton-adjoint with all terms of the reduced Hessian, via the projected Lagrangian Hessian.
 */

import type {
  AdjointNewtonExactOptions, AdjointNewtonExactResult, ConstrainedCostFn, ConstrainedResidualFn, ConstraintFn
} from './types.js';
import { runAdjointNewton } from './adjointNewton.js';

/**
 * Minimize f(p,x(p)) with H=Z^T ∇²(f-lambda^T c) Z and the adjoint reduced gradient.
 * Includes mixed partials and nonlinear state curvature. Numerical derivatives have finite-difference error.
 * Local quadratic convergence requires accurate derivatives/state solves, a regular solution, and eventual unit steps.
 */
export function adjointNewtonExact(
  initialParameters: Float64Array, initialStates: Float64Array,
  objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  options: AdjointNewtonExactOptions = {}
): AdjointNewtonExactResult {
  return runAdjointNewton(initialParameters, initialStates, objective, constraint, options, 'exact');
}

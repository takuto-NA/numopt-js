/** Public implicit-residual derivative API without internal logging parameters. */

import type { Matrix } from 'ml-matrix';
import { computeEffectiveJacobian as computeInternalJacobian, type EffectiveJacobianOptions } from './effectiveJacobian.js';
import type { ConstrainedResidualFn, ConstraintFn } from './types.js';

export type { EffectiveJacobianOptions } from './effectiveJacobian.js';

/**
 * Compute J_eff = r_p - r_x C_x^+ C_p at caller-supplied feasible states.
 * For f=1/2 ||r||² the reduced gradient is J_eff^T r. Omitted partials use finite differences.
 * A true state derivative requires full column rank and compatible parameter derivatives;
 * otherwise the result is a least-squares/regularized linearization, not proof of x(p).
 * Regular square/tall systems share one LU/QR factorization across parameter columns.
 * Does not solve or project states. See the mechanical-character example for BFGS integration.
 */
export function computeEffectiveJacobian(
  parameters: Float64Array,
  states: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: EffectiveJacobianOptions = {}
): Matrix {
  return computeInternalJacobian(parameters, states, residualFunction, constraintFunction, options);
}

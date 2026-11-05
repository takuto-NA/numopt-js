/**
 * Full reduced Hessian Z^T L_zz Z with L=f-lambda^T c and Z=[I;dx/dp].
 * Constraint curvature includes the contracted second state sensitivities without storing a tensor.
 */

import { Matrix } from 'ml-matrix';
import type { AdjointNewtonExactOptions, ConstraintFn } from './types.js';
import type { ReducedGradientEvaluation } from './adjointReducedSpace.js';
import { checkedSymmetricMatrix, computeStateSensitivities, directionalHessian } from './adjointNewtonModel.js';
import { dotProduct } from '../utils/matrix.js';

const DEFAULT_HESSIAN_STEP_SIZE = 1e-4;

function tangentBasis(parameterCount: number, sensitivities: Matrix): Matrix {
  const basis = Matrix.zeros(parameterCount + sensitivities.rows, parameterCount);
  for (let column = 0; column < parameterCount; column++) {
    basis.set(column, column, 1);
    for (let row = 0; row < sensitivities.rows; row++) {
      basis.set(parameterCount + row, column, sensitivities.get(row, column));
    }
  }
  return basis;
}

function projectedAnalyticalHessian(
  parameters: Float64Array, states: Float64Array, lambda: Float64Array,
  basis: Matrix, options: AdjointNewtonExactOptions
): Matrix | undefined {
  const dimension = basis.rows;
  if (options.lagrangianHessian) {
    const joint = checkedSymmetricMatrix(options.lagrangianHessian(parameters, states, new Float64Array(lambda)),
      dimension, 'lagrangianHessian');
    return basis.transpose().mmul(joint).mmul(basis);
  }
  if (!options.lagrangianHessianVector) return undefined;
  const products = Matrix.zeros(dimension, basis.columns);
  for (let column = 0; column < basis.columns; column++) {
    const product = options.lagrangianHessianVector(parameters, states, new Float64Array(lambda),
      new Float64Array(basis.getColumn(column)));
    if (product.length !== dimension || !product.every(Number.isFinite)) {
      throw new Error(`lagrangianHessianVector must return ${dimension} finite entries in [p,x] order`);
    }
    products.setColumn(column, Array.from(product));
  }
  return basis.transpose().mmul(products);
}

export function computeExactAdjointHessian(
  parameters: Float64Array, states: Float64Array,
  cost: (parameters: Float64Array, states: Float64Array) => number, constraint: ConstraintFn,
  reduced: ReducedGradientEvaluation, options: AdjointNewtonExactOptions
): Matrix {
  if (options.lagrangianHessian && options.lagrangianHessianVector) {
    throw new Error('Supply only one of lagrangianHessian and lagrangianHessianVector');
  }
  const sensitivities = computeStateSensitivities(reduced.partials, options);
  const basis = tangentBasis(parameters.length, sensitivities);
  const analytical = projectedAnalyticalHessian(parameters, states, reduced.lambda, basis, options);
  if (analytical) return checkedSymmetricMatrix(analytical, parameters.length, 'Full reduced Hessian');

  const joint = new Float64Array(basis.rows);
  joint.set(parameters);
  joint.set(states, parameters.length);
  // WHY: Fix lambda and move along the tangent, without restoring states. The
  // Lagrangian constraint Hessian supplies precisely the state-curvature term.
  const lagrangian = (values: Float64Array): number => {
    const trialParameters = values.slice(0, parameters.length);
    const trialStates = values.slice(parameters.length);
    return cost(trialParameters, trialStates) - dotProduct(reduced.lambda, constraint(trialParameters, trialStates));
  };
  return directionalHessian(lagrangian, joint, basis, options.hessianStepSize ?? DEFAULT_HESSIAN_STEP_SIZE);
}

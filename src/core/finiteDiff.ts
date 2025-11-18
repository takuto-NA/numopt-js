/**
 * This file implements numerical differentiation methods for computing
 * gradients and Jacobian matrices when analytical derivatives are not available.
 * 
 * Role in system:
 * - Provides automatic gradient/Jacobian computation via finite differences
 * - Used when users don't provide analytical derivatives
 * - Critical for algorithms that require gradient information
 * 
 * For first-time readers:
 * - Start with finiteDiffGradient for general optimization
 * - finiteDiffJacobian is for nonlinear least squares problems
 * - Central difference is used for better accuracy than forward difference
 */

import { Matrix } from 'ml-matrix';
import type {
  CostFn,
  ResidualFn,
  NumericalDifferentiationOptions
} from './types';

const DEFAULT_STEP_SIZE = 1e-6;

/**
 * Computes the gradient vector using central difference method.
 * 
 * Central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 * 
 * This is more accurate than forward difference but requires two function
 * evaluations per parameter. The trade-off is worth it for better convergence.
 */
export function finiteDiffGradient(
  parameters: Float64Array,
  costFunction: CostFn,
  options: NumericalDifferentiationOptions = {}
): Float64Array {
  const stepSize = options.stepSize ?? DEFAULT_STEP_SIZE;
  const parameterCount = parameters.length;
  const gradient = new Float64Array(parameterCount);

  for (let i = 0; i < parameterCount; i++) {
    // Forward point: x + h
    const forwardParams = new Float64Array(parameters);
    forwardParams[i] += stepSize;
    const forwardCost = costFunction(forwardParams);

    // Backward point: x - h
    const backwardParams = new Float64Array(parameters);
    backwardParams[i] -= stepSize;
    const backwardCost = costFunction(backwardParams);

    // Central difference: (f(x+h) - f(x-h)) / (2h)
    gradient[i] = (forwardCost - backwardCost) / (2.0 * stepSize);
  }

  return gradient;
}

/**
 * Computes the Jacobian matrix using central difference method.
 * 
 * The Jacobian J has dimensions (residualCount × parameterCount) where:
 * - Each row corresponds to a residual component
 * - Each column corresponds to a parameter
 * - J[i][j] = ∂r_i / ∂p_j
 * 
 * Central difference is used for each partial derivative.
 */
export function finiteDiffJacobian(
  residualFunction: ResidualFn,
  parameters: Float64Array,
  options: NumericalDifferentiationOptions = {}
): Matrix {
  const stepSize = options.stepSize ?? DEFAULT_STEP_SIZE;
  const parameterCount = parameters.length;

  // Compute residual at current point to determine dimension
  const currentResidual = residualFunction(parameters);
  const residualCount = currentResidual.length;

  // Initialize Jacobian matrix (residualCount × parameterCount)
  const jacobianData: number[][] = [];
  for (let i = 0; i < residualCount; i++) {
    jacobianData.push(new Array(parameterCount).fill(0));
  }

  // Compute each column of the Jacobian (derivative w.r.t. each parameter)
  for (let paramIndex = 0; paramIndex < parameterCount; paramIndex++) {
    // Forward point: p + h * e_j (where e_j is unit vector)
    const forwardParams = new Float64Array(parameters);
    forwardParams[paramIndex] += stepSize;
    const forwardResidual = residualFunction(forwardParams);

    // Backward point: p - h * e_j
    const backwardParams = new Float64Array(parameters);
    backwardParams[paramIndex] -= stepSize;
    const backwardResidual = residualFunction(backwardParams);

    // Central difference for each residual component
    for (let residualIndex = 0; residualIndex < residualCount; residualIndex++) {
      const derivative = (forwardResidual[residualIndex] - backwardResidual[residualIndex]) / (2.0 * stepSize);
      jacobianData[residualIndex][paramIndex] = derivative;
    }
  }

  return new Matrix(jacobianData);
}


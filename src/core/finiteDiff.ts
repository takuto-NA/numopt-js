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
} from './types.js';

const DEFAULT_STEP_SIZE = 1e-6;
const CENTRAL_DIFFERENCE_DENOMINATOR = 2.0; // Denominator for central difference formula: (f(x+h) - f(x-h)) / (2h)

/**
 * Computes the gradient vector using central difference method.
 * 
 * Central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 * 
 * This is more accurate than forward difference but requires two function
 * evaluations per parameter. The trade-off is worth it for better convergence.
 * 
 * @param parameters - The point at which to evaluate the gradient
 * @param costFunction - The cost function to differentiate
 * @param options - Optional numerical differentiation settings
 * @returns The gradient vector at the given parameters
 * 
 * @example
 * ```typescript
 * // Standalone usage - compute gradient at a specific point
 * const costFn = (params) => params[0] ** 2 + params[1] ** 2;
 * const params = new Float64Array([1.0, 2.0]);
 * const gradient = finiteDiffGradient(params, costFn);
 * // gradient ≈ [2.0, 4.0]
 * ```
 * 
 * @example
 * ```typescript
 * // Usage with gradientDescent - note the parameter order!
 * import { gradientDescent, finiteDiffGradient } from 'numopt-js';
 * 
 * const costFn = (params) => Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
 * 
 * const result = gradientDescent(
 *   new Float64Array([0, 0]),
 *   costFn,
 *   (params) => finiteDiffGradient(params, costFn),  // ✅ Correct: params first!
 *   { maxIterations: 100, tolerance: 1e-6 }
 * );
 * ```
 * 
 * @example
 * ```typescript
 * // For easier usage with optimizers, consider using createFiniteDiffGradient:
 * import { gradientDescent, createFiniteDiffGradient } from 'numopt-js';
 * 
 * const costFn = (params) => Math.pow(params[0] - 3, 2) + Math.pow(params[1] - 2, 2);
 * const gradientFn = createFiniteDiffGradient(costFn);  // No parameter order confusion!
 * 
 * const result = gradientDescent(
 *   new Float64Array([0, 0]),
 *   costFn,
 *   gradientFn,
 *   { maxIterations: 100, tolerance: 1e-6 }
 * );
 * ```
 * 
 * @remarks
 * **Important:** When using with optimization algorithms, note the parameter order:
 * - ✅ Correct: `(params) => finiteDiffGradient(params, costFn)`
 * - ❌ Wrong: `(params) => finiteDiffGradient(costFn, params)`
 * 
 * Consider using {@link createFiniteDiffGradient} for a more intuitive API.
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
    gradient[i] = (forwardCost - backwardCost) / (CENTRAL_DIFFERENCE_DENOMINATOR * stepSize);
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
      const derivative = (forwardResidual[residualIndex] - backwardResidual[residualIndex]) / (CENTRAL_DIFFERENCE_DENOMINATOR * stepSize);
      jacobianData[residualIndex][paramIndex] = derivative;
    }
  }

  return new Matrix(jacobianData);
}


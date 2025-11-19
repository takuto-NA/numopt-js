/**
 * This file implements line search algorithms for determining optimal step sizes
 * in optimization algorithms. The implementation follows the backtracking
 * Armijo line search described in Nocedal & Wright, "Numerical Optimization"
 * (2nd ed.), Algorithm 3.1.
 *
 * Role in system:
 * - Used by gradient descent to find appropriate step sizes
 * - Implements backtracking line search with Armijo condition
 * - Critical for convergence of gradient-based methods
 *
 * For first-time readers:
 * - Start with backtrackingLineSearch function
 * - Understand Armijo condition (sufficient decrease)
 * - Line search prevents overshooting the minimum
 */

import type { CostFn, GradientFn, LineSearchOptions } from './types.js';
import { vectorNorm } from '../utils/matrix.js';

const DEFAULT_INITIAL_STEP_SIZE = 1.0;
const GRADIENT_NORM_THRESHOLD = 1e-10; // Threshold below which we use default step size to avoid numerical instability
// Typical values recommended in Nocedal & Wright (Algorithm 3.1) are β = 0.5 and c = 1e-4.
const DEFAULT_CONTRACTION_FACTOR = 0.5;
const DEFAULT_ARMIJO_PARAMETER = 1e-4;
const DEFAULT_MAX_LINE_SEARCH_ITERATIONS = 50;
const INVALID_STEP_SIZE = 0.0; // Returned when search direction is not a descent direction
const NON_DESCENT_DIRECTION_THRESHOLD = 0.0; // Threshold for directional derivative: >= 0 means not a descent direction

/**
 * Performs backtracking line search to find a step size that satisfies
 * the Armijo condition (sufficient decrease). This follows the textbook
 * backtracking scheme in Nocedal & Wright, "Numerical Optimization"
 * (2nd ed.), Algorithm 3.1.
 *
 * The Armijo condition ensures that the function value decreases sufficiently:
 * f(x + α * d) <= f(x) + c * α * ∇f(x)^T * d
 *
 * where:
 * - α is the step size
 * - d is the search direction (typically -gradient)
 * - c is the Armijo parameter (typically chosen around 1e-4 per Nocedal & Wright)
 * - β is the backtracking contraction factor (Algorithm 3.1 suggests β = 0.5)
 *
 * Initial step size selection:
 * - If `initialStepSize` is explicitly provided in options, it is used as-is.
 * - Otherwise, the initial step size is automatically scaled by the gradient norm:
 *   α₀ = 1.0 / ||∇f(x)||
 *   This prevents steps from being too large when gradients are large, improving
 *   convergence performance. If the gradient norm is very small (< 1e-10) or the
 *   computed step size is not finite, the default value of 1.0 is used.
 */
export function backtrackingLineSearch(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  options: LineSearchOptions = {}
): number {
  const contractionFactor = options.contractionFactor ?? DEFAULT_CONTRACTION_FACTOR;
  const armijoParameter = options.armijoParameter ?? DEFAULT_ARMIJO_PARAMETER;
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_LINE_SEARCH_ITERATIONS;

  const currentCost = costFunction(currentParameters);
  const currentGradient = gradientFunction(currentParameters);

  // Determine initial step size: use provided value, or scale by gradient norm if not provided
  let initialStepSize: number;
  if (options.initialStepSize !== undefined) {
    // Use explicitly provided initial step size (maintains backward compatibility)
    initialStepSize = options.initialStepSize;
  } else {
    // Scale initial step size by gradient norm: 1.0 / ||gradient||
    // This prevents steps from being too large when gradients are large
    const gradientNorm = vectorNorm(currentGradient);
    
    // Handle edge cases: very small or zero gradient norm
    if (gradientNorm < GRADIENT_NORM_THRESHOLD) {
      initialStepSize = DEFAULT_INITIAL_STEP_SIZE;
    } else {
      initialStepSize = 1.0 / gradientNorm;
      
      // Additional safety check for NaN or Infinity
      if (!isFinite(initialStepSize)) {
        initialStepSize = DEFAULT_INITIAL_STEP_SIZE;
      }
    }
  }

  // Compute directional derivative: ∇f(x)^T * d
  let directionalDerivative = 0.0;
  for (let i = 0; i < currentGradient.length; i++) {
    directionalDerivative += currentGradient[i] * searchDirection[i];
  }

  // Early return if search direction is not a descent direction
  // Directional derivative >= 0 means moving in this direction increases the cost
  if (directionalDerivative >= NON_DESCENT_DIRECTION_THRESHOLD) {
    return INVALID_STEP_SIZE;
  }

  let stepSize = initialStepSize;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Compute trial point: x + α * d
    const trialParameters = new Float64Array(currentParameters.length);
    for (let i = 0; i < currentParameters.length; i++) {
      trialParameters[i] = currentParameters[i] + stepSize * searchDirection[i];
    }

    const trialCost = costFunction(trialParameters);

    // Armijo condition: f(x + αd) <= f(x) + c * α * ∇f(x)^T * d
    const armijoThreshold = currentCost + armijoParameter * stepSize * directionalDerivative;

    if (trialCost <= armijoThreshold) {
      return stepSize;
    }

    // Reduce step size and try again
    stepSize *= contractionFactor;
  }

  // If we couldn't find a suitable step size, return a very small value
  return stepSize;
}


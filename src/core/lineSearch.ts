/**
 * This file implements line search algorithms for determining optimal step sizes
 * in optimization algorithms.
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

import type { CostFn, GradientFn, LineSearchOptions } from './types';

const DEFAULT_INITIAL_STEP_SIZE = 1.0;
const DEFAULT_CONTRACTION_FACTOR = 0.5;
const DEFAULT_ARMIJO_PARAMETER = 0.1;
const DEFAULT_MAX_LINE_SEARCH_ITERATIONS = 50;
const INVALID_STEP_SIZE = 0.0; // Returned when search direction is not a descent direction
const NON_DESCENT_DIRECTION_THRESHOLD = 0.0; // Threshold for directional derivative: >= 0 means not a descent direction

/**
 * Performs backtracking line search to find a step size that satisfies
 * the Armijo condition (sufficient decrease).
 * 
 * The Armijo condition ensures that the function value decreases sufficiently:
 * f(x + α * d) <= f(x) + c * α * ∇f(x)^T * d
 * 
 * where:
 * - α is the step size
 * - d is the search direction (typically -gradient)
 * - c is the Armijo parameter (typically 0.1)
 */
export function backtrackingLineSearch(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  options: LineSearchOptions = {}
): number {
  const initialStepSize = options.initialStepSize ?? DEFAULT_INITIAL_STEP_SIZE;
  const contractionFactor = options.contractionFactor ?? DEFAULT_CONTRACTION_FACTOR;
  const armijoParameter = options.armijoParameter ?? DEFAULT_ARMIJO_PARAMETER;
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_LINE_SEARCH_ITERATIONS;

  const currentCost = costFunction(currentParameters);
  const currentGradient = gradientFunction(currentParameters);
  
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


/**
 * This file implements line search algorithms for determining step sizes.
 *
 * Role in system:
 * - Provides step size selection for gradient-based optimizers
 * - Backtracking Armijo: simple and robust default (used by gradient descent)
 * - Strong Wolfe: preferred for quasi-Newton methods (BFGS / L-BFGS) to help satisfy
 *   the curvature condition \(s^T y > 0\), improving Hessian approximation stability
 *
 * For first-time readers:
 * - Start with `backtrackingLineSearch` (simpler)
 * - Then read `strongWolfeLineSearch` and `zoom` (more subtle but more powerful)
 */

import type { CostFn, GradientFn, LineSearchOptions, StrongWolfeLineSearchOptions } from './types.js';
import { dotProduct, vectorNorm } from '../utils/matrix.js';

const DEFAULT_INITIAL_STEP_SIZE = 1.0;
const GRADIENT_NORM_THRESHOLD = 1e-10; // Threshold below which we use default step size to avoid numerical instability
// Typical values recommended in Nocedal & Wright (Algorithm 3.1) are β = 0.5 and c = 1e-4.
const DEFAULT_CONTRACTION_FACTOR = 0.5;
const DEFAULT_ARMIJO_PARAMETER = 1e-4;
const DEFAULT_MAX_LINE_SEARCH_ITERATIONS = 50;
const INVALID_STEP_SIZE = 0.0; // Returned when search direction is not a descent direction
const NON_DESCENT_DIRECTION_THRESHOLD = 0.0; // Threshold for directional derivative: >= 0 means not a descent direction

// Typical Strong Wolfe defaults (Nocedal & Wright, 2nd ed.)
const DEFAULT_WOLFE_C1 = 1e-4;
const DEFAULT_WOLFE_C2 = 0.9;
const DEFAULT_MAX_STRONG_WOLFE_ITERATIONS = 25;
const DEFAULT_MAX_ZOOM_ITERATIONS = 25;
const DEFAULT_STEP_SIZE_GROWTH_FACTOR = 2.0;
const DEFAULT_STRONG_WOLFE_INITIAL_STEP_SIZE = 1.0;
const MINIMUM_STEP_SIZE = 1e-20; // Prevents infinite loops when step size underflows
const MAXIMUM_STEP_SIZE = 1e20; // Prevents overflow in x + alpha * p

type LineSearchPointEvaluation = {
  stepSize: number;
  cost: number;
  directionalDerivative: number;
};

function clampStepSize(stepSize: number): number {
  if (!isFinite(stepSize)) return DEFAULT_INITIAL_STEP_SIZE;
  if (stepSize < MINIMUM_STEP_SIZE) return MINIMUM_STEP_SIZE;
  if (stepSize > MAXIMUM_STEP_SIZE) return MAXIMUM_STEP_SIZE;
  return stepSize;
}

function computeTrialParameters(
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  stepSize: number
): Float64Array {
  const trialParameters = new Float64Array(currentParameters.length);
  for (let index = 0; index < currentParameters.length; index++) {
    trialParameters[index] = currentParameters[index] + stepSize * searchDirection[index];
  }
  return trialParameters;
}

function evaluateCostAndDirectionalDerivative(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  stepSize: number
): LineSearchPointEvaluation {
  const clampedStepSize = clampStepSize(stepSize);
  const trialParameters = computeTrialParameters(currentParameters, searchDirection, clampedStepSize);
  const trialCost = costFunction(trialParameters);
  const trialGradient = gradientFunction(trialParameters);
  const trialDirectionalDerivative = dotProduct(trialGradient, searchDirection);
  return { stepSize: clampedStepSize, cost: trialCost, directionalDerivative: trialDirectionalDerivative };
}

function determineInitialStepSize(
  providedInitialStepSize: number | undefined,
  currentGradient: Float64Array
): number {
  if (providedInitialStepSize !== undefined) {
    return clampStepSize(providedInitialStepSize);
  }

  // WHY: Strong Wolfe line search is typically started with α=1.0 for quasi-Newton methods.
  // Using 1/||g|| can be overly conservative and slow convergence in practice.
  // We keep the gradient-norm scaling behavior in backtrackingLineSearch (separate implementation).
  void currentGradient; // Kept in signature for symmetry; not used by design.
  return DEFAULT_STRONG_WOLFE_INITIAL_STEP_SIZE;
}

function satisfiesArmijoCondition(
  trialCost: number,
  currentCost: number,
  wolfeC1: number,
  stepSize: number,
  directionalDerivativeAtZero: number
): boolean {
  const armijoThreshold = currentCost + wolfeC1 * stepSize * directionalDerivativeAtZero;
  return trialCost <= armijoThreshold;
}

function satisfiesStrongWolfeCurvatureCondition(
  trialDirectionalDerivative: number,
  wolfeC2: number,
  directionalDerivativeAtZero: number
): boolean {
  const leftSide = Math.abs(trialDirectionalDerivative);
  const rightSide = wolfeC2 * Math.abs(directionalDerivativeAtZero);
  return leftSide <= rightSide;
}

function computeBisectionStepSize(stepSizeLow: number, stepSizeHigh: number): number {
  return 0.5 * (stepSizeLow + stepSizeHigh);
}

function zoom(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  currentCost: number,
  directionalDerivativeAtZero: number,
  wolfeC1: number,
  wolfeC2: number,
  stepSizeLowInitial: number,
  stepSizeHighInitial: number,
  costAtStepSizeLowInitial: number,
  maxZoomIterations: number
): number {
  let stepSizeLow = stepSizeLowInitial;
  let stepSizeHigh = stepSizeHighInitial;
  let costAtStepSizeLow = costAtStepSizeLowInitial;

  for (let zoomIteration = 0; zoomIteration < maxZoomIterations; zoomIteration++) {
    const trialStepSize = computeBisectionStepSize(stepSizeLow, stepSizeHigh);
    const evaluation = evaluateCostAndDirectionalDerivative(
      costFunction,
      gradientFunction,
      currentParameters,
      searchDirection,
      trialStepSize
    );

    if (!satisfiesArmijoCondition(evaluation.cost, currentCost, wolfeC1, evaluation.stepSize, directionalDerivativeAtZero)) {
      stepSizeHigh = evaluation.stepSize;
      continue;
    }

    if (evaluation.cost >= costAtStepSizeLow) {
      stepSizeHigh = evaluation.stepSize;
      continue;
    }

    if (satisfiesStrongWolfeCurvatureCondition(evaluation.directionalDerivative, wolfeC2, directionalDerivativeAtZero)) {
      return evaluation.stepSize;
    }

    const bracketWidth = stepSizeHigh - stepSizeLow;
    const shouldSwapBracketSide = evaluation.directionalDerivative * bracketWidth >= 0.0;
    if (shouldSwapBracketSide) {
      stepSizeHigh = stepSizeLow;
    }

    stepSizeLow = evaluation.stepSize;
    costAtStepSizeLow = evaluation.cost;
  }

  // If zoom fails to find a point satisfying Strong Wolfe, return the best-known lower bound.
  return clampStepSize(stepSizeLow);
}

/**
 * Strong Wolfe line search (Nocedal & Wright, 2nd ed., Algorithm 3.5).
 *
 * WHY: For quasi-Newton methods, satisfying the curvature condition improves the chance that
 * the update will maintain a stable approximation (e.g., positive definiteness).
 */
export function strongWolfeLineSearch(
  costFunction: CostFn,
  gradientFunction: GradientFn,
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  options: StrongWolfeLineSearchOptions = {}
): number {
  const wolfeC1 = options.wolfeC1 ?? DEFAULT_WOLFE_C1;
  const wolfeC2 = options.wolfeC2 ?? DEFAULT_WOLFE_C2;
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_STRONG_WOLFE_ITERATIONS;
  const maxZoomIterations = options.maxZoomIterations ?? DEFAULT_MAX_ZOOM_ITERATIONS;
  const stepSizeGrowthFactor = options.stepSizeGrowthFactor ?? DEFAULT_STEP_SIZE_GROWTH_FACTOR;

  const currentCost = costFunction(currentParameters);
  const currentGradient = gradientFunction(currentParameters);
  const directionalDerivativeAtZero = dotProduct(currentGradient, searchDirection);

  // Strong Wolfe requires a descent direction (phi'(0) < 0). Otherwise, a line search is ill-posed.
  if (directionalDerivativeAtZero >= NON_DESCENT_DIRECTION_THRESHOLD) {
    return INVALID_STEP_SIZE;
  }

  let previousStepSize = 0.0;
  let previousCost = currentCost;
  let stepSize = determineInitialStepSize(options.initialStepSize, currentGradient);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const evaluation = evaluateCostAndDirectionalDerivative(
      costFunction,
      gradientFunction,
      currentParameters,
      searchDirection,
      stepSize
    );

    const violatesArmijo = !satisfiesArmijoCondition(
      evaluation.cost,
      currentCost,
      wolfeC1,
      evaluation.stepSize,
      directionalDerivativeAtZero
    );
    const isNotImprovingEnough = iteration > 0 && evaluation.cost >= previousCost;
    if (violatesArmijo || isNotImprovingEnough) {
      return zoom(
        costFunction,
        gradientFunction,
        currentParameters,
        searchDirection,
        currentCost,
        directionalDerivativeAtZero,
        wolfeC1,
        wolfeC2,
        previousStepSize,
        evaluation.stepSize,
        previousCost,
        maxZoomIterations
      );
    }

    if (satisfiesStrongWolfeCurvatureCondition(evaluation.directionalDerivative, wolfeC2, directionalDerivativeAtZero)) {
      return evaluation.stepSize;
    }

    if (evaluation.directionalDerivative >= 0.0) {
      const stepSizeLow = Math.min(previousStepSize, evaluation.stepSize);
      const stepSizeHigh = Math.max(previousStepSize, evaluation.stepSize);
      const costAtStepSizeLow = stepSizeLow === previousStepSize ? previousCost : evaluation.cost;
      return zoom(
        costFunction,
        gradientFunction,
        currentParameters,
        searchDirection,
        currentCost,
        directionalDerivativeAtZero,
        wolfeC1,
        wolfeC2,
        stepSizeLow,
        stepSizeHigh,
        costAtStepSizeLow,
        maxZoomIterations
      );
    }

    previousStepSize = evaluation.stepSize;
    previousCost = evaluation.cost;
    stepSize = clampStepSize(evaluation.stepSize * stepSizeGrowthFactor);
  }

  // If we couldn't satisfy Strong Wolfe within the iteration limit, return the last tried step size.
  return clampStepSize(stepSize);
}

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
  const directionalDerivative = dotProduct(currentGradient, searchDirection);

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


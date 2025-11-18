/**
 * This file implements the gradient descent optimization algorithm.
 * 
 * Role in system:
 * - Phase 1 foundation algorithm (simple, testable)
 * - Establishes basic optimization framework
 * - Used as building block for more advanced methods
 * 
 * For first-time readers:
 * - Start with gradientDescent function
 * - Understand how it uses line search or fixed step size
 * - Check convergence criteria implementation
 */

import type {
  CostFn,
  GradientFn,
  GradientDescentOptions,
  GradientDescentResult
} from './types';
import { backtrackingLineSearch } from './lineSearch';
import { vectorNorm, scaleVector, addVectors } from '../utils/matrix';
import { checkGradientConvergence, checkStepSizeConvergence, createConvergenceResult } from './convergence';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 0.01;
const DEFAULT_USE_LINE_SEARCH = true;
const ZERO_STEP_SIZE = 0.0; // Indicates line search found no valid step (not a descent direction)

/**
 * Determines the step size for gradient descent iteration.
 * Uses line search if enabled, otherwise uses fixed step size.
 * Returns the step size and whether line search was used.
 */
function determineStepSize(
  currentGradient: Float64Array,
  currentParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  useLineSearch: boolean,
  fixedStepSize: number | undefined
): { stepSize: number; usedLineSearch: boolean } {
  // Early return for fixed step size case
  if (!useLineSearch || fixedStepSize !== undefined) {
    return { stepSize: fixedStepSize ?? DEFAULT_STEP_SIZE, usedLineSearch: false };
  }

  // Use line search when enabled and no fixed step size provided
  const searchDirection = scaleVector(currentGradient, -1.0);
  const stepSize = backtrackingLineSearch(
    costFunction,
    gradientFunction,
    currentParameters,
    searchDirection
  );
  return { stepSize, usedLineSearch: true };
}

/**
 * Updates parameters by taking a step in the negative gradient direction.
 * Returns the new parameters and the step vector.
 */
function updateParametersWithGradientStep(
  currentParameters: Float64Array,
  currentGradient: Float64Array,
  stepSize: number
): { newParameters: Float64Array; step: Float64Array } {
  const step = scaleVector(currentGradient, -stepSize);
  const newParameters = addVectors(currentParameters, step);
  return { newParameters, step };
}

/**
 * Performs a single gradient descent iteration.
 * Returns the updated state or a convergence result if converged.
 */
function performGradientDescentIteration(
  iteration: number,
  currentParameters: Float64Array,
  currentCost: number,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  tolerance: number,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  onIteration: ((iteration: number, cost: number, parameters: Float64Array) => void) | undefined,
  verbose: boolean,
  usedLineSearchFlag: boolean
): { converged: boolean; result?: GradientDescentResult; newParameters?: Float64Array; newCost?: number; newUsedLineSearch?: boolean } {
  const currentGradient = gradientFunction(currentParameters);
  const gradientNorm = vectorNorm(currentGradient);

  // Handle callback (different behavior for first iteration)
  if (onIteration) {
    const callbackIteration = iteration === 0 ? 0 : iteration;
    onIteration(callbackIteration, currentCost, currentParameters);
  }

  // Check gradient convergence - early return
  if (checkGradientConvergence(gradientNorm, tolerance, iteration)) {
    if (verbose) {
      console.log(`Converged at iteration ${iteration}: gradient norm = ${gradientNorm}`);
    }
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return { converged: true, result: { ...result, usedLineSearch: usedLineSearchFlag } };
  }

  // Determine step size
  const stepSizeResult = determineStepSize(
    currentGradient,
    currentParameters,
    costFunction,
    gradientFunction,
    useLineSearch,
    fixedStepSize
  );

  // Early return: line search failed
  if (stepSizeResult.stepSize === ZERO_STEP_SIZE) {
    if (verbose) {
      console.log(`Line search failed at iteration ${iteration}`);
    }
    return {
      converged: true,
      result: {
        parameters: currentParameters,
        iterations: iteration,
        converged: false,
        finalCost: currentCost,
        finalGradientNorm: gradientNorm,
        usedLineSearch: true
      }
    };
  }

  const newUsedLineSearch = usedLineSearchFlag || stepSizeResult.usedLineSearch;

  // Update parameters
  const { newParameters, step } = updateParametersWithGradientStep(
    currentParameters,
    currentGradient,
    stepSizeResult.stepSize
  );
  const newCost = costFunction(newParameters);

  // Check step size convergence - early return
  const stepNorm = vectorNorm(step);
  if (checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
    if (verbose) {
      console.log(`Converged at iteration ${iteration}: step size = ${stepNorm}`);
    }
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return { converged: true, result: { ...result, usedLineSearch: newUsedLineSearch } };
  }

  // Log progress if verbose
  if (verbose) {
    console.log(
      `Iteration ${iteration}: cost = ${currentCost}, gradient norm = ${gradientNorm}, step size = ${stepSizeResult.stepSize}`
    );
  }

  return { converged: false, newParameters, newCost, newUsedLineSearch };
}

/**
 * Performs gradient descent optimization to minimize a cost function.
 * 
 * Algorithm:
 * 1. Start with initial parameters
 * 2. Compute gradient at current point
 * 3. Move in negative gradient direction (steepest descent)
 * 4. Use line search or fixed step size to determine step
 * 5. Repeat until convergence or max iterations
 * 
 * Convergence criteria:
 * - Gradient norm < tolerance
 * - Step size < tolerance
 * - Maximum iterations reached
 */
export function gradientDescent(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options: GradientDescentOptions = {}
): GradientDescentResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const stepSize = options.stepSize;
  const useLineSearch = options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH;
  const onIteration = options.onIteration;
  const verbose = options.verbose ?? false;

  let currentParameters = new Float64Array(initialParameters);
  let currentCost = costFunction(currentParameters);
  let usedLineSearchFlag = false;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const iterationResult = performGradientDescentIteration(
      iteration,
      currentParameters,
      currentCost,
      costFunction,
      gradientFunction,
      tolerance,
      useLineSearch,
      stepSize,
      onIteration,
      verbose,
      usedLineSearchFlag
    );

    if (iterationResult.converged && iterationResult.result) {
      return iterationResult.result;
    }

    if (iterationResult.newParameters && iterationResult.newCost !== undefined) {
      currentParameters = new Float64Array(iterationResult.newParameters);
      currentCost = iterationResult.newCost;
      if (iterationResult.newUsedLineSearch !== undefined) {
        usedLineSearchFlag = iterationResult.newUsedLineSearch;
      }
    }
  }

  // Maximum iterations reached
  const finalGradient = gradientFunction(currentParameters);
  const finalGradientNorm = vectorNorm(finalGradient);

  if (verbose) {
    console.log(`Maximum iterations reached: ${maxIterations}`);
  }

  return {
    parameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: currentCost,
    finalGradientNorm: finalGradientNorm,
    usedLineSearch: usedLineSearchFlag
  };
}


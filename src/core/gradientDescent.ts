/**
 * This file implements the gradient descent optimization algorithm.
 *
 * References:
 * - Nocedal & Wright, "Numerical Optimization" (2nd ed.), Chapter 2 (steepest descent and line search basics)
 * - Boyd & Vandenberghe, "Convex Optimization", Section 9.3 (backtracking line search with Armijo rule)
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
} from './types.js';
import { backtrackingLineSearch } from './lineSearch.js';
import { vectorNorm, scaleVector, addVectors } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence, createConvergenceResult } from './convergence.js';
import { Logger } from './logger.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 0.01;
const DEFAULT_USE_LINE_SEARCH = true;
const ZERO_STEP_SIZE = 0.0; // Indicates line search found no valid step (not a descent direction)
const NEGATIVE_GRADIENT_DIRECTION = -1.0; // Multiplier for negative gradient direction (steepest descent)

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
  const searchDirection = scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
  // Backtracking line search with Armijo condition (Boyd & Vandenberghe, Sec. 9.3)
  // to choose a step satisfying sufficient decrease (Nocedal & Wright, Ch. 2)
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
  const negativeStepSize = NEGATIVE_GRADIENT_DIRECTION * stepSize;
  const step = scaleVector(currentGradient, negativeStepSize);
  const newParameters = addVectors(currentParameters, step);
  return { newParameters, step };
}

/**
 * Checks gradient convergence and returns result if converged.
 * Early return pattern to reduce nesting.
 */
function checkGradientConvergenceAndReturn(
  currentParameters: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  tolerance: number,
  usedLineSearchFlag: boolean,
  logger: Logger
): { converged: boolean; result?: GradientDescentResult } {
  if (checkGradientConvergence(gradientNorm, tolerance, iteration)) {
    logger.info('gradientDescent', iteration, 'Converged', [
      { key: 'Cost:', value: currentCost },
      { key: 'Gradient norm:', value: gradientNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return { converged: true, result: { ...result, usedLineSearch: usedLineSearchFlag } };
  }
  return { converged: false };
}

/**
 * Handles line search failure case.
 * Returns convergence result indicating failure.
 */
function handleLineSearchFailure(
  currentParameters: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  logger: Logger
): { converged: boolean; result: GradientDescentResult } {
  logger.warn('gradientDescent', iteration, 'Line search failed', [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm }
  ]);
  return {
    converged: true,
    result: {
      finalParameters: currentParameters,
      iterations: iteration,
      converged: false,
      finalCost: currentCost,
      finalGradientNorm: gradientNorm,
      usedLineSearch: true
    }
  };
}

/**
 * Checks step size convergence and returns result if converged.
 * Early return pattern to reduce nesting.
 */
function checkStepSizeConvergenceAndReturn(
  currentParameters: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  stepNorm: number,
  tolerance: number,
  newUsedLineSearch: boolean,
  logger: Logger
): { converged: boolean; result?: GradientDescentResult } {
  if (checkStepSizeConvergence(stepNorm, tolerance, iteration)) {
    logger.info('gradientDescent', iteration, 'Converged', [
      { key: 'Cost:', value: currentCost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Step size:', value: stepNorm }
    ]);
    const result = createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    return { converged: true, result: { ...result, usedLineSearch: newUsedLineSearch } };
  }
  return { converged: false };
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
  logger: Logger,
  usedLineSearchFlag: boolean
): { converged: boolean; result?: GradientDescentResult; newParameters?: Float64Array; newCost?: number; newUsedLineSearch?: boolean } {
  const currentGradient = gradientFunction(currentParameters);
  const gradientNorm = vectorNorm(currentGradient); // Uses Euclidean norm for steepest descent direction (Nocedal & Wright, Ch. 2)

  // Handle callback (different behavior for first iteration)
  if (onIteration) {
    const callbackIteration = iteration === 0 ? 0 : iteration;
    onIteration(callbackIteration, currentCost, currentParameters);
  }

  // Check gradient convergence - early return
  const gradientConvergenceResult = checkGradientConvergenceAndReturn(
    currentParameters,
    iteration,
    currentCost,
    gradientNorm,
    tolerance,
    usedLineSearchFlag,
    logger
  );
  if (gradientConvergenceResult.converged && gradientConvergenceResult.result) {
    return { converged: true, result: gradientConvergenceResult.result };
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
    const failureResult = handleLineSearchFailure(
      currentParameters,
      iteration,
      currentCost,
      gradientNorm,
      logger
    );
    return failureResult;
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
  const stepNorm = vectorNorm(step); // Step length via 2-norm for step-size convergence (Boyd & Vandenberghe, Sec. 9.3)
  const stepSizeConvergenceResult = checkStepSizeConvergenceAndReturn(
    currentParameters,
    iteration,
    currentCost,
    gradientNorm,
    stepNorm,
    tolerance,
    newUsedLineSearch,
    logger
  );
  if (stepSizeConvergenceResult.converged && stepSizeConvergenceResult.result) {
    return { converged: true, result: stepSizeConvergenceResult.result };
  }

  // Log progress
  logger.debug('gradientDescent', iteration, 'Progress', [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm },
    { key: 'Step size:', value: stepSizeResult.stepSize }
  ]);

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
  const logger = new Logger(options.logLevel, options.verbose);

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
      logger,
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

  logger.warn('gradientDescent', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: currentCost },
    { key: 'Final gradient norm:', value: finalGradientNorm }
  ]);

  return {
    finalParameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: currentCost,
    finalGradientNorm: finalGradientNorm,
    usedLineSearch: usedLineSearchFlag
  };
}


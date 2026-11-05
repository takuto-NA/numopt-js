/**
 * Limited-memory BFGS for unconstrained smooth optimization.
 * Uses Strong Wolfe and two-loop recursion.
 * Entry point: `lbfgs`.
 */

import type { CostFn, GradientFn, OptimizationResult, LbfgsOptions } from './types.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { checkGradientConvergence, createConvergenceResult } from './convergence.js';
import { addVectors, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';

import { createEmptyHistory, computeLbfgsSearchDirection, ensureDescentDirectionOrFallback, updateHistoryIfCurvatureIsValid } from './lbfgsUpdate.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_HISTORY_SIZE = 10;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_FIXED_STEP_SIZE = 1.0;
const INVALID_STEP_SIZE = 0.0;




function computeNextParameters(
  currentParameters: Float64Array,
  searchDirection: Float64Array,
  stepSize: number
): Float64Array {
  const stepVector = scaleVector(searchDirection, stepSize);
  return addVectors(currentParameters, stepVector);
}

function handleLineSearchFailure(
  currentParameters: Float64Array,
  iteration: number,
  currentCost: number,
  gradientNorm: number,
  logger: Logger
): OptimizationResult {
  logger.warn('lbfgs', iteration, 'Line search failed (non-descent direction).', [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm }
  ]);
  return {
    finalParameters: currentParameters,
    parameters: currentParameters,
    iterations: iteration + 1,
    converged: false,
    finalCost: currentCost,
    finalGradientNorm: gradientNorm
  };
}

export function lbfgs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options: LbfgsOptions = {}
): OptimizationResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const historySize = options.historySize ?? DEFAULT_HISTORY_SIZE;
  if (!Number.isInteger(historySize) || historySize <= 0) throw new Error('historySize must be a positive integer');
  const useLineSearch = options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH;
  const fixedStepSize = options.stepSize ?? DEFAULT_FIXED_STEP_SIZE;
  const onIteration = options.onIteration;
  const logger = new Logger(options.logLevel, options.verbose);

  let currentParameters = new Float64Array(initialParameters);
  let currentCost = costFunction(currentParameters);
  const history = createEmptyHistory();

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const currentGradient = gradientFunction(currentParameters);
    const gradientNorm = vectorNorm(currentGradient);

    if (onIteration) onIteration(iteration, currentCost, currentParameters);

    if (checkGradientConvergence(gradientNorm, tolerance, iteration)) {
      logger.info('lbfgs', iteration, 'Converged', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm }
      ]);
      return createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    }

    const proposedSearchDirection = computeLbfgsSearchDirection(currentGradient, history);
    const searchDirection = ensureDescentDirectionOrFallback(
      currentGradient,
      proposedSearchDirection,
      history,
      logger,
      iteration,
      currentCost
    );

    const stepSize = useLineSearch
      ? strongWolfeLineSearch(costFunction, gradientFunction, currentParameters, searchDirection, options.lineSearchOptions)
      : fixedStepSize;

    if (stepSize === INVALID_STEP_SIZE) {
      return handleLineSearchFailure(currentParameters, iteration, currentCost, gradientNorm, logger);
    }

    const newParameters = computeNextParameters(currentParameters, searchDirection, stepSize);
    const stepVector = subtractVectors(newParameters, currentParameters);
    const stepNorm = vectorNorm(stepVector);

    const newCost = costFunction(newParameters);
    const newGradient = gradientFunction(newParameters);
    const gradientChangeVector = subtractVectors(newGradient, currentGradient);

    updateHistoryIfCurvatureIsValid(history, historySize, stepVector, gradientChangeVector, logger, iteration, newCost);

    logger.debug('lbfgs', iteration, 'Progress', [
      { key: 'Cost:', value: currentCost },
      { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Step size:', value: stepSize },
      { key: 'Step norm:', value: stepNorm }
    ]);

    currentParameters = new Float64Array(newParameters);
    currentCost = newCost;
  }

  const finalGradient = gradientFunction(currentParameters);
  const finalGradientNorm = vectorNorm(finalGradient);

  logger.warn('lbfgs', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'Final cost:', value: currentCost },
    { key: 'Final gradient norm:', value: finalGradientNorm }
  ]);

  return {
    finalParameters: currentParameters,
    parameters: currentParameters,
    iterations: maxIterations,
    converged: false,
    finalCost: currentCost,
    finalGradientNorm: finalGradientNorm
  };
}


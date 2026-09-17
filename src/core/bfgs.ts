/**
 * Dense BFGS for unconstrained smooth optimization.
 * Uses Strong Wolfe line search and a full inverse Hessian approximation.
 * Entry point: `bfgs`.
 */

import type { BfgsOptions, CostFn, GradientFn, OptimizationResult } from './types.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { checkGradientConvergence, createConvergenceResult } from './convergence.js';
import { addVectors, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';
import {
  computeBfgsSearchDirection,
  createIdentityInverseHessian,
  ensureDescentDirectionOrFallback,
  updateInverseHessianApproximation
} from './bfgsUpdate.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_FIXED_STEP_SIZE = 1.0;
const INVALID_STEP_SIZE = 0.0;
const BFGS_ALGORITHM_NAME = 'bfgs';

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
  logger.warn('bfgs', iteration, 'Line search failed (non-descent direction).', [
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

export function bfgs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  gradientFunction: GradientFn,
  options: BfgsOptions = {}
): OptimizationResult {
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const useLineSearch = options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH;
  const fixedStepSize = options.stepSize ?? DEFAULT_FIXED_STEP_SIZE;
  const onIteration = options.onIteration;
  const logger = new Logger(options.logLevel, options.verbose);

  let currentParameters = new Float64Array(initialParameters);
  let currentCost = costFunction(currentParameters);
  let inverseHessianApproximation = createIdentityInverseHessian(currentParameters.length);

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const currentGradient = gradientFunction(currentParameters);
    const gradientNorm = vectorNorm(currentGradient);

    if (onIteration) onIteration(iteration, currentCost, currentParameters);

    if (checkGradientConvergence(gradientNorm, tolerance, iteration)) {
      logger.info('bfgs', iteration, 'Converged', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm }
      ]);
      return createConvergenceResult(currentParameters, iteration, true, currentCost, gradientNorm);
    }

    const proposedSearchDirection = computeBfgsSearchDirection(inverseHessianApproximation, currentGradient);
    const descentResult = ensureDescentDirectionOrFallback(
      currentGradient,
      proposedSearchDirection,
      inverseHessianApproximation,
      logger,
      iteration,
      currentCost,
      BFGS_ALGORITHM_NAME
    );
    const searchDirection = descentResult.searchDirection;
    inverseHessianApproximation = descentResult.inverseHessianApproximation;

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

    inverseHessianApproximation = updateInverseHessianApproximation(
      inverseHessianApproximation,
      stepVector,
      gradientChangeVector,
      logger,
      iteration,
      newCost,
      BFGS_ALGORITHM_NAME
    );

    logger.debug('bfgs', iteration, 'Progress', [
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

  logger.warn('bfgs', undefined, 'Maximum iterations reached', [
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


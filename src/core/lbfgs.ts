/**
 * This file implements the L-BFGS (Limited-memory BFGS) algorithm for unconstrained
 * smooth optimization.
 *
 * Role in system:
 * - Quasi-Newton optimizer for scalar cost functions with user-provided gradients
 * - Uses Strong Wolfe line search to obtain steps that typically satisfy curvature conditions
 * - Memory-efficient alternative to dense BFGS for medium/large parameter counts
 *
 * For first-time readers:
 * - Start with `lbfgs` (main entry point)
 * - Then read `computeLbfgsSearchDirection` (two-loop recursion)
 * - Finally, check safeguard helpers (descent direction / curvature checks)
 */

import type { CostFn, GradientFn, OptimizationResult, LbfgsOptions } from './types.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { checkGradientConvergence, createConvergenceResult } from './convergence.js';
import { addVectors, dotProduct, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_HISTORY_SIZE = 10;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_FIXED_STEP_SIZE = 1.0;
const INVALID_STEP_SIZE = 0.0;
const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const MINIMUM_CURVATURE_THRESHOLD = 1e-10;
const DEFAULT_INITIAL_SCALING_FACTOR = 1.0;

type LbfgsHistory = {
  stepVectorHistory: Float64Array[];
  gradientChangeVectorHistory: Float64Array[];
  reciprocalCurvatureHistory: number[];
};

function createEmptyHistory(): LbfgsHistory {
  return { stepVectorHistory: [], gradientChangeVectorHistory: [], reciprocalCurvatureHistory: [] };
}

function clearHistory(history: LbfgsHistory): void {
  history.stepVectorHistory.length = 0;
  history.gradientChangeVectorHistory.length = 0;
  history.reciprocalCurvatureHistory.length = 0;
}

function computeInitialScalingFactor(history: LbfgsHistory): number {
  const historyLength = history.stepVectorHistory.length;
  if (historyLength === 0) return DEFAULT_INITIAL_SCALING_FACTOR;

  const lastIndex = historyLength - 1;
  const lastStepVector = history.stepVectorHistory[lastIndex];
  const lastGradientChangeVector = history.gradientChangeVectorHistory[lastIndex];

  const stepDotGradientChange = dotProduct(lastStepVector, lastGradientChangeVector);
  const gradientChangeDotGradientChange = dotProduct(lastGradientChangeVector, lastGradientChangeVector);

  if (stepDotGradientChange <= 0.0) return DEFAULT_INITIAL_SCALING_FACTOR;
  if (gradientChangeDotGradientChange <= 0.0) return DEFAULT_INITIAL_SCALING_FACTOR;

  const scalingFactor = stepDotGradientChange / gradientChangeDotGradientChange;
  if (!isFinite(scalingFactor) || scalingFactor <= 0.0) return DEFAULT_INITIAL_SCALING_FACTOR;
  return scalingFactor;
}

function computeLbfgsSearchDirection(currentGradient: Float64Array, history: LbfgsHistory): Float64Array {
  const historyLength = history.stepVectorHistory.length;
  if (historyLength === 0) {
    return scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
  }

  // NOTE: Avoid `new Float64Array(existingFloat64Array)` because TS can infer
  // `ArrayBufferLike` for the resulting buffer, which conflicts with stricter lib types.
  const qVectorInitial = new Float64Array(currentGradient.length);
  qVectorInitial.set(currentGradient);
  let qVector: Float64Array = qVectorInitial;
  const alphaCoefficients = new Array<number>(historyLength);

  for (let index = historyLength - 1; index >= 0; index--) {
    const stepVector = history.stepVectorHistory[index];
    const reciprocalCurvature = history.reciprocalCurvatureHistory[index];
    const alphaCoefficient = reciprocalCurvature * dotProduct(stepVector, qVector);
    alphaCoefficients[index] = alphaCoefficient;
    qVector = subtractVectors(qVector, scaleVector(history.gradientChangeVectorHistory[index], alphaCoefficient));
  }

  const initialScalingFactor = computeInitialScalingFactor(history);
  let rVector = scaleVector(qVector, initialScalingFactor);

  for (let index = 0; index < historyLength; index++) {
    const gradientChangeVector = history.gradientChangeVectorHistory[index];
    const reciprocalCurvature = history.reciprocalCurvatureHistory[index];
    const betaCoefficient = reciprocalCurvature * dotProduct(gradientChangeVector, rVector);
    const correctionCoefficient = alphaCoefficients[index] - betaCoefficient;
    rVector = addVectors(rVector, scaleVector(history.stepVectorHistory[index], correctionCoefficient));
  }

  return scaleVector(rVector, NEGATIVE_GRADIENT_DIRECTION);
}

function ensureDescentDirectionOrFallback(
  currentGradient: Float64Array,
  proposedSearchDirection: Float64Array,
  history: LbfgsHistory,
  logger: Logger,
  iteration: number,
  currentCost: number
): Float64Array {
  const directionalDerivative = dotProduct(currentGradient, proposedSearchDirection);
  const isDescentDirection = directionalDerivative < 0.0;
  if (isDescentDirection) return proposedSearchDirection;

  // WHY: If numerical issues break descent, fall back to steepest descent and clear history.
  clearHistory(history);
  logger.warn('lbfgs', iteration, 'Non-descent direction detected; falling back to negative gradient and clearing history.', [
    { key: 'Cost:', value: currentCost },
    { key: 'Directional derivative:', value: directionalDerivative }
  ]);
  return scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
}

function updateHistoryIfCurvatureIsValid(
  history: LbfgsHistory,
  historySize: number,
  stepVector: Float64Array,
  gradientChangeVector: Float64Array,
  logger: Logger,
  iteration: number,
  currentCost: number
): void {
  const stepDotGradientChange = dotProduct(stepVector, gradientChangeVector);
  const curvatureIsTooWeak = stepDotGradientChange <= MINIMUM_CURVATURE_THRESHOLD;
  if (curvatureIsTooWeak) {
    // WHY: Weak/negative curvature can destabilize updates; clearing history is the simplest safe recovery.
    clearHistory(history);
    logger.warn('lbfgs', iteration, 'Curvature condition too weak; clearing history to regain robustness.', [
      { key: 'Cost:', value: currentCost },
      { key: 'stepDotGradientChange:', value: stepDotGradientChange }
    ]);
    return;
  }

  const reciprocalCurvature = 1.0 / stepDotGradientChange;
  history.stepVectorHistory.push(stepVector);
  history.gradientChangeVectorHistory.push(gradientChangeVector);
  history.reciprocalCurvatureHistory.push(reciprocalCurvature);

  while (history.stepVectorHistory.length > historySize) {
    history.stepVectorHistory.shift();
    history.gradientChangeVectorHistory.shift();
    history.reciprocalCurvatureHistory.shift();
  }
}

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


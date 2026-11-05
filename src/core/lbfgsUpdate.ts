/**
 * Shared limited-memory correction pairs, two-loop recursion, and descent safeguards.
 * Stores vectors only; neither solver constructs a dense inverse Hessian.
 */

import { dotProduct, scaleVector } from '../utils/matrix.js';
import { Logger } from './logger.js';

const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const MINIMUM_CURVATURE_THRESHOLD = 1e-10;
const DEFAULT_INITIAL_SCALING_FACTOR = 1.0;

export type LbfgsHistory = {
  stepVectorHistory: Float64Array[];
  gradientChangeVectorHistory: Float64Array[];
  reciprocalCurvatureHistory: number[];
};

export function createEmptyHistory(): LbfgsHistory {
  return { stepVectorHistory: [], gradientChangeVectorHistory: [], reciprocalCurvatureHistory: [] };
}

export function clearHistory(history: LbfgsHistory): void {
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

function addScaledVectorInPlace(target: Float64Array, source: Float64Array, factor: number): void {
  for (let index = 0; index < target.length; index++) target[index] += factor * source[index];
}

export function computeLbfgsSearchDirection(currentGradient: Float64Array, history: LbfgsHistory): Float64Array {
  const historyLength = history.stepVectorHistory.length;
  if (historyLength === 0) {
    return scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
  }

  // NOTE: Avoid `new Float64Array(existingFloat64Array)` because TS can infer
  // `ArrayBufferLike` for the resulting buffer, which conflicts with stricter lib types.
  const qVectorInitial = new Float64Array(currentGradient.length);
  qVectorInitial.set(currentGradient);
  const qVector: Float64Array = qVectorInitial;
  const alphaCoefficients = new Array<number>(historyLength);

  for (let index = historyLength - 1; index >= 0; index--) {
    const stepVector = history.stepVectorHistory[index];
    const reciprocalCurvature = history.reciprocalCurvatureHistory[index];
    const alphaCoefficient = reciprocalCurvature * dotProduct(stepVector, qVector);
    alphaCoefficients[index] = alphaCoefficient;
    addScaledVectorInPlace(qVector, history.gradientChangeVectorHistory[index], -alphaCoefficient);
  }

  const initialScalingFactor = computeInitialScalingFactor(history);
  const rVector = scaleVector(qVector, initialScalingFactor);

  for (let index = 0; index < historyLength; index++) {
    const gradientChangeVector = history.gradientChangeVectorHistory[index];
    const reciprocalCurvature = history.reciprocalCurvatureHistory[index];
    const betaCoefficient = reciprocalCurvature * dotProduct(gradientChangeVector, rVector);
    const correctionCoefficient = alphaCoefficients[index] - betaCoefficient;
    addScaledVectorInPlace(rVector, history.stepVectorHistory[index], correctionCoefficient);
  }

  return scaleVector(rVector, NEGATIVE_GRADIENT_DIRECTION);
}

export function ensureDescentDirectionOrFallback(
  currentGradient: Float64Array,
  proposedSearchDirection: Float64Array,
  history: LbfgsHistory,
  logger: Logger,
  iteration: number,
  currentCost: number,
  algorithmName = 'lbfgs'
): Float64Array {
  const directionalDerivative = dotProduct(currentGradient, proposedSearchDirection);
  const isDescentDirection = Number.isFinite(directionalDerivative) && directionalDerivative < 0.0;
  if (isDescentDirection) return proposedSearchDirection;

  // WHY: If numerical issues break descent, fall back to steepest descent and clear history.
  clearHistory(history);
  logger.warn(algorithmName, iteration, 'Non-descent direction detected; falling back to negative gradient and clearing history.', [
    { key: 'Cost:', value: currentCost },
    { key: 'Directional derivative:', value: directionalDerivative }
  ]);
  return scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
}

export function updateHistoryIfCurvatureIsValid(
  history: LbfgsHistory,
  historySize: number,
  stepVector: Float64Array,
  gradientChangeVector: Float64Array,
  logger: Logger,
  iteration: number,
  currentCost: number,
  algorithmName = 'lbfgs',
  minimumCurvature = MINIMUM_CURVATURE_THRESHOLD
): void {
  const stepDotGradientChange = dotProduct(stepVector, gradientChangeVector);
  const curvatureIsTooWeak = !Number.isFinite(stepDotGradientChange) || stepDotGradientChange <= minimumCurvature ||
    !Number.isFinite(1 / stepDotGradientChange);
  if (curvatureIsTooWeak) {
    // WHY: Weak/negative curvature can destabilize updates; clearing history is the simplest safe recovery.
    clearHistory(history);
    logger.warn(algorithmName, iteration, 'Curvature condition too weak; clearing history to regain robustness.', [
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

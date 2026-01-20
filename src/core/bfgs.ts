/**
 * This file implements the (dense) BFGS algorithm for unconstrained smooth optimization.
 *
 * Role in system:
 * - Quasi-Newton optimizer for scalar cost functions with user-provided gradients
 * - Uses Strong Wolfe line search to encourage curvature conditions needed for stable updates
 * - Dense method: stores a full inverse Hessian approximation (O(n^2) memory)
 *
 * For first-time readers:
 * - Start with `bfgs` (main entry point)
 * - Then read `updateInverseHessianApproximation` (core BFGS update)
 * - Finally, check safeguard helpers (descent direction / curvature checks)
 */

import { Matrix } from 'ml-matrix';
import type { BfgsOptions, CostFn, GradientFn, OptimizationResult } from './types.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { checkGradientConvergence, createConvergenceResult } from './convergence.js';
import { addVectors, dotProduct, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_FIXED_STEP_SIZE = 1.0;
const INVALID_STEP_SIZE = 0.0;
const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const MINIMUM_CURVATURE_THRESHOLD = 1e-10;

function createIdentityMatrix(dimension: number): Matrix {
  return Matrix.eye(dimension);
}

function multiplyMatrixVector(matrix: Matrix, vector: Float64Array): Float64Array {
  const result = new Float64Array(vector.length);
  for (let rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
    let sum = 0.0;
    for (let columnIndex = 0; columnIndex < matrix.columns; columnIndex++) {
      sum += matrix.get(rowIndex, columnIndex) * vector[columnIndex];
    }
    result[rowIndex] = sum;
  }
  return result;
}

function computeBfgsSearchDirection(inverseHessianApproximation: Matrix, currentGradient: Float64Array): Float64Array {
  const approximateNewtonDirection = multiplyMatrixVector(inverseHessianApproximation, currentGradient);
  return scaleVector(approximateNewtonDirection, NEGATIVE_GRADIENT_DIRECTION);
}

function ensureDescentDirectionOrFallback(
  currentGradient: Float64Array,
  proposedSearchDirection: Float64Array,
  currentInverseHessianApproximation: Matrix,
  logger: Logger,
  iteration: number,
  currentCost: number
): { searchDirection: Float64Array; inverseHessianApproximation: Matrix } {
  const directionalDerivative = dotProduct(currentGradient, proposedSearchDirection);
  const isDescentDirection = directionalDerivative < 0.0;
  if (isDescentDirection) {
    return { searchDirection: proposedSearchDirection, inverseHessianApproximation: currentInverseHessianApproximation };
  }

  // WHY: If numerical issues yield a non-descent direction, reset H to identity and fall back to -g.
  logger.warn('bfgs', iteration, 'Non-descent direction detected; resetting inverse Hessian and using negative gradient.', [
    { key: 'Cost:', value: currentCost },
    { key: 'Directional derivative:', value: directionalDerivative }
  ]);
  return {
    searchDirection: scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION),
    inverseHessianApproximation: createIdentityMatrix(currentGradient.length)
  };
}

function updateInverseHessianApproximation(
  inverseHessianApproximation: Matrix,
  stepVector: Float64Array,
  gradientChangeVector: Float64Array,
  logger: Logger,
  iteration: number,
  currentCost: number
): Matrix {
  const stepDotGradientChange = dotProduct(stepVector, gradientChangeVector);
  const curvatureIsTooWeak = stepDotGradientChange <= MINIMUM_CURVATURE_THRESHOLD;
  if (curvatureIsTooWeak) {
    // WHY: If curvature is weak/negative, the BFGS update can break positive definiteness.
    logger.warn('bfgs', iteration, 'Curvature condition too weak; resetting inverse Hessian approximation.', [
      { key: 'Cost:', value: currentCost },
      { key: 'stepDotGradientChange:', value: stepDotGradientChange }
    ]);
    return createIdentityMatrix(stepVector.length);
  }

  const curvatureScaling = 1.0 / stepDotGradientChange;
  const stepMatrix = Matrix.columnVector(Array.from(stepVector));
  const gradientChangeMatrix = Matrix.columnVector(Array.from(gradientChangeVector));

  const identityMatrix = createIdentityMatrix(stepVector.length);
  const stepGradientOuterProduct = stepMatrix.mmul(gradientChangeMatrix.transpose()).mul(curvatureScaling);
  const gradientStepOuterProduct = gradientChangeMatrix.mmul(stepMatrix.transpose()).mul(curvatureScaling);

  const leftFactor = identityMatrix.sub(stepGradientOuterProduct);
  const rightFactor = identityMatrix.sub(gradientStepOuterProduct);

  const rankTwoPart = leftFactor.mmul(inverseHessianApproximation).mmul(rightFactor);
  const rankOnePart = stepMatrix.mmul(stepMatrix.transpose()).mul(curvatureScaling);

  return rankTwoPart.add(rankOnePart);
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
  let inverseHessianApproximation = createIdentityMatrix(currentParameters.length);

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
      currentCost
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
      newCost
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


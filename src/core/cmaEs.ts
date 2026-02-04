/**
 * This file implements vanilla CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 * for unconstrained black-box optimization (no gradients required).
 *
 * Role in system:
 * - Provides a derivative-free optimizer for scalar cost functions
 * - Complements gradient-based optimizers when gradients are unavailable
 * - Mirrors libcmaes (CMAES_DEFAULT) default parameter formulas and core stop criteria
 *
 * For first-time readers:
 * - Start with `cmaEs()` (public entry point)
 * - `computeLibcmaesDefaults()` matches libcmaes initialize_parameters() behavior
 * - Stop criteria mirror libcmaes: MAXITER/MAXFEVALS/FTARGET/TOLHISTFUN/TOLX
 */

import { CholeskyDecomposition, Matrix } from 'ml-matrix';
import type { CostFn, CmaEsOptions, CmaEsResult } from './types.js';
import { Logger } from './logger.js';
import { createSeededRandom } from '../utils/random.js';

const DEFAULT_MAX_ITERATIONS = 1000;

const DEFAULT_FUNCTION_TOLERANCE = 1e-12; // libcmaes default
const DEFAULT_PARAMETER_TOLERANCE = 1e-12; // libcmaes default
const MINIMUM_FUNCTION_TOLERANCE = 1e-12;
const MINIMUM_PARAMETER_TOLERANCE = 1e-12;

const MINIMUM_POPULATION_SIZE = 2;
const DEFAULT_STEP_SIZE_FALLBACK_SCALE = 1.0; // used to compute 1 / dim fallback

const DEFAULT_COVARIANCE_REGULARIZATION = 1e-12;
const MAX_REGULARIZATION_ATTEMPTS = 8;
const REGULARIZATION_GROWTH_BASE = 10;

const H_SIGMA_BASE = 1.4;
const H_SIGMA_DIMENSION_FACTOR_NUMERATOR = 2.0;
const H_SIGMA_POWER_FACTOR = 2.0;

const LARGE_DIMENSION_THRESHOLD_FOR_CSIGMA = 1000;

type LibcmaesDefaults = {
  populationSize: number;
  parentCount: number;
  weights: Float64Array;
  csigma: number;
  cc: number;
  c1: number;
  cmu: number;
  dsigma: number;
  psFactor: number;
  pcFactor: number;
  chiN: number;
};

type Candidate = {
  parameters: Float64Array;
  normalizedStep: Float64Array; // (x_i - mean_old) / sigma
  cost: number;
};

type StopReason = 'CONT' | 'MAXITER' | 'MAXFEVALS' | 'FTARGET' | 'TOLHISTFUN' | 'TOLX';

type StopResult = { shouldStop: boolean; converged: boolean; reason: StopReason };

type CmaEsState = {
  mean: Float64Array;
  covariance: Matrix;
  psigma: Float64Array;
  pc: Float64Array;
  sigma: number;
  sigmaInit: number;
  bestCost: number;
  bestParameters: Float64Array;
  functionEvaluations: number;
  bestCostHistory: number[];
};

function assertValidDimension(dimension: number): void {
  // Guard: CMA-ES requires at least one parameter dimension.
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`CMA-ES requires dimension >= 1, got ${dimension}`);
  }
}

function computeDefaultPopulationSize(dimension: number): number {
  // libcmaes default when lambda is unspecified or < 2:
  // lambda = 4 + floor(3 * log(dim))
  const value = 4 + Math.floor(3.0 * Math.log(dimension));
  return Math.max(MINIMUM_POPULATION_SIZE, value);
}

function computeLibcmaesDefaults(dimension: number, populationSize: number): LibcmaesDefaults {
  const parentCount = Math.floor(populationSize / 2.0);
  const weights = computeLibcmaesWeights(parentCount);

  const effectiveParentCount = computeEffectiveParentCount(weights);
  const csigma = computeLibcmaesCsigma(dimension, effectiveParentCount);
  const cc = computeLibcmaesCc(dimension, effectiveParentCount);
  const c1 = computeLibcmaesC1(dimension, effectiveParentCount);
  const cmu = computeLibcmaesCmu(dimension, effectiveParentCount, c1);
  const dsigma = computeLibcmaesDsigma(dimension, effectiveParentCount, csigma);

  return {
    populationSize,
    parentCount,
    weights,
    csigma,
    cc,
    c1,
    cmu,
    dsigma,
    psFactor: Math.sqrt(csigma * (2.0 - csigma) * effectiveParentCount),
    pcFactor: Math.sqrt(cc * (2.0 - cc) * effectiveParentCount),
    chiN: computeLibcmaesChiN(dimension)
  };
}

function computeLibcmaesWeights(parentCount: number): Float64Array {
  const weights = new Float64Array(parentCount);
  let sum = 0.0;

  for (let index = 0; index < parentCount; index++) {
    const weight = Math.log(parentCount + 1) - Math.log(index + 1);
    weights[index] = weight;
    sum += weight;
  }

  for (let index = 0; index < parentCount; index++) {
    weights[index] /= sum;
  }

  return weights;
}

function computeEffectiveParentCount(weights: Float64Array): number {
  let sum = 0.0;
  let sumSquared = 0.0;
  for (let index = 0; index < weights.length; index++) {
    sum += weights[index];
    sumSquared += weights[index] * weights[index];
  }
  return (sum * sum) / sumSquared;
}

function computeLibcmaesCsigma(dimension: number, effectiveParentCount: number): number {
  if (dimension < LARGE_DIMENSION_THRESHOLD_FOR_CSIGMA) {
    return (effectiveParentCount + 2.0) / (dimension + effectiveParentCount + 5.0);
  }
  return (Math.sqrt(effectiveParentCount) + 2.0) / (Math.sqrt(dimension) + Math.sqrt(effectiveParentCount) + 3.0);
}

function computeLibcmaesCc(dimension: number, effectiveParentCount: number): number {
  return (4.0 + effectiveParentCount / dimension) / (dimension + 4.0 + (2.0 * effectiveParentCount) / dimension);
}

function computeLibcmaesC1(dimension: number, effectiveParentCount: number): number {
  return 2.0 / (Math.pow(dimension + 1.3, 2) + effectiveParentCount);
}

function computeLibcmaesCmu(dimension: number, effectiveParentCount: number, c1: number): number {
  const cmuUnclamped =
    (2.0 * (effectiveParentCount - 2.0 + 1.0 / effectiveParentCount)) /
    (Math.pow(dimension + 2.0, 2) + effectiveParentCount);
  return Math.min(1.0 - c1, cmuUnclamped);
}

function computeLibcmaesDsigma(dimension: number, effectiveParentCount: number, csigma: number): number {
  const term = Math.sqrt((effectiveParentCount - 1.0) / (dimension + 1.0)) - 1.0;
  return 1.0 + csigma + 2.0 * Math.max(0.0, term);
}

function computeLibcmaesChiN(dimension: number): number {
  return (
    Math.sqrt(dimension) *
    (1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension * dimension))
  );
}

function computeInitialStepSize(initialStepSize: number | undefined, dimension: number, logger: Logger): number {
  if (initialStepSize !== undefined && initialStepSize > 0.0) return initialStepSize;

  // Guard: libcmaes falls back to sigma0 = 1 / dim when sigma0 is not positive.
  logger.warn('cmaEs', undefined, 'initialStepSize is missing or non-positive; falling back to 1/dim.', [
    { key: 'dim:', value: dimension }
  ]);
  return DEFAULT_STEP_SIZE_FALLBACK_SCALE / dimension;
}

function sanitizeCost(rawCost: number): number {
  return Number.isFinite(rawCost) ? rawCost : Number.POSITIVE_INFINITY;
}

function createIdentityMatrix(dimension: number): Matrix {
  return Matrix.eye(dimension, dimension);
}

function computeMaxDiagonalElement(matrix: Matrix): number {
  let maxValue = 0.0;
  for (let index = 0; index < matrix.rows; index++) {
    maxValue = Math.max(maxValue, matrix.get(index, index));
  }
  return maxValue;
}

function symmetrizeMatrixInPlace(matrix: Matrix): void {
  for (let rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
    for (let colIndex = rowIndex + 1; colIndex < matrix.columns; colIndex++) {
      const average = 0.5 * (matrix.get(rowIndex, colIndex) + matrix.get(colIndex, rowIndex));
      matrix.set(rowIndex, colIndex, average);
      matrix.set(colIndex, rowIndex, average);
    }
  }
}

function computeRegularizationLambda(base: number, attempt: number): number {
  return base * Math.pow(REGULARIZATION_GROWTH_BASE, attempt);
}

function computeCholeskyLowerOrRegularize(
  covarianceMatrix: Matrix,
  regularizationBase: number,
  logger: Logger
): Matrix {
  for (let attempt = 0; attempt < MAX_REGULARIZATION_ATTEMPTS; attempt++) {
    const lambda = computeRegularizationLambda(regularizationBase, attempt);
    const regularized = covarianceMatrix.add(createIdentityMatrix(covarianceMatrix.rows).mul(lambda));
    try {
      const decomposition = new CholeskyDecomposition(regularized);
      if (decomposition.isPositiveDefinite()) {
        if (attempt > 0) {
          logger.warn('cmaEs', undefined, 'Covariance not SPD; recovered via diagonal regularization.', [
            { key: 'regularization:', value: lambda }
          ]);
        }
        return decomposition.lowerTriangularMatrix;
      }
    } catch {
      continue;
    }
  }

  // Guard: if covariance is badly broken, reset to identity to keep optimizer running.
  logger.warn('cmaEs', undefined, 'Covariance Cholesky failed; resetting covariance to identity.', []);
  return new CholeskyDecomposition(createIdentityMatrix(covarianceMatrix.rows)).lowerTriangularMatrix;
}

function solveLowerTriangularSystem(lowerTriangular: Matrix, rhs: Float64Array): Float64Array {
  const dimension = rhs.length;
  const solution = new Float64Array(dimension);

  for (let rowIndex = 0; rowIndex < dimension; rowIndex++) {
    let sum = rhs[rowIndex];
    for (let colIndex = 0; colIndex < rowIndex; colIndex++) {
      sum -= lowerTriangular.get(rowIndex, colIndex) * solution[colIndex];
    }
    solution[rowIndex] = sum / lowerTriangular.get(rowIndex, rowIndex);
  }

  return solution;
}

function vectorNormSquared(vector: Float64Array): number {
  let sum = 0.0;
  for (let index = 0; index < vector.length; index++) {
    sum += vector[index] * vector[index];
  }
  return sum;
}

function vectorNorm(vector: Float64Array): number {
  return Math.sqrt(vectorNormSquared(vector));
}

function addScaledInPlace(target: Float64Array, source: Float64Array, scale: number): void {
  for (let index = 0; index < target.length; index++) {
    target[index] += scale * source[index];
  }
}

function scaleInPlace(vector: Float64Array, scale: number): void {
  for (let index = 0; index < vector.length; index++) {
    vector[index] *= scale;
  }
}

function subtractVectors(a: Float64Array, b: Float64Array): Float64Array {
  const result = new Float64Array(a.length);
  for (let index = 0; index < a.length; index++) {
    result[index] = a[index] - b[index];
  }
  return result;
}

function computeWeightedMean(candidates: Candidate[], weights: Float64Array, parentCount: number): Float64Array {
  const dimension = candidates[0].parameters.length;
  const mean = new Float64Array(dimension);
  for (let index = 0; index < parentCount; index++) {
    addScaledInPlace(mean, candidates[index].parameters, weights[index]);
  }
  return mean;
}

function computePcOuterProduct(pc: Float64Array): Matrix {
  const dimension = pc.length;
  const result = Matrix.zeros(dimension, dimension);
  for (let rowIndex = 0; rowIndex < dimension; rowIndex++) {
    const vRow = pc[rowIndex];
    for (let colIndex = 0; colIndex < dimension; colIndex++) {
      result.set(rowIndex, colIndex, vRow * pc[colIndex]);
    }
  }
  return result;
}

function addWeightedOuterProductInPlace(accumulator: Matrix, vector: Float64Array, weight: number): void {
  const dimension = vector.length;
  for (let rowIndex = 0; rowIndex < dimension; rowIndex++) {
    const vRow = vector[rowIndex];
    for (let colIndex = 0; colIndex < dimension; colIndex++) {
      accumulator.set(rowIndex, colIndex, accumulator.get(rowIndex, colIndex) + weight * vRow * vector[colIndex]);
    }
  }
}

function computeDefaultMaxHistorySize(dimension: number, populationSize: number): number {
  const base = 10;
  const scale = 30;
  return base + Math.ceil((scale * dimension) / populationSize);
}

function checkStopMaxFevals(functionEvaluations: number, maxFunctionEvaluations: number | undefined): StopResult {
  if (maxFunctionEvaluations !== undefined && functionEvaluations >= maxFunctionEvaluations) {
    return { shouldStop: true, converged: false, reason: 'MAXFEVALS' };
  }
  return { shouldStop: false, converged: false, reason: 'CONT' };
}

function checkStopFtarget(bestCost: number, targetCost: number | undefined): StopResult {
  if (targetCost !== undefined && bestCost <= targetCost) {
    return { shouldStop: true, converged: true, reason: 'FTARGET' };
  }
  return { shouldStop: false, converged: false, reason: 'CONT' };
}

function checkStopMaxIter(iteration: number, maxIterations: number): StopResult {
  if (iteration >= maxIterations) {
    return { shouldStop: true, converged: false, reason: 'MAXITER' };
  }
  return { shouldStop: false, converged: false, reason: 'CONT' };
}

function checkStopTolHistFun(bestCostHistory: number[], maxHistorySize: number, functionTolerance: number): StopResult {
  if (bestCostHistory.length < maxHistorySize) {
    return { shouldStop: false, converged: false, reason: 'CONT' };
  }

  let recentMin = Number.POSITIVE_INFINITY;
  let recentMax = Number.NEGATIVE_INFINITY;
  for (let index = bestCostHistory.length - maxHistorySize; index < bestCostHistory.length; index++) {
    recentMin = Math.min(recentMin, bestCostHistory[index]);
    recentMax = Math.max(recentMax, bestCostHistory[index]);
  }

  const range = Math.abs(recentMax - recentMin);
  if (range < functionTolerance) {
    return { shouldStop: true, converged: true, reason: 'TOLHISTFUN' };
  }

  return { shouldStop: false, converged: false, reason: 'CONT' };
}

function checkStopTolX(args: {
  iteration: number;
  sigma: number;
  sigmaInit: number;
  parameterTolerance: number;
  pc: Float64Array;
  covariance: Matrix;
}): StopResult {
  // Guard: TolX is not meaningful for iteration 0 (matches libcmaes behavior).
  if (args.iteration <= 0) return { shouldStop: false, converged: false, reason: 'CONT' };

  const factor = args.sigma / args.sigmaInit;
  const thresholdFactor = args.parameterTolerance * factor;

  for (let index = 0; index < args.pc.length; index++) {
    if (args.pc[index] >= thresholdFactor) {
      return { shouldStop: false, converged: false, reason: 'CONT' };
    }
  }

  for (let index = 0; index < args.covariance.rows; index++) {
    const diagonalStd = Math.sqrt(args.covariance.get(index, index));
    if (diagonalStd >= thresholdFactor) {
      return { shouldStop: false, converged: false, reason: 'CONT' };
    }
  }

  return { shouldStop: true, converged: true, reason: 'TOLX' };
}

function checkLibcmaesStopCriteria(args: {
  iteration: number;
  maxIterations: number;
  functionEvaluations: number;
  maxFunctionEvaluations: number | undefined;
  bestCost: number;
  targetCost: number | undefined;
  bestCostHistory: number[];
  maxHistorySize: number;
  functionTolerance: number;
  sigma: number;
  sigmaInit: number;
  parameterTolerance: number;
  pc: Float64Array;
  covariance: Matrix;
}): StopResult {
  const maxFevals = checkStopMaxFevals(args.functionEvaluations, args.maxFunctionEvaluations);
  if (maxFevals.shouldStop) return maxFevals;

  const ftarget = checkStopFtarget(args.bestCost, args.targetCost);
  if (ftarget.shouldStop) return ftarget;

  const maxIter = checkStopMaxIter(args.iteration, args.maxIterations);
  if (maxIter.shouldStop) return maxIter;

  const tolHistFun = checkStopTolHistFun(args.bestCostHistory, args.maxHistorySize, args.functionTolerance);
  if (tolHistFun.shouldStop) return tolHistFun;

  return checkStopTolX({
    iteration: args.iteration,
    sigma: args.sigma,
    sigmaInit: args.sigmaInit,
    parameterTolerance: args.parameterTolerance,
    pc: args.pc,
    covariance: args.covariance
  });
}

function computeHsigThreshold(iteration: number, csigma: number, chiN: number, dimension: number): number {
  const decay = Math.pow(1.0 - csigma, H_SIGMA_POWER_FACTOR * (iteration + 1));
  const normalization = Math.sqrt(1.0 - decay);
  const dimensionFactor = H_SIGMA_BASE + H_SIGMA_DIMENSION_FACTOR_NUMERATOR / (dimension + 1.0);
  return normalization * dimensionFactor * chiN;
}

function sampleCandidate(
  mean: Float64Array,
  sigma: number,
  lowerTriangular: Matrix,
  nextStandardNormal: () => number
): { parameters: Float64Array; normalizedStep: Float64Array } {
  const dimension = mean.length;
  const z = new Float64Array(dimension);
  for (let index = 0; index < dimension; index++) z[index] = nextStandardNormal();

  const y = new Float64Array(dimension);
  for (let rowIndex = 0; rowIndex < dimension; rowIndex++) {
    let sum = 0.0;
    for (let colIndex = 0; colIndex <= rowIndex; colIndex++) {
      sum += lowerTriangular.get(rowIndex, colIndex) * z[colIndex];
    }
    y[rowIndex] = sum;
  }

  const parameters = new Float64Array(dimension);
  for (let index = 0; index < dimension; index++) parameters[index] = mean[index] + sigma * y[index];
  return { parameters, normalizedStep: y };
}

function initializeState(
  initialParameters: Float64Array,
  sigmaInit: number,
  costFunction: CostFn
): CmaEsState {
  const mean = new Float64Array(initialParameters);
  const bestCost = sanitizeCost(costFunction(mean));
  return {
    mean,
    covariance: createIdentityMatrix(initialParameters.length),
    psigma: new Float64Array(initialParameters.length),
    pc: new Float64Array(initialParameters.length),
    sigma: sigmaInit,
    sigmaInit,
    bestCost,
    bestParameters: new Float64Array(mean),
    functionEvaluations: 1,
    bestCostHistory: [bestCost]
  };
}

function updateBestIfImproved(state: CmaEsState, bestCandidate: Candidate): void {
  if (bestCandidate.cost >= state.bestCost) return;
  state.bestCost = bestCandidate.cost;
  state.bestParameters = new Float64Array(bestCandidate.parameters);
}

function pushBestCostHistory(state: CmaEsState, bestCost: number, maxHistorySize: number): void {
  state.bestCostHistory.push(bestCost);
  while (state.bestCostHistory.length > maxHistorySize) state.bestCostHistory.shift();
}

function buildResult(state: CmaEsState, defaults: LibcmaesDefaults, iterations: number, converged: boolean): CmaEsResult {
  const finalMaxStdDev = state.sigma * Math.sqrt(Math.max(0.0, computeMaxDiagonalElement(state.covariance)));
  return {
    finalParameters: state.bestParameters,
    parameters: state.bestParameters,
    iterations,
    converged,
    finalCost: state.bestCost,
    populationSize: defaults.populationSize,
    functionEvaluations: state.functionEvaluations,
    finalStepSize: state.sigma,
    finalMaxStdDev
  };
}

function runOneGeneration(args: {
  state: CmaEsState;
  defaults: LibcmaesDefaults;
  costFunction: CostFn;
  nextStandardNormal: () => number;
  maxFunctionEvaluations: number | undefined;
  covarianceRegularizationBase: number;
  logger: Logger;
}): { candidates: Candidate[]; lowerTriangular: Matrix } {
  const lowerTriangular = computeCholeskyLowerOrRegularize(args.state.covariance, args.covarianceRegularizationBase, args.logger);

  const candidates: Candidate[] = [];
  for (let sampleIndex = 0; sampleIndex < args.defaults.populationSize; sampleIndex++) {
    const sampled = sampleCandidate(args.state.mean, args.state.sigma, lowerTriangular, args.nextStandardNormal);
    const cost = sanitizeCost(args.costFunction(sampled.parameters));
    args.state.functionEvaluations += 1;
    candidates.push({ parameters: sampled.parameters, normalizedStep: sampled.normalizedStep, cost });

    if (args.maxFunctionEvaluations !== undefined && args.state.functionEvaluations >= args.maxFunctionEvaluations) {
      break;
    }
  }

  candidates.sort((a, b) => a.cost - b.cost);
  return { candidates, lowerTriangular };
}

function updateDistributionParameters(args: {
  state: CmaEsState;
  defaults: LibcmaesDefaults;
  candidates: Candidate[];
  lowerTriangular: Matrix;
  iteration: number;
}): void {
  const parentCount = Math.min(args.defaults.parentCount, args.candidates.length);
  const xmean = computeWeightedMean(args.candidates, args.defaults.weights, parentCount);

  const diffxmean = subtractVectors(xmean, args.state.mean);
  scaleInPlace(diffxmean, 1.0 / args.state.sigma);

  scaleInPlace(args.state.psigma, 1.0 - args.defaults.csigma);
  const csqinvDiff = solveLowerTriangularSystem(args.lowerTriangular, diffxmean);
  addScaledInPlace(args.state.psigma, csqinvDiff, args.defaults.psFactor);
  const normPs = vectorNorm(args.state.psigma);

  const hsigThreshold = computeHsigThreshold(args.iteration, args.defaults.csigma, args.defaults.chiN, args.state.mean.length);
  const hsig = normPs < hsigThreshold ? 1.0 : 0.0;

  scaleInPlace(args.state.pc, 1.0 - args.defaults.cc);
  addScaledInPlace(args.state.pc, diffxmean, hsig * args.defaults.pcFactor);

  const spc = computePcOuterProduct(args.state.pc);
  const wdiff = Matrix.zeros(args.state.mean.length, args.state.mean.length);
  for (let index = 0; index < parentCount; index++) {
    addWeightedOuterProductInPlace(wdiff, args.candidates[index].normalizedStep, args.defaults.weights[index]);
  }

  const covarianceScale =
    1.0 -
    args.defaults.c1 -
    args.defaults.cmu +
    (1.0 - hsig) * args.defaults.c1 * args.defaults.cc * (2.0 - args.defaults.cc);
  args.state.covariance = args.state.covariance.mul(covarianceScale).add(spc.mul(args.defaults.c1)).add(wdiff.mul(args.defaults.cmu));
  symmetrizeMatrixInPlace(args.state.covariance);

  const sigmaExponent = (args.defaults.csigma / args.defaults.dsigma) * (normPs / args.defaults.chiN - 1.0);
  args.state.sigma *= Math.exp(sigmaExponent);

  args.state.mean = xmean;
}

export function cmaEs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  options: CmaEsOptions = {}
): CmaEsResult {
  const dimension = initialParameters.length;
  assertValidDimension(dimension);

  const logger = new Logger(options.logLevel, options.verbose);

  const populationSize = Math.max(MINIMUM_POPULATION_SIZE, options.populationSize ?? computeDefaultPopulationSize(dimension));
  const defaults = computeLibcmaesDefaults(dimension, populationSize);

  const sigmaInit = computeInitialStepSize(options.initialStepSize, dimension, logger);
  const functionTolerance = Math.max(options.functionTolerance ?? DEFAULT_FUNCTION_TOLERANCE, MINIMUM_FUNCTION_TOLERANCE);
  const parameterTolerance = Math.max(options.parameterTolerance ?? DEFAULT_PARAMETER_TOLERANCE, MINIMUM_PARAMETER_TOLERANCE);

  const covarianceRegularizationBase = options.covarianceRegularization ?? DEFAULT_COVARIANCE_REGULARIZATION;
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const maxFunctionEvaluations = options.maxFunctionEvaluations;
  const targetCost = options.targetCost;

  const maxHistorySize =
    options.maxHistorySize && options.maxHistorySize > 0
      ? options.maxHistorySize
      : computeDefaultMaxHistorySize(dimension, defaults.populationSize);

  const seededRandom = createSeededRandom(options.randomSeed);
  const state = initializeState(initialParameters, sigmaInit, costFunction);

  logger.info('cmaEs', 0, 'Starting', [
    { key: 'dim:', value: dimension },
    { key: 'lambda:', value: defaults.populationSize },
    { key: 'mu:', value: defaults.parentCount },
    { key: 'sigma0:', value: sigmaInit }
  ]);

  const initialStop = checkLibcmaesStopCriteria({
    iteration: 0,
    maxIterations,
    functionEvaluations: state.functionEvaluations,
    maxFunctionEvaluations,
    bestCost: state.bestCost,
    targetCost,
    bestCostHistory: state.bestCostHistory,
    maxHistorySize,
    functionTolerance,
    sigma: state.sigma,
    sigmaInit: state.sigmaInit,
    parameterTolerance,
    pc: state.pc,
    covariance: state.covariance
  });
  if (initialStop.shouldStop) return buildResult(state, defaults, 0, initialStop.converged);

  const onIteration = options.onIteration;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const { candidates, lowerTriangular } = runOneGeneration({
      state,
      defaults,
      costFunction,
      nextStandardNormal: seededRandom.nextStandardNormal,
      maxFunctionEvaluations,
      covarianceRegularizationBase,
      logger
    });

    if (candidates.length === 0) break;

    updateBestIfImproved(state, candidates[0]);
    pushBestCostHistory(state, state.bestCost, maxHistorySize);
    if (onIteration) onIteration(iteration, state.bestCost, state.bestParameters);

    const stop = checkLibcmaesStopCriteria({
      iteration: iteration + 1,
      maxIterations,
      functionEvaluations: state.functionEvaluations,
      maxFunctionEvaluations,
      bestCost: state.bestCost,
      targetCost,
      bestCostHistory: state.bestCostHistory,
      maxHistorySize,
      functionTolerance,
      sigma: state.sigma,
      sigmaInit: state.sigmaInit,
      parameterTolerance,
      pc: state.pc,
      covariance: state.covariance
    });
    if (stop.shouldStop) {
      logger.info('cmaEs', iteration, `Stopped (${stop.reason})`, [
        { key: 'bestCost:', value: state.bestCost },
        { key: 'sigma:', value: state.sigma },
        { key: 'fevals:', value: state.functionEvaluations }
      ]);
      return buildResult(state, defaults, iteration + 1, stop.converged);
    }

    updateDistributionParameters({ state, defaults, candidates, lowerTriangular, iteration });

    logger.debug('cmaEs', iteration, 'Progress', [
      { key: 'bestCost:', value: state.bestCost },
      { key: 'sigma:', value: state.sigma },
      { key: 'fevals:', value: state.functionEvaluations }
    ]);
  }

  logger.warn('cmaEs', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: maxIterations },
    { key: 'bestCost:', value: state.bestCost },
    { key: 'fevals:', value: state.functionEvaluations }
  ]);

  return buildResult(state, defaults, maxIterations, false);
}


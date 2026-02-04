/**
 * This file implements vanilla CMA-ES and IPOP-CMA-ES restart strategy
 * for unconstrained black-box optimization (no gradients required).
 *
 * Role in system:
 * - Provides a derivative-free optimizer for scalar cost functions
 * - Adds IPOP restarts (λ doubles per restart) while preserving libcmaes semantics
 * - Mirrors libcmaes default parameter formulas and core stop criteria
 *
 * For first-time readers:
 * - Start with `cmaEs()` (public entry point)
 * - `runSingleCmaEs()` executes one CMA-ES run (no restarts)
 * - Restart logic wraps `runSingleCmaEs()` when `restartStrategy: "ipop"`
 */

import { CholeskyDecomposition, Matrix } from 'ml-matrix';
import type { CostFn, CmaEsOptions, CmaEsResult } from './types.js';
import { Logger } from './logger.js';
import { createSeededRandom } from '../utils/random.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_MAX_RESTARTS = 9; // libcmaes default
const DEFAULT_RESTART_STRATEGY: 'none' | 'ipop' = 'none';

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
const IPOPN_LAMBDA_MULTIPLIER = 2;

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
  bestCostHistory: number[];
};

type RunCounters = {
  iterations: number;
  functionEvaluations: number;
};

type RunContext = {
  dimension: number;
  defaults: LibcmaesDefaults;
  maxHistorySize: number;
  functionTolerance: number;
  parameterTolerance: number;
  covarianceRegularizationBase: number;
  maxIterations: number;
  maxFunctionEvaluations: number | undefined;
  targetCost: number | undefined;
  costFunction: CostFn;
  logger: Logger;
  nextStandardNormal: () => number;
  onIteration: CmaEsOptions['onIteration'] | undefined;
  counters: RunCounters;
  profiling: Profiling | undefined;
};

type RunResult = {
  stop: StopResult;
  state: CmaEsState;
};

type Profiling = {
  totalMs: number;
  costMs: number;
  choleskyMs: number;
  samplingMs: number;
  updateMs: number;
};

function nowMs(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function assertValidDimension(dimension: number): void {
  // Guard: CMA-ES requires at least one parameter dimension.
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`CMA-ES requires dimension >= 1, got ${dimension}`);
  }
}

function normalizePopulationSize(
  dimension: number,
  populationSize: number | undefined,
  logger: Logger
): number {
  const defaultValue = computeDefaultPopulationSize(dimension);
  if (populationSize === undefined) return defaultValue;
  if (populationSize < MINIMUM_POPULATION_SIZE || !Number.isFinite(populationSize)) {
    logger.warn('cmaEs', undefined, 'Invalid populationSize; falling back to default.', [
      { key: 'populationSize:', value: populationSize },
      { key: 'default:', value: defaultValue }
    ]);
    return defaultValue;
  }
  return Math.floor(populationSize);
}

function normalizeMaxIterations(value: number | undefined, logger: Logger): number {
  if (value === undefined) return DEFAULT_MAX_ITERATIONS;
  if (!Number.isFinite(value) || value <= 0) {
    logger.warn('cmaEs', undefined, 'Invalid maxIterations; falling back to default.', [
      { key: 'maxIterations:', value }
    ]);
    return DEFAULT_MAX_ITERATIONS;
  }
  return Math.floor(value);
}

function normalizeMaxFunctionEvaluations(value: number | undefined, logger: Logger): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isFinite(value) || value <= 0) {
    logger.warn('cmaEs', undefined, 'Invalid maxFunctionEvaluations; disabling evaluation budget.', [
      { key: 'maxFunctionEvaluations:', value }
    ]);
    return undefined;
  }
  return Math.floor(value);
}

function normalizeMaxRestarts(value: number | undefined, logger: Logger): number {
  if (value === undefined) return DEFAULT_MAX_RESTARTS;
  if (!Number.isFinite(value) || value < 0) {
    logger.warn('cmaEs', undefined, 'Invalid maxRestarts; falling back to default.', [
      { key: 'maxRestarts:', value }
    ]);
    return DEFAULT_MAX_RESTARTS;
  }
  return Math.floor(value);
}

function normalizeRestartStrategy(value: CmaEsOptions['restartStrategy'], logger: Logger): 'none' | 'ipop' {
  if (value === undefined) return DEFAULT_RESTART_STRATEGY;
  if (value === 'none' || value === 'ipop') return value;
  logger.warn('cmaEs', undefined, 'Unknown restartStrategy; falling back to "none".', [
    { key: 'restartStrategy:', value: Number.NaN }
  ]);
  return DEFAULT_RESTART_STRATEGY;
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
  costFunction: CostFn,
  counters: RunCounters,
  profiling: Profiling | undefined
): CmaEsState {
  const mean = new Float64Array(initialParameters);
  const costStart = profiling ? nowMs() : 0;
  const bestCost = sanitizeCost(costFunction(mean));
  if (profiling) profiling.costMs += nowMs() - costStart;
  counters.functionEvaluations += 1;
  return {
    mean,
    covariance: createIdentityMatrix(initialParameters.length),
    psigma: new Float64Array(initialParameters.length),
    pc: new Float64Array(initialParameters.length),
    sigma: sigmaInit,
    sigmaInit,
    bestCost,
    bestParameters: new Float64Array(mean),
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

function buildResult(
  state: CmaEsState,
  defaults: LibcmaesDefaults,
  iterations: number,
  converged: boolean,
  stopReason?: CmaEsResult['stopReason'],
  functionEvaluations?: number,
  profiling?: Profiling
): CmaEsResult {
  const finalMaxStdDev = state.sigma * Math.sqrt(Math.max(0.0, computeMaxDiagonalElement(state.covariance)));
  return {
    finalParameters: state.bestParameters,
    parameters: state.bestParameters,
    iterations,
    converged,
    finalCost: state.bestCost,
    populationSize: defaults.populationSize,
    functionEvaluations: functionEvaluations ?? 0,
    finalStepSize: state.sigma,
    finalMaxStdDev,
    stopReason,
    profiling
  };
}

function runOneGeneration(context: RunContext, state: CmaEsState): { candidates: Candidate[]; lowerTriangular: Matrix } {
  const choleskyStart = context.profiling ? nowMs() : 0;
  const lowerTriangular = computeCholeskyLowerOrRegularize(
    state.covariance,
    context.covarianceRegularizationBase,
    context.logger
  );
  if (context.profiling) {
    context.profiling.choleskyMs += nowMs() - choleskyStart;
  }
  const candidates: Candidate[] = [];

  for (let sampleIndex = 0; sampleIndex < context.defaults.populationSize; sampleIndex++) {
    const sampleStart = context.profiling ? nowMs() : 0;
    const sampled = sampleCandidate(state.mean, state.sigma, lowerTriangular, context.nextStandardNormal);
    if (context.profiling) {
      context.profiling.samplingMs += nowMs() - sampleStart;
    }

    const costStart = context.profiling ? nowMs() : 0;
    const cost = sanitizeCost(context.costFunction(sampled.parameters));
    if (context.profiling) {
      context.profiling.costMs += nowMs() - costStart;
    }
    context.counters.functionEvaluations += 1;
    candidates.push({ parameters: sampled.parameters, normalizedStep: sampled.normalizedStep, cost });

    if (
      context.maxFunctionEvaluations !== undefined &&
      context.counters.functionEvaluations >= context.maxFunctionEvaluations
    ) {
      break;
    }
  }

  candidates.sort((a, b) => a.cost - b.cost);
  return { candidates, lowerTriangular };
}

function updateDistributionParameters(
  context: RunContext,
  state: CmaEsState,
  candidates: Candidate[],
  lowerTriangular: Matrix,
  iteration: number
): void {
  const updateStart = context.profiling ? nowMs() : 0;
  const parentCount = Math.min(context.defaults.parentCount, candidates.length);
  const xmean = computeWeightedMean(candidates, context.defaults.weights, parentCount);

  const diffxmean = subtractVectors(xmean, state.mean);
  scaleInPlace(diffxmean, 1.0 / state.sigma);

  scaleInPlace(state.psigma, 1.0 - context.defaults.csigma);
  const csqinvDiff = solveLowerTriangularSystem(lowerTriangular, diffxmean);
  addScaledInPlace(state.psigma, csqinvDiff, context.defaults.psFactor);
  const normPs = vectorNorm(state.psigma);

  const hsigThreshold = computeHsigThreshold(iteration, context.defaults.csigma, context.defaults.chiN, context.dimension);
  const hsig = normPs < hsigThreshold ? 1.0 : 0.0;

  scaleInPlace(state.pc, 1.0 - context.defaults.cc);
  addScaledInPlace(state.pc, diffxmean, hsig * context.defaults.pcFactor);

  const spc = computePcOuterProduct(state.pc);
  const wdiff = Matrix.zeros(context.dimension, context.dimension);
  for (let index = 0; index < parentCount; index++) {
    addWeightedOuterProductInPlace(wdiff, candidates[index].normalizedStep, context.defaults.weights[index]);
  }

  const covarianceScale =
    1.0 -
    context.defaults.c1 -
    context.defaults.cmu +
    (1.0 - hsig) * context.defaults.c1 * context.defaults.cc * (2.0 - context.defaults.cc);
  state.covariance = state.covariance.mul(covarianceScale).add(spc.mul(context.defaults.c1)).add(wdiff.mul(context.defaults.cmu));
  symmetrizeMatrixInPlace(state.covariance);

  const sigmaExponent = (context.defaults.csigma / context.defaults.dsigma) * (normPs / context.defaults.chiN - 1.0);
  state.sigma *= Math.exp(sigmaExponent);

  state.mean = xmean;
  if (context.profiling) {
    context.profiling.updateMs += nowMs() - updateStart;
  }
}

function runSingleCmaEs(context: RunContext, state: CmaEsState): RunResult {
  const initialStop = checkLibcmaesStopCriteria({
    iteration: context.counters.iterations,
    maxIterations: context.maxIterations,
    functionEvaluations: context.counters.functionEvaluations,
    maxFunctionEvaluations: context.maxFunctionEvaluations,
    bestCost: state.bestCost,
    targetCost: context.targetCost,
    bestCostHistory: state.bestCostHistory,
    maxHistorySize: context.maxHistorySize,
    functionTolerance: context.functionTolerance,
    sigma: state.sigma,
    sigmaInit: state.sigmaInit,
    parameterTolerance: context.parameterTolerance,
    pc: state.pc,
    covariance: state.covariance
  });
  if (initialStop.shouldStop) {
    return { stop: initialStop, state };
  }

  while (true) {
    const { candidates, lowerTriangular } = runOneGeneration(context, state);
    if (candidates.length === 0) {
      const budgetStop = checkStopMaxFevals(context.counters.functionEvaluations, context.maxFunctionEvaluations);
      return { stop: budgetStop, state };
    }

    updateBestIfImproved(state, candidates[0]);
    pushBestCostHistory(state, state.bestCost, context.maxHistorySize);

    if (context.onIteration) {
      context.onIteration(context.counters.iterations, state.bestCost, state.bestParameters);
    }

    const stop = checkLibcmaesStopCriteria({
      iteration: context.counters.iterations + 1,
      maxIterations: context.maxIterations,
      functionEvaluations: context.counters.functionEvaluations,
      maxFunctionEvaluations: context.maxFunctionEvaluations,
      bestCost: state.bestCost,
      targetCost: context.targetCost,
      bestCostHistory: state.bestCostHistory,
      maxHistorySize: context.maxHistorySize,
      functionTolerance: context.functionTolerance,
      sigma: state.sigma,
      sigmaInit: state.sigmaInit,
      parameterTolerance: context.parameterTolerance,
      pc: state.pc,
      covariance: state.covariance
    });
    if (stop.shouldStop) {
      return { stop, state };
    }

    updateDistributionParameters(context, state, candidates, lowerTriangular, context.counters.iterations);
    context.counters.iterations += 1;

    context.logger.debug('cmaEs', context.counters.iterations, 'Progress', [
      { key: 'bestCost:', value: state.bestCost },
      { key: 'sigma:', value: state.sigma },
      { key: 'fevals:', value: context.counters.functionEvaluations }
    ]);
  }
}

export function cmaEs(
  initialParameters: Float64Array,
  costFunction: CostFn,
  options: CmaEsOptions = {}
): CmaEsResult {
  const dimension = initialParameters.length;
  assertValidDimension(dimension);

  const logger = new Logger(options.logLevel, options.verbose);
  const restartStrategy = normalizeRestartStrategy(options.restartStrategy, logger);
  const maxRestarts = normalizeMaxRestarts(options.maxRestarts, logger);
  const maxIterations = normalizeMaxIterations(options.maxIterations, logger);
  const maxFunctionEvaluations = normalizeMaxFunctionEvaluations(options.maxFunctionEvaluations, logger);

  const functionTolerance = Math.max(options.functionTolerance ?? DEFAULT_FUNCTION_TOLERANCE, MINIMUM_FUNCTION_TOLERANCE);
  const parameterTolerance = Math.max(options.parameterTolerance ?? DEFAULT_PARAMETER_TOLERANCE, MINIMUM_PARAMETER_TOLERANCE);
  const covarianceRegularizationBase = options.covarianceRegularization ?? DEFAULT_COVARIANCE_REGULARIZATION;

  const sigmaInit = computeInitialStepSize(options.initialStepSize, dimension, logger);
  const targetCost = options.targetCost;
  const onIteration = options.onIteration;

  const seededRandom = createSeededRandom(options.randomSeed);
  const counters: RunCounters = { iterations: 0, functionEvaluations: 0 };
  const profiling: Profiling | undefined = options.profiling
    ? { totalMs: 0, costMs: 0, choleskyMs: 0, samplingMs: 0, updateMs: 0 }
    : undefined;
  const totalStart = profiling ? nowMs() : 0;

  let populationSize = normalizePopulationSize(dimension, options.populationSize, logger);
  let defaults = computeLibcmaesDefaults(dimension, populationSize);
  let maxHistorySize =
    options.maxHistorySize && options.maxHistorySize > 0
      ? options.maxHistorySize
      : computeDefaultMaxHistorySize(dimension, defaults.populationSize);

  logger.info('cmaEs', 0, 'Starting', [
    { key: 'dim:', value: dimension },
    { key: 'lambda:', value: defaults.populationSize },
    { key: 'mu:', value: defaults.parentCount },
    { key: 'sigma0:', value: sigmaInit }
  ]);

  let globalBestCost = Number.POSITIVE_INFINITY;
  let globalBestParameters = new Float64Array(initialParameters);
  let globalStopReason: CmaEsResult['stopReason'] | undefined;
  let globalConverged = false;
  let globalState: CmaEsState | null = null;

  const totalRuns = restartStrategy === 'ipop' ? maxRestarts + 1 : 1;
  for (let runIndex = 0; runIndex < totalRuns; runIndex++) {
    defaults = computeLibcmaesDefaults(dimension, populationSize);
    maxHistorySize =
      options.maxHistorySize && options.maxHistorySize > 0
        ? options.maxHistorySize
        : computeDefaultMaxHistorySize(dimension, defaults.populationSize);

    const state = initializeState(initialParameters, sigmaInit, costFunction, counters, profiling);
    const context: RunContext = {
      dimension,
      defaults,
      maxHistorySize,
      functionTolerance,
      parameterTolerance,
      covarianceRegularizationBase,
      maxIterations,
      maxFunctionEvaluations,
      targetCost,
      costFunction,
      logger,
      nextStandardNormal: seededRandom.nextStandardNormal,
      onIteration,
      counters,
      profiling
    };

    const runResult = runSingleCmaEs(context, state);
    globalState = runResult.state;
    if (globalState.bestCost < globalBestCost) {
      globalBestCost = globalState.bestCost;
      globalBestParameters = new Float64Array(globalState.bestParameters);
    }

    if (runResult.stop.reason === 'FTARGET') {
      globalStopReason = 'FTARGET';
      globalConverged = true;
      break;
    }

    if (runResult.stop.reason === 'MAXFEVALS' || runResult.stop.reason === 'MAXITER') {
      globalStopReason = runResult.stop.reason;
      globalConverged = false;
      break;
    }

    if (restartStrategy !== 'ipop') {
      globalStopReason = runResult.stop.reason === 'CONT' ? undefined : runResult.stop.reason;
      globalConverged = runResult.stop.converged;
      break;
    }

    if (runIndex >= maxRestarts) {
      globalStopReason = 'IPOP_MAX_RESTARTS';
      globalConverged = false;
      break;
    }

    populationSize *= IPOPN_LAMBDA_MULTIPLIER;
  }

  if (!globalState) {
    const fallbackDefaults = computeLibcmaesDefaults(dimension, populationSize);
    const fallbackState = initializeState(initialParameters, sigmaInit, costFunction, counters, profiling);
    if (profiling) profiling.totalMs = nowMs() - totalStart;
    return buildResult(
      fallbackState,
      fallbackDefaults,
      counters.iterations,
      false,
      globalStopReason,
      counters.functionEvaluations,
      profiling
    );
  }

  globalState.bestCost = globalBestCost;
  globalState.bestParameters = globalBestParameters;
  if (profiling) profiling.totalMs = nowMs() - totalStart;

  return buildResult(
    globalState,
    defaults,
    counters.iterations,
    globalConverged,
    globalStopReason,
    counters.functionEvaluations,
    profiling
  );
}


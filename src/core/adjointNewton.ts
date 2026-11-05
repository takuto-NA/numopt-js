/**
 * Reduced-space Newton-adjoint using H = f_pp + S^T f_xx S.
 * Uses damped Cholesky and verified Armijo trials; only a small feasible gradient means convergence.
 */

import type {
  AdjointNewtonOptions, AdjointNewtonExactOptions, AdjointNewtonResult, AdjointNewtonStopReason,
  ConstrainedCostFn, ConstrainedResidualFn, ConstraintFn, LineSearchOptions
} from './types.js';
import {
  areParametersEqual, computeCost, computeReducedGradient, createReducedObjective,
  initializeFeasibleReducedSpace, resolveAdjointReducedSpaceSettings,
  type AdjointReducedSpaceSettings, type ObjectiveKind, type ReducedGradientEvaluation
} from './adjointReducedSpace.js';
import { computeApproximateAdjointHessian, computeSafeguardedNewtonDirection } from './adjointNewtonModel.js';
import { computeExactAdjointHessian } from './adjointNewtonExactModel.js';
import { backtrackingLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { addVectors, dotProduct, scaleVector, vectorNorm } from '../utils/matrix.js';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_TOLERANCE = 1e-12;
const DEFAULT_INITIAL_STEP = 1;
const DEFAULT_ARMIJO_PARAMETER = 1e-4;
const DEFAULT_CONTRACTION_FACTOR = 0.5;
const DEFAULT_LINE_SEARCH_ITERATIONS = 50;
const MAXIMUM_ARMIJO_PARAMETER = 0.5;
const STATE_SOLVE_ROUNDOFF_FACTOR = 100;
const MAX_UNCHANGED_COST_STEPS = 3;

type NewtonOptions = AdjointNewtonOptions & AdjointNewtonExactOptions;
type RuntimeSettings = NewtonOptions & AdjointReducedSpaceSettings & {
  maxIterations: number;
  tolerance: number;
  stepTolerance: number;
  hessianDamping: number;
  lineSearchOptions: Required<LineSearchOptions>;
  hessianModel: 'approximate' | 'exact';
  algorithmName: string;
};

function requirePositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
}

function resolveSettings(options: NewtonOptions, hessianModel: 'approximate' | 'exact'): RuntimeSettings {
  const settings = {
    ...options,
    ...resolveAdjointReducedSpaceSettings(options),
    maxIterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    tolerance: options.tolerance ?? DEFAULT_TOLERANCE,
    stepTolerance: options.stepTolerance ?? DEFAULT_STEP_TOLERANCE,
    hessianDamping: options.hessianDamping ?? 0,
    hessianModel,
    algorithmName: hessianModel === 'exact' ? 'adjointNewtonExact' : 'adjointNewton',
    lineSearchOptions: {
      initialStepSize: options.lineSearchOptions?.initialStepSize ?? DEFAULT_INITIAL_STEP,
      contractionFactor: options.lineSearchOptions?.contractionFactor ?? DEFAULT_CONTRACTION_FACTOR,
      armijoParameter: options.lineSearchOptions?.armijoParameter ?? DEFAULT_ARMIJO_PARAMETER,
      maxIterations: options.lineSearchOptions?.maxIterations ?? DEFAULT_LINE_SEARCH_ITERATIONS
    }
  };
  for (const name of ['tolerance', 'constraintTolerance', 'stepTolerance', 'stepSizeP', 'stepSizeX'] as const) {
    requirePositive(settings[name], name);
  }
  for (const name of ['hessianStepSizeP', 'hessianStepSizeX', 'hessianStepSize'] as const) {
    if (settings[name] !== undefined) requirePositive(settings[name]!, name);
  }
  for (const name of ['regularization', 'hessianDamping'] as const) {
    if (!Number.isFinite(settings[name]) || settings[name] < 0) throw new Error(`${name} must be finite and nonnegative`);
  }
  if (!Number.isInteger(settings.maxIterations) || settings.maxIterations < 0) {
    throw new Error('maxIterations must be a nonnegative integer');
  }
  validateLineSearch(settings.lineSearchOptions);
  return settings;
}

function validateLineSearch(options: Required<LineSearchOptions>): void {
  requirePositive(options.initialStepSize, 'lineSearchOptions.initialStepSize');
  requirePositive(options.contractionFactor, 'lineSearchOptions.contractionFactor');
  requirePositive(options.armijoParameter, 'lineSearchOptions.armijoParameter');
  if (options.contractionFactor >= 1 || options.armijoParameter >= MAXIMUM_ARMIJO_PARAMETER) {
    throw new Error('Line search requires contractionFactor < 1 and armijoParameter < 0.5');
  }
  if (!Number.isInteger(options.maxIterations) || options.maxIterations <= 0) {
    throw new Error('lineSearchOptions.maxIterations must be a positive integer');
  }
}

type Iterate = { parameters: Float64Array; states: Float64Array; cost: number; reduced: ReducedGradientEvaluation };
type Diagnostics = {
  hessianEvaluations: number; dampedSteps: number; gradientFallbackSteps: number;
  finalHessianDamping: number; usedLineSearch: boolean;
};

function buildResult(
  iterate: Iterate, constraint: ConstraintFn, iterations: number,
  stopReason: AdjointNewtonStopReason, diagnostics: Diagnostics
): AdjointNewtonResult {
  return {
    finalParameters: iterate.parameters, parameters: iterate.parameters, finalStates: iterate.states,
    finalCost: iterate.cost, finalGradientNorm: vectorNorm(iterate.reduced.gradient),
    finalConstraintNorm: vectorNorm(constraint(iterate.parameters, iterate.states)),
    iterations, converged: stopReason === 'gradientTolerance', stopReason, ...diagnostics
  };
}

function evaluateReduced(
  parameters: Float64Array, states: Float64Array,
  objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  settings: RuntimeSettings, logger: Logger, objectiveKind: ObjectiveKind, knownCost?: number
): Iterate {
  const cost = knownCost ?? computeCost(objective, parameters, states, objectiveKind);
  const reduced = computeReducedGradient(parameters, states, objective, constraint,
    settings, logger, settings.algorithmName, objectiveKind);
  if (!Number.isFinite(cost) || !reduced.gradient.every(Number.isFinite) || !Number.isFinite(vectorNorm(reduced.gradient))) {
    throw new Error('adjointNewton requires finite cost and reduced gradient at accepted states');
  }
  return { parameters, states, cost, reduced };
}

function findStep(
  current: Iterate, direction: Float64Array,
  objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  settings: RuntimeSettings, logger: Logger, objectiveKind: ObjectiveKind
): { parameters: Float64Array; states: Float64Array; cost: number } | undefined {
  const reducedObjective = createReducedObjective(current.parameters, current.states, objective,
    constraint, settings, logger, settings.algorithmName, objectiveKind, current.reduced.partials);
  let cachedParameters = current.parameters;
  let cachedCost = current.cost;
  const evaluateCost = (parameters: Float64Array): number => {
    if (areParametersEqual(parameters, cachedParameters)) return cachedCost;
    const value = reducedObjective.evaluateCost(parameters);
    cachedParameters = new Float64Array(parameters);
    cachedCost = Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
    return cachedCost;
  };
  const stepSize = backtrackingLineSearch(evaluateCost, () => current.reduced.gradient,
    current.parameters, direction, settings.lineSearchOptions);
  const parameters = addVectors(current.parameters, scaleVector(direction, stepSize));
  const cost = evaluateCost(parameters);
  const threshold = current.cost + settings.lineSearchOptions.armijoParameter * stepSize *
    dotProduct(current.reduced.gradient, direction);
  // WHY: The shared backtracker returns its final step even on exhaustion. Verify before accepting it.
  if (!Number.isFinite(cost) || cost > threshold || stepSize <= 0) return undefined;
  const states = reducedObjective.restoreStates(parameters);
  if (states === undefined) return undefined;
  return { parameters, states, cost };
}

function proposeStep(
  current: Iterate, objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  settings: RuntimeSettings, logger: Logger, objectiveKind: ObjectiveKind, diagnostics: Diagnostics
): ReturnType<typeof findStep> {
  const cost = (parameters: Float64Array, states: Float64Array): number => computeCost(objective, parameters, states, objectiveKind);
  const hessian = settings.hessianModel === 'exact'
    ? computeExactAdjointHessian(current.parameters, current.states, cost, constraint, current.reduced, settings)
    : computeApproximateAdjointHessian(current.parameters, current.states, cost, current.reduced.partials, settings);
  diagnostics.hessianEvaluations++;
  const direction = computeSafeguardedNewtonDirection(hessian, current.reduced.gradient, settings.hessianDamping);
  diagnostics.finalHessianDamping = direction.damping;
  if (direction.damping > 0) diagnostics.dampedSteps++;
  if (direction.usedGradientFallback) diagnostics.gradientFallbackSteps++;
  diagnostics.usedLineSearch = true;
  return findStep(current, direction.direction, objective, constraint, settings, logger, objectiveKind);
}

/**
 * Minimizes f(p,x(p)) with the section-12 partial-Hessian approximation.
 * Accepts a scalar cost or residuals (f = 1/2 ||r||²); all Hessian callbacks refer to f.
 * Does not include mixed partials or nonlinear state curvature. See README for limitations.
 */
export function adjointNewton(
  initialParameters: Float64Array, initialStates: Float64Array,
  objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  options: AdjointNewtonOptions = {}
): AdjointNewtonResult {
  return runAdjointNewton(initialParameters, initialStates, objective, constraint, options, 'approximate');
}

/** Shared globalized iteration: only the Hessian model differs between the two public Newton solvers. */
export function runAdjointNewton(
  initialParameters: Float64Array, initialStates: Float64Array,
  objective: ConstrainedCostFn | ConstrainedResidualFn, constraint: ConstraintFn,
  options: NewtonOptions, hessianModel: 'approximate' | 'exact'
): AdjointNewtonResult {
  const settings = resolveSettings(options, hessianModel);
  if (initialParameters.length === 0 || initialStates.length === 0 ||
      !initialParameters.every(Number.isFinite) || !initialStates.every(Number.isFinite)) {
    throw new Error('adjointNewton requires nonempty finite parameter and state vectors');
  }
  const logger = new Logger(settings.logLevel, settings.verbose);
  // WHY: A fixed loose state residual can dominate the O(||g||²) decrease near a
  // Newton solution, especially when f_x is nonzero. Respect the user's upper bound.
  const stateRoundoff = STATE_SOLVE_ROUNDOFF_FACTOR * Number.EPSILON * Math.max(1, vectorNorm(initialStates));
  settings.constraintTolerance = Math.min(settings.constraintTolerance,
    Math.max(stateRoundoff, settings.tolerance * settings.tolerance));
  const start = initializeFeasibleReducedSpace(initialParameters, initialStates, objective,
    constraint, settings, logger, settings.algorithmName);
  let current = evaluateReduced(start.parameters, start.states, objective, constraint, settings, logger, start.objectiveKind, start.cost);
  const diagnostics: Diagnostics = {
    hessianEvaluations: 0, dampedSteps: 0, gradientFallbackSteps: 0,
    finalHessianDamping: 0, usedLineSearch: false
  };
  let unchangedCostSteps = 0;
  for (let iteration = 0; iteration <= settings.maxIterations; iteration++) {
    const gradientNorm = vectorNorm(current.reduced.gradient);
    const constraintNorm = vectorNorm(constraint(current.parameters, current.states));
    settings.onIteration?.(iteration, current.cost, new Float64Array(current.parameters));
    if (gradientNorm <= settings.tolerance && constraintNorm <= settings.constraintTolerance) {
      return buildResult(current, constraint, iteration, 'gradientTolerance', diagnostics);
    }
    if (iteration === settings.maxIterations) {
      return buildResult(current, constraint, iteration, 'maxIterations', diagnostics);
    }
    const next = proposeStep(current, objective, constraint, settings, logger, start.objectiveKind, diagnostics);
    if (next === undefined) return buildResult(current, constraint, iteration, 'lineSearchFailed', diagnostics);
    const stepNorm = vectorNorm(Float64Array.from(next.parameters, (value, index) => value - current.parameters[index]));
    const relativeThreshold = settings.stepTolerance * Math.max(1, vectorNorm(current.parameters));
    logger.debug(settings.algorithmName, iteration, 'Accepted Newton-adjoint step', [
      { key: 'Cost:', value: current.cost }, { key: 'Gradient norm:', value: gradientNorm },
      { key: 'Step norm:', value: stepNorm }, { key: 'Hessian damping:', value: diagnostics.finalHessianDamping }
    ]);
    unchangedCostSteps = next.cost === current.cost ? unchangedCostSteps + 1 : 0;
    current = evaluateReduced(next.parameters, next.states, objective, constraint, settings, logger, start.objectiveKind, next.cost);
    if (stepNorm <= relativeThreshold && vectorNorm(current.reduced.gradient) > settings.tolerance) {
      return buildResult(current, constraint, iteration + 1, 'stepStagnation', diagnostics);
    }
    if (unchangedCostSteps >= MAX_UNCHANGED_COST_STEPS && vectorNorm(current.reduced.gradient) > settings.tolerance) {
      return buildResult(current, constraint, iteration + 1, 'costStagnation', diagnostics);
    }
  }
  // The inclusive loop always returns at maxIterations.
  return buildResult(current, constraint, settings.maxIterations, 'maxIterations', diagnostics);
}

/**
 * Reduced-space L-BFGS with true adjoint gradients on restored states.
 * Shares vector-only two-loop recursion with unconstrained L-BFGS.
 */

import type {
  AdjointLbfgsOptions, AdjointLbfgsResult, AdjointLbfgsStopReason,
  ConstrainedCostFn, ConstrainedResidualFn, ConstraintFn
} from './types.js';
import { Logger } from './logger.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { addVectors, dotProduct, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';
import {
  computeCost, computeReducedGradient, createReducedObjective, initializeFeasibleReducedSpace,
  resolveAdjointReducedSpaceSettings, type ReducedGradientEvaluation
} from './adjointReducedSpace.js';
import {
  createEmptyHistory, computeLbfgsSearchDirection, ensureDescentDirectionOrFallback,
  updateHistoryIfCurvatureIsValid
} from './lbfgsUpdate.js';

const ALGORITHM_NAME = 'adjointLbfgs';
const DEFAULT_HISTORY_SIZE = 10;
const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 1;
const DEFAULT_ARMIJO_PARAMETER = 1e-4;
const ROUNDING_ALLOWANCE_FACTOR = 16;

type Iterate = {
  parameters: Float64Array;
  states: Float64Array;
  cost: number;
  constraintNorm: number;
  reduced: ReducedGradientEvaluation;
};

function validateOptions(options: AdjointLbfgsOptions): void {
  const historySize = options.historySize ?? DEFAULT_HISTORY_SIZE;
  if (!Number.isInteger(historySize) || historySize <= 0) {
    throw new Error('historySize must be a positive integer');
  }
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  if (!Number.isInteger(maxIterations) || maxIterations < 0) {
    throw new Error('maxIterations must be a nonnegative integer');
  }
  for (const [name, value] of Object.entries({
    tolerance: options.tolerance, constraintTolerance: options.constraintTolerance, stepSize: options.stepSize
  })) {
    if (value !== undefined && (!Number.isFinite(value) || value <= 0)) {
      throw new Error(`${name} must be finite and positive`);
    }
  }
}

function isFiniteIterate(iterate: Iterate): boolean {
  return Number.isFinite(iterate.cost) && Number.isFinite(iterate.constraintNorm) &&
    Number.isFinite(vectorNorm(iterate.reduced.gradient));
}

function buildResult(
  iterate: Iterate, iterations: number, stopReason: AdjointLbfgsStopReason,
  usedLineSearch: boolean, maxHistoryPairs: number
): AdjointLbfgsResult {
  return {
    finalParameters: iterate.parameters, parameters: iterate.parameters, finalStates: iterate.states,
    finalCost: iterate.cost, finalConstraintNorm: iterate.constraintNorm,
    finalGradientNorm: vectorNorm(iterate.reduced.gradient), iterations,
    converged: stopReason === 'gradientTolerance', stopReason, usedLineSearch, maxHistoryPairs
  };
}

/**
 * Minimize f(p,x(p)), c(p,x(p))=0, storing O(historySize * p.length) correction data.
 * State Jacobians and their dense linear solves retain their existing cost.
 * A small step with a large gradient reports stagnation, never convergence.
 */
export function adjointLbfgs(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointLbfgsOptions = {}
): AdjointLbfgsResult {
  validateOptions(options);
  const settings = resolveAdjointReducedSpaceSettings(options);
  const logger = new Logger(options.logLevel, options.verbose);
  const historySize = options.historySize ?? DEFAULT_HISTORY_SIZE;
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const useLineSearch = options.useLineSearch ?? true;
  const start = initializeFeasibleReducedSpace(
    initialParameters, initialStates, costFunction, constraintFunction, settings, logger, ALGORITHM_NAME
  );
  const evaluateGradient = (parameters: Float64Array, states: Float64Array) => computeReducedGradient(
    parameters, states, costFunction, constraintFunction, settings, logger, ALGORITHM_NAME, start.objectiveKind
  );
  let current: Iterate = { ...start, reduced: evaluateGradient(start.parameters, start.states) };
  const history = createEmptyHistory();
  let maxHistoryPairs = 0;
  let usedLineSearch = false;
  const finish = (iteration: number, reason: AdjointLbfgsStopReason) =>
    buildResult(current, iteration, reason, usedLineSearch, maxHistoryPairs);

  function takeStep(direction: Float64Array): Iterate | AdjointLbfgsStopReason {
    const reducedObjective = createReducedObjective(
      current.parameters, current.states, costFunction, constraintFunction,
      settings, logger, ALGORITHM_NAME, start.objectiveKind, current.reduced.partials
    );
    const stepSize = useLineSearch ? strongWolfeLineSearch(
      reducedObjective.evaluateCost, reducedObjective.evaluateGradient, current.parameters, direction,
      { initialStepSize: DEFAULT_STEP_SIZE, ...options.lineSearchOptions }
    ) : options.stepSize ?? DEFAULT_STEP_SIZE;
    usedLineSearch ||= useLineSearch;
    if (!Number.isFinite(stepSize) || stepSize <= 0) return 'lineSearchFailure';
    const parameters = addVectors(current.parameters, scaleVector(direction, stepSize));
    const states = reducedObjective.restoreStates(parameters);
    if (states === undefined) return 'stateRestorationFailure';
    const cost = computeCost(costFunction, parameters, states, start.objectiveKind);
    // Recheck actual decrease before accepting the restored point.
    const allowance = ROUNDING_ALLOWANCE_FACTOR * Number.EPSILON * Math.max(1, Math.abs(current.cost));
    const armijoBound = current.cost + (options.lineSearchOptions?.wolfeC1 ?? DEFAULT_ARMIJO_PARAMETER) *
      stepSize * dotProduct(current.reduced.gradient, direction);
    if (useLineSearch && cost > armijoBound + allowance) return 'lineSearchFailure';
    const next = { parameters, states, cost, constraintNorm: vectorNorm(constraintFunction(parameters, states)),
      reduced: evaluateGradient(parameters, states) };
    return isFiniteIterate(next) ? next : 'nonFiniteEvaluation';
  }

  for (let iteration = 0; iteration <= maxIterations; iteration++) {
    if (!isFiniteIterate(current)) return finish(iteration, 'nonFiniteEvaluation');
    options.onIteration?.(iteration, current.cost, current.parameters);
    if (vectorNorm(current.reduced.gradient) <= tolerance && current.constraintNorm <= settings.constraintTolerance) {
      return finish(iteration, 'gradientTolerance');
    }
    if (iteration === maxIterations) return finish(iteration, 'maxIterations');
    const direction = ensureDescentDirectionOrFallback(
      current.reduced.gradient, computeLbfgsSearchDirection(current.reduced.gradient, history),
      history, logger, iteration, current.cost, ALGORITHM_NAME
    );
    const next = takeStep(direction);
    if (typeof next === 'string') return finish(iteration, next);
    const step = subtractVectors(next.parameters, current.parameters);
    const gradientChange = subtractVectors(next.reduced.gradient, current.reduced.gradient);
    updateHistoryIfCurvatureIsValid(
      history, historySize, step, gradientChange, logger, iteration, next.cost, ALGORITHM_NAME
    );
    maxHistoryPairs = Math.max(maxHistoryPairs, history.stepVectorHistory.length);
    current = next;
    // WHY: Parameter scaling must not let a tiny step certify a large remaining gradient.
    const roundingStep = ROUNDING_ALLOWANCE_FACTOR * Number.EPSILON * Math.max(1, vectorNorm(current.parameters));
    if (vectorNorm(step) <= roundingStep && vectorNorm(current.reduced.gradient) > tolerance) {
      return finish(iteration + 1, 'stepStagnation');
    }
  }
  return finish(maxIterations, 'maxIterations');
}

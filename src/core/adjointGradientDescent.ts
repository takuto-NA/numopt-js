/**
 * Equality-constrained gradient descent via the adjoint method.
 * Search is only over p; states stay on c(p,x)=0 by tangent prediction plus Newton projection.
 * Entry point: `adjointGradientDescent`.
 */

import type {
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn,
  AdjointGradientDescentOptions,
  AdjointGradientDescentResult
} from './types.js';
import { backtrackingLineSearch } from './lineSearch.js';
import { vectorNorm, scaleVector, addVectors } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence } from './convergence.js';
import { Logger } from './logger.js';
import {
  type AdjointReducedSpaceSettings,
  type ObjectiveKind,
  type ReducedPartials,
  areParametersEqual,
  computeCost,
  computePartialDerivatives,
  computeReducedGradient,
  initializeFeasibleReducedSpace,
  resolveAdjointReducedSpaceSettings,
  restoreFeasibleStates,
  UNPROJECTABLE_TRIAL_COST
} from './adjointReducedSpace.js';

const ADJOINT_ALGORITHM_NAME = 'adjointGradientDescent';
const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 0.01;
const DEFAULT_USE_LINE_SEARCH = true;
const ZERO_STEP_SIZE = 0.0;
const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const MAX_DIMENSION_FOR_DETAILED_LOGGING = 3;

type AdjointRuntimeSettings = AdjointGradientDescentOptions &
  AdjointReducedSpaceSettings & {
    maxIterations: number;
    tolerance: number;
    useLineSearch: boolean;
  };

function resolveAdjointRuntimeSettings(options: AdjointGradientDescentOptions): AdjointRuntimeSettings {
  return {
    ...options,
    ...resolveAdjointReducedSpaceSettings(options),
    maxIterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    tolerance: options.tolerance ?? DEFAULT_TOLERANCE,
    useLineSearch: options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH
  };
}

/**
 * Line-search cost on the restored manifold, not the tangent predictor alone.
 */
function createCostFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointRuntimeSettings,
  logger: Logger,
  objectiveKind: ObjectiveKind,
  cachedPartials?: { dcdx: ReducedPartials['dcdx']; dcdp: ReducedPartials['dcdp'] }
): (params: Float64Array) => number {
  const partials =
    cachedPartials ??
    computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      options,
      objectiveKind,
      ADJOINT_ALGORITHM_NAME
    );
  const { dcdx, dcdp } = partials;

  return (params: Float64Array): number => {
    const trialStates = restoreFeasibleStates(
      currentParameters,
      currentStates,
      params,
      dcdx,
      dcdp,
      constraintFunction,
      options,
      logger,
      ADJOINT_ALGORITHM_NAME
    );
    if (trialStates === undefined) {
      return UNPROJECTABLE_TRIAL_COST;
    }
    return computeCost(costFunction, params, trialStates, objectiveKind);
  };
}

/**
 * Armijo-only gradient wrapper. Restore failure returns the start gradient because
 * backtracking reads cost only. Do not copy this into Strong Wolfe / Adjoint BFGS.
 */
function createGradientFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  currentGradient: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointRuntimeSettings,
  logger: Logger,
  objectiveKind: ObjectiveKind,
  cachedPartials?: ReducedPartials
): (_params: Float64Array) => Float64Array {
  const currentPartials =
    cachedPartials ??
    computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      options,
      objectiveKind,
      ADJOINT_ALGORITHM_NAME
    );
  const { dcdx: currentDcdx, dcdp: currentDcdp } = currentPartials;

  return (trialParams: Float64Array): Float64Array => {
    if (areParametersEqual(trialParams, currentParameters)) {
      return new Float64Array(currentGradient);
    }

    const trialStates = restoreFeasibleStates(
      currentParameters,
      currentStates,
      trialParams,
      currentDcdx,
      currentDcdp,
      constraintFunction,
      options,
      logger,
      ADJOINT_ALGORITHM_NAME
    );
    if (trialStates === undefined) {
      // WHY: Armijo backtracking only reads cost. Returning the start gradient is
      // safe here and must not be copied into Strong Wolfe / Adjoint BFGS.
      return new Float64Array(currentGradient);
    }

    return computeReducedGradient(
      trialParams,
      trialStates,
      costFunction,
      constraintFunction,
      options,
      logger,
      ADJOINT_ALGORITHM_NAME,
      objectiveKind
    ).gradient;
  };
}

function determineStepSize(
  currentGradient: Float64Array,
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  useLineSearch: boolean,
  fixedStepSize: number | undefined,
  options: AdjointRuntimeSettings,
  logger: Logger,
  objectiveKind: ObjectiveKind,
  cachedPartials?: ReducedPartials
): { stepSize: number; usedLineSearch: boolean } {
  if (!useLineSearch || fixedStepSize !== undefined) {
    return { stepSize: fixedStepSize ?? DEFAULT_STEP_SIZE, usedLineSearch: false };
  }

  const partials =
    cachedPartials ??
    computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      options,
      objectiveKind,
      ADJOINT_ALGORITHM_NAME
    );

  const costFnWrapper = createCostFunctionWrapper(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    logger,
    objectiveKind,
    { dcdx: partials.dcdx, dcdp: partials.dcdp }
  );
  const gradientFnWrapper = createGradientFunctionWrapper(
    currentParameters,
    currentStates,
    currentGradient,
    costFunction,
    constraintFunction,
    options,
    logger,
    objectiveKind,
    partials
  );

  const searchDirection = scaleVector(currentGradient, NEGATIVE_GRADIENT_DIRECTION);
  const stepSize = backtrackingLineSearch(
    costFnWrapper,
    gradientFnWrapper,
    currentParameters,
    searchDirection
  );

  return { stepSize, usedLineSearch: true };
}

function checkConstraintViolation(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  iteration: number,
  logger: Logger
): { constraint: Float64Array; constraintNorm: number } {
  const constraint = constraintFunction(currentParameters, currentStates);
  const constraintNorm = vectorNorm(constraint);
  if (constraintNorm > constraintTolerance) {
    logger.warn('adjointGradientDescent', iteration, 'Constraint violation detected', [
      { key: '||c(p,x)||:', value: constraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }
  return { constraint, constraintNorm };
}

function addArrayToLogDetails(
  details: Array<{ key: string; value: number }>,
  array: Float64Array,
  prefix: string
): void {
  for (let index = 0; index < array.length; index++) {
    details.push({ key: `${prefix}[${index}]:`, value: array[index] });
  }
}

function createProgressLogDetails(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraint: Float64Array,
  currentCost: number,
  gradientNorm: number,
  stepSize: number,
  constraintNorm: number
): Array<{ key: string; value: number }> {
  const logDetails: Array<{ key: string; value: number }> = [
    { key: 'Cost:', value: currentCost },
    { key: 'Gradient norm:', value: gradientNorm },
    { key: 'Step size:', value: stepSize },
    { key: 'Constraint norm:', value: constraintNorm }
  ];

  if (
    currentParameters.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING &&
    currentStates.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING
  ) {
    addArrayToLogDetails(logDetails, currentParameters, 'p');
    addArrayToLogDetails(logDetails, currentStates, 'x');
    if (constraint.length <= MAX_DIMENSION_FOR_DETAILED_LOGGING) {
      addArrayToLogDetails(logDetails, constraint, 'c');
    }
  }

  return logDetails;
}

function buildConstrainedResult(
  parameters: Float64Array,
  states: Float64Array,
  iterations: number,
  converged: boolean,
  cost: number,
  gradientNorm: number,
  constraintNorm: number,
  usedLineSearch: boolean
): AdjointGradientDescentResult {
  return {
    finalParameters: parameters,
    parameters,
    iterations,
    converged,
    finalCost: cost,
    finalGradientNorm: gradientNorm,
    usedLineSearch,
    finalStates: states,
    finalConstraintNorm: constraintNorm
  };
}

/**
 * Performs adjoint gradient descent optimization to minimize a constrained cost function.
 *
 * Supports both cost functions f(p,x) and residual functions r(p,x) where f = 1/2 r^T r.
 */
export function adjointGradientDescent(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointGradientDescentOptions = {}
): AdjointGradientDescentResult {
  const settings = resolveAdjointRuntimeSettings(options);
  const logger = new Logger(settings.logLevel, settings.verbose);
  const start = initializeFeasibleReducedSpace(
    initialParameters,
    initialStates,
    costFunction,
    constraintFunction,
    settings,
    logger,
    ADJOINT_ALGORITHM_NAME
  );

  let currentParameters = start.parameters;
  let currentStates = start.states;
  const objectiveKind = start.objectiveKind;
  let currentCost = start.cost;
  let usedLineSearch = false;

  for (let iteration = 0; iteration < settings.maxIterations; iteration++) {
    const { constraint, constraintNorm } = checkConstraintViolation(
      currentParameters,
      currentStates,
      constraintFunction,
      settings.constraintTolerance,
      iteration,
      logger
    );

    const reduced = computeReducedGradient(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      settings,
      logger,
      ADJOINT_ALGORITHM_NAME,
      objectiveKind
    );
    const adjointGradient = reduced.gradient;
    const partials = reduced.partials;
    const gradientNorm = vectorNorm(adjointGradient);

    if (settings.onIteration) {
      settings.onIteration(iteration, currentCost, currentParameters);
    }

    if (
      constraintNorm <= settings.constraintTolerance &&
      checkGradientConvergence(gradientNorm, settings.tolerance, iteration)
    ) {
      logger.info('adjointGradientDescent', iteration, 'Converged', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Constraint norm:', value: constraintNorm }
      ]);
      return buildConstrainedResult(
        currentParameters,
        currentStates,
        iteration,
        true,
        currentCost,
        gradientNorm,
        constraintNorm,
        usedLineSearch
      );
    }

    const stepSizeResult = determineStepSize(
      adjointGradient,
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      settings.useLineSearch,
      settings.stepSize,
      settings,
      logger,
      objectiveKind,
      partials
    );

    if (stepSizeResult.stepSize === ZERO_STEP_SIZE) {
      logger.warn('adjointGradientDescent', iteration, 'Line search failed', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm }
      ]);
      return buildConstrainedResult(
        currentParameters,
        currentStates,
        iteration,
        false,
        currentCost,
        gradientNorm,
        constraintNorm,
        true
      );
    }

    usedLineSearch = usedLineSearch || stepSizeResult.usedLineSearch;
    const parameterStep = scaleVector(
      adjointGradient,
      NEGATIVE_GRADIENT_DIRECTION * stepSizeResult.stepSize
    );
    const newParameters = addVectors(currentParameters, parameterStep);
    const newStates = restoreFeasibleStates(
      currentParameters,
      currentStates,
      newParameters,
      partials.dcdx,
      partials.dcdp,
      constraintFunction,
      settings,
      logger,
      ADJOINT_ALGORITHM_NAME
    );
    if (newStates === undefined) {
      logger.warn('adjointGradientDescent', iteration, 'Failed to restore feasible states after the trial step', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Step size:', value: stepSizeResult.stepSize }
      ]);
      return buildConstrainedResult(
        currentParameters,
        currentStates,
        iteration,
        false,
        currentCost,
        gradientNorm,
        constraintNorm,
        usedLineSearch
      );
    }
    const newCost = computeCost(costFunction, newParameters, newStates, objectiveKind);
    const stepNorm = vectorNorm(parameterStep);

    if (
      constraintNorm <= settings.constraintTolerance &&
      checkStepSizeConvergence(stepNorm, settings.tolerance, iteration)
    ) {
      logger.info('adjointGradientDescent', iteration, 'Converged', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Step size:', value: stepNorm }
      ]);
      return buildConstrainedResult(
        currentParameters,
        currentStates,
        iteration,
        true,
        currentCost,
        gradientNorm,
        constraintNorm,
        usedLineSearch
      );
    }

    logger.debug(
      'adjointGradientDescent',
      iteration,
      'Progress',
      createProgressLogDetails(
        currentParameters,
        currentStates,
        constraint,
        currentCost,
        gradientNorm,
        stepSizeResult.stepSize,
        constraintNorm
      )
    );

    currentParameters = new Float64Array(newParameters);
    currentStates = new Float64Array(newStates);
    currentCost = newCost;
  }

  const finalReduced = computeReducedGradient(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    settings,
    logger,
    ADJOINT_ALGORITHM_NAME,
    objectiveKind
  );
  const finalGradientNorm = vectorNorm(finalReduced.gradient);
  const finalConstraintNorm = vectorNorm(constraintFunction(currentParameters, currentStates));

  logger.warn('adjointGradientDescent', undefined, 'Maximum iterations reached', [
    { key: 'Iterations:', value: settings.maxIterations },
    { key: 'Final cost:', value: currentCost },
    { key: 'Final gradient norm:', value: finalGradientNorm },
    { key: 'Final constraint norm:', value: finalConstraintNorm }
  ]);

  return buildConstrainedResult(
    currentParameters,
    currentStates,
    settings.maxIterations,
    false,
    currentCost,
    finalGradientNorm,
    finalConstraintNorm,
    usedLineSearch
  );
}

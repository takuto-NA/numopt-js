/**
 * Reduced-space BFGS: standard dense BFGS on f̃(p)=f(p,x(p)).
 * Gradient is the existing adjoint; Hessian approximation lives in p-space only.
 * Entry point: `adjointBfgs`. Does not call the public unconstrained `bfgs`.
 */

import type {
  AdjointBfgsOptions,
  AdjointBfgsResult,
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn
} from './types.js';
import { strongWolfeLineSearch } from './lineSearch.js';
import { Logger } from './logger.js';
import { checkGradientConvergence, checkStepSizeConvergence } from './convergence.js';
import { addVectors, scaleVector, subtractVectors, vectorNorm } from '../utils/matrix.js';
import {
  type AdjointReducedSpaceSettings,
  computeCost,
  computeReducedGradient,
  createReducedObjective,
  initializeFeasibleReducedSpace,
  resolveAdjointReducedSpaceSettings,
  restoreFeasibleStates
} from './adjointReducedSpace.js';
import {
  computeBfgsSearchDirection,
  createIdentityInverseHessian,
  ensureDescentDirectionOrFallback,
  updateInverseHessianApproximation
} from './bfgsUpdate.js';

const ADJOINT_BFGS_ALGORITHM_NAME = 'adjointBfgs';
const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_FIXED_STEP_SIZE = 1.0;
const INVALID_STEP_SIZE = 0.0;
const DEFAULT_STRONG_WOLFE_INITIAL_STEP_SIZE = 1.0;
const MAX_DIMENSION_FOR_DETAILED_LOGGING = 3;

type AdjointBfgsRuntimeSettings = AdjointBfgsOptions &
  AdjointReducedSpaceSettings & {
    maxIterations: number;
    tolerance: number;
    useLineSearch: boolean;
    stepSize: number;
  };

function resolveAdjointBfgsRuntimeSettings(options: AdjointBfgsOptions): AdjointBfgsRuntimeSettings {
  return {
    ...options,
    ...resolveAdjointReducedSpaceSettings(options),
    maxIterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    tolerance: options.tolerance ?? DEFAULT_TOLERANCE,
    useLineSearch: options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH,
    stepSize: options.stepSize ?? DEFAULT_FIXED_STEP_SIZE
  };
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
    logger.warn(ADJOINT_BFGS_ALGORITHM_NAME, iteration, 'Constraint violation detected', [
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
): AdjointBfgsResult {
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
 * Performs reduced-space BFGS on f̃(p)=f(p,x(p)).
 *
 * Search stays in p. Trial and accepted gradients are adjoint evaluations
 * on restored states. Does not call the public unconstrained `bfgs`.
 *
 * Supports both cost functions f(p,x) and residual functions r(p,x) where f = 1/2 r^T r.
 * For small-residual least squares, prefer constrained Gauss–Newton / LM.
 */
export function adjointBfgs(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointBfgsOptions = {}
): AdjointBfgsResult {
  const settings = resolveAdjointBfgsRuntimeSettings(options);
  const logger = new Logger(settings.logLevel, settings.verbose);
  const start = initializeFeasibleReducedSpace(
    initialParameters,
    initialStates,
    costFunction,
    constraintFunction,
    settings,
    logger,
    ADJOINT_BFGS_ALGORITHM_NAME
  );

  let currentParameters = start.parameters;
  let currentStates = start.states;
  const objectiveKind = start.objectiveKind;
  let currentCost = start.cost;
  let inverseHessianApproximation = createIdentityInverseHessian(currentParameters.length);
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
      ADJOINT_BFGS_ALGORITHM_NAME,
      objectiveKind
    );
    const currentGradient = reduced.gradient;
    const currentPartials = reduced.partials;
    const gradientNorm = vectorNorm(currentGradient);

    if (settings.onIteration) {
      settings.onIteration(iteration, currentCost, currentParameters);
    }

    if (
      constraintNorm <= settings.constraintTolerance &&
      checkGradientConvergence(gradientNorm, settings.tolerance, iteration)
    ) {
      logger.info(ADJOINT_BFGS_ALGORITHM_NAME, iteration, 'Converged', [
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

    const proposedSearchDirection = computeBfgsSearchDirection(
      inverseHessianApproximation,
      currentGradient
    );
    const descentResult = ensureDescentDirectionOrFallback(
      currentGradient,
      proposedSearchDirection,
      inverseHessianApproximation,
      logger,
      iteration,
      currentCost,
      ADJOINT_BFGS_ALGORITHM_NAME
    );
    const searchDirection = descentResult.searchDirection;
    inverseHessianApproximation = descentResult.inverseHessianApproximation;

    let stepSize: number;
    if (settings.useLineSearch) {
      const reducedObjective = createReducedObjective(
        currentParameters,
        currentStates,
        costFunction,
        constraintFunction,
        settings,
        logger,
        ADJOINT_BFGS_ALGORITHM_NAME,
        objectiveKind,
        { dcdx: currentPartials.dcdx, dcdp: currentPartials.dcdp }
      );
      stepSize = strongWolfeLineSearch(
        reducedObjective.evaluateCost,
        reducedObjective.evaluateGradient,
        currentParameters,
        searchDirection,
        {
          // WHY: Nocedal §6.1 — try the unit quasi-Newton step first. The shared
          // Strong Wolfe default of 1/||∇f̃|| shrinks that step when ||∇f̃|| is large
          // (implicit chain), so H never gets to take Newton-like steps.
          initialStepSize: DEFAULT_STRONG_WOLFE_INITIAL_STEP_SIZE,
          ...settings.lineSearchOptions
        }
      );
      usedLineSearch = true;
    } else {
      stepSize = settings.stepSize;
    }

    if (stepSize === INVALID_STEP_SIZE) {
      logger.warn(ADJOINT_BFGS_ALGORITHM_NAME, iteration, 'Line search failed', [
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
        usedLineSearch
      );
    }

    const parameterStep = scaleVector(searchDirection, stepSize);
    const newParameters = addVectors(currentParameters, parameterStep);
    const newStates = restoreFeasibleStates(
      currentParameters,
      currentStates,
      newParameters,
      currentPartials.dcdx,
      currentPartials.dcdp,
      constraintFunction,
      settings,
      logger,
      ADJOINT_BFGS_ALGORITHM_NAME
    );
    if (newStates === undefined) {
      logger.warn(ADJOINT_BFGS_ALGORITHM_NAME, iteration, 'Failed to restore feasible states after the trial step', [
        { key: 'Cost:', value: currentCost },
        { key: 'Gradient norm:', value: gradientNorm },
        { key: 'Step size:', value: stepSize }
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

    // WHY: y must be a true reduced-gradient difference on the manifold, not a frozen predictor.
    const newCost = computeCost(costFunction, newParameters, newStates, objectiveKind);
    const nextReduced = computeReducedGradient(
      newParameters,
      newStates,
      costFunction,
      constraintFunction,
      settings,
      logger,
      ADJOINT_BFGS_ALGORITHM_NAME,
      objectiveKind
    );
    const stepVector = subtractVectors(newParameters, currentParameters);
    const stepNorm = vectorNorm(stepVector);
    const gradientChangeVector = subtractVectors(nextReduced.gradient, currentGradient);

    if (
      constraintNorm <= settings.constraintTolerance &&
      checkStepSizeConvergence(stepNorm, settings.tolerance, iteration)
    ) {
      logger.info(ADJOINT_BFGS_ALGORITHM_NAME, iteration, 'Converged', [
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

    inverseHessianApproximation = updateInverseHessianApproximation(
      inverseHessianApproximation,
      stepVector,
      gradientChangeVector,
      logger,
      iteration,
      newCost,
      ADJOINT_BFGS_ALGORITHM_NAME
    );

    logger.debug(
      ADJOINT_BFGS_ALGORITHM_NAME,
      iteration,
      'Progress',
      createProgressLogDetails(
        currentParameters,
        currentStates,
        constraint,
        currentCost,
        gradientNorm,
        stepSize,
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
    ADJOINT_BFGS_ALGORITHM_NAME,
    objectiveKind
  );
  const finalGradientNorm = vectorNorm(finalReduced.gradient);
  const finalConstraintNorm = vectorNorm(constraintFunction(currentParameters, currentStates));

  logger.warn(ADJOINT_BFGS_ALGORITHM_NAME, undefined, 'Maximum iterations reached', [
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

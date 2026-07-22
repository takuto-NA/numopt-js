/**
 * This file implements the adjoint method for constrained optimization problems.
 * 
 * The adjoint method efficiently computes gradients for constrained optimization
 * by solving for an adjoint variable λ instead of explicitly inverting matrices.
 * 
 * Mathematical background:
 * - For constraint c(p, x) = 0, the implicit function theorem gives:
 *   df/dp = ∂f/∂p - ∂f/∂x (∂c/∂x)^-1 ∂c/∂p
 * - Instead of computing (∂c/∂x)^-1 ∂c/∂p explicitly, we solve:
 *   (∂c/∂x)^T λ = (∂f/∂x)^T
 *   Then: df/dp = ∂f/∂p - λ^T ∂c/∂p
 * - This requires solving only one linear system per iteration instead of
 *   paramCount systems, making it much more efficient.
 * 
 * For residual functions r(p, x) where f = 1/2 r^T r:
 * - Solve: (∂c/∂x)^T λ = r^T ∂r/∂x
 * - Then: df/dp = r^T ∂r/∂p - λ^T ∂c/∂p
 * 
 * References:
 * - Nocedal & Wright, "Numerical Optimization" (2nd ed.), Chapter 12 (constrained optimization)
 * - Adjoint method is widely used in optimal control and shape optimization
 * 
 * Role in system:
 * - Equality-constrained gradient descent via the adjoint method
 * - Supports scalar cost or residual objectives (residual → f = 1/2 ||r||²)
 * - Uses finite differences or analytical derivatives; linear solves live in constrainedUtils
 *
 * For first-time readers:
 * - Start with adjointGradientDescent
 * - Objective kind is resolved once at entry, then threaded through helpers
 * - States are updated with a first-order constraint correction after each parameter step
 */

import { Matrix } from 'ml-matrix';
import type {
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn,
  AdjointGradientDescentOptions,
  AdjointGradientDescentResult
} from './types.js';
import {
  finiteDiffPartialP,
  finiteDiffPartialX,
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX,
  finiteDiffResidualPartialP,
  finiteDiffResidualPartialX
} from './finiteDiff.js';
import { backtrackingLineSearch } from './lineSearch.js';
import { vectorNorm, scaleVector, addVectors, subtractVectors } from '../utils/matrix.js';
import { checkGradientConvergence, checkStepSizeConvergence } from './convergence.js';
import { Logger } from './logger.js';
import { float64ArrayToMatrix } from '../utils/matrix.js';
import {
  solveAdjointEquation,
  updateStates,
  validateInitialConditions
} from './constrainedUtils.js';

const DEFAULT_ADJOINT_REGULARIZATION = 0.0;
const ADJOINT_ALGORITHM_NAME = 'adjointGradientDescent';

const DEFAULT_MAX_ITERATIONS = 1000;
const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE = 0.01;
const DEFAULT_USE_LINE_SEARCH = true;
const DEFAULT_CONSTRAINT_TOLERANCE = 1e-6;
const DEFAULT_STEP_SIZE_P = 1e-6;
const DEFAULT_STEP_SIZE_X = 1e-6;
const ZERO_STEP_SIZE = 0.0;
const NEGATIVE_GRADIENT_DIRECTION = -1.0;
const RESIDUAL_COST_COEFFICIENT = 0.5; // Coefficient for residual cost: f = 1/2 r^T r
const MAX_DIMENSION_FOR_DETAILED_LOGGING = 3; // Maximum dimension for detailed parameter/state logging
const FLOATING_POINT_EQUALITY_TOLERANCE = 1e-15; // Tolerance for floating point equality comparisons

/**
 * Entry defaults resolved once. Internals must not re-default these fields.
 */
type AdjointRuntimeSettings = AdjointGradientDescentOptions & {
  maxIterations: number;
  tolerance: number;
  useLineSearch: boolean;
  constraintTolerance: number;
  stepSizeP: number;
  stepSizeX: number;
  regularization: number;
};

function resolveAdjointRuntimeSettings(options: AdjointGradientDescentOptions): AdjointRuntimeSettings {
  return {
    ...options,
    maxIterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    tolerance: options.tolerance ?? DEFAULT_TOLERANCE,
    useLineSearch: options.useLineSearch ?? DEFAULT_USE_LINE_SEARCH,
    constraintTolerance: options.constraintTolerance ?? DEFAULT_CONSTRAINT_TOLERANCE,
    stepSizeP: options.stepSizeP ?? DEFAULT_STEP_SIZE_P,
    stepSizeX: options.stepSizeX ?? DEFAULT_STEP_SIZE_X,
    regularization: options.regularization ?? DEFAULT_ADJOINT_REGULARIZATION
  };
}

function rowVectorToFloat64Array(matrix: Matrix): Float64Array {
  if (matrix.rows !== 1) {
    throw new Error('Expected row vector (1 x n)');
  }
  const result = new Float64Array(matrix.columns);
  for (let column = 0; column < matrix.columns; column++) {
    result[column] = matrix.get(0, column);
  }
  return result;
}

/**
 * Computes the adjoint gradient: df/dp = ∂f/∂p - λ^T ∂c/∂p
 */
function computeAdjointGradient(
  dfdp: Float64Array,
  lambda: Float64Array,
  dcdp: Matrix
): Float64Array {
  const lambdaTdcdp = float64ArrayToMatrix(lambda).transpose().mmul(dcdp);
  return subtractVectors(dfdp, rowVectorToFloat64Array(lambdaTdcdp));
}


type ObjectiveKind = 'cost' | 'residual';

/**
 * Sniff objective kind once from a sample evaluation at the entry point.
 * Helpers must take the resolved kind — never re-evaluate just to branch.
 */
function resolveObjectiveKind(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  parameters: Float64Array,
  states: Float64Array
): ObjectiveKind {
  const sample = costFunction(parameters, states);
  return sample instanceof Float64Array ? 'residual' : 'cost';
}

/**
 * Computes cost from either a cost function or residual function.
 * For residual functions r(p,x), computes f = 1/2 r^T r.
 */
function computeCost(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  parameters: Float64Array,
  states: Float64Array,
  objectiveKind: ObjectiveKind
): number {
  if (objectiveKind === 'residual') {
    const residual = costFunction(parameters, states) as Float64Array;
    const residualNorm = vectorNorm(residual);
    return RESIDUAL_COST_COEFFICIENT * residualNorm * residualNorm;
  }

  return costFunction(parameters, states) as number;
}

/**
 * Computes gradient from residual function: df/dp = r^T ∂r/∂p
 * This formula comes from the chain rule applied to the residual cost function f = 1/2 r^T r.
 */
function computeGradientFromResidual(
  residual: Float64Array,
  derivativeMatrix: Matrix
): Float64Array {
  const residualMatrix = float64ArrayToMatrix(residual);
  const gradientMatrix = residualMatrix.transpose().mmul(derivativeMatrix);
  return rowVectorToFloat64Array(gradientMatrix);
}

/**
 * Computes ∂f/∂p or ∂r/∂p using analytical functions or finite differences.
 */
function computeDfdp(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  options: AdjointRuntimeSettings,
  objectiveKind: ObjectiveKind
): Float64Array {
  const stepSizeP = options.stepSizeP;

  if (options.dfdp) {
    return options.dfdp(parameters, states);
  }

  if (objectiveKind === 'residual') {
    const residualFunction = costFunction as ConstrainedResidualFn;
    const derivativeResidualPartialP = finiteDiffResidualPartialP(
      parameters,
      states,
      residualFunction,
      { stepSize: stepSizeP }
    );
    const residual = residualFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialP);
  }

  return finiteDiffPartialP(
    parameters,
    states,
    costFunction as ConstrainedCostFn,
    { stepSize: stepSizeP }
  );
}

/**
 * Computes ∂f/∂x or ∂r/∂x using analytical functions or finite differences.
 */
function computeDfdx(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  options: AdjointRuntimeSettings,
  objectiveKind: ObjectiveKind
): Float64Array {
  const stepSizeX = options.stepSizeX;

  if (options.dfdx) {
    return options.dfdx(parameters, states);
  }

  if (objectiveKind === 'residual') {
    const residualFunction = costFunction as ConstrainedResidualFn;
    const derivativeResidualPartialX = finiteDiffResidualPartialX(
      parameters,
      states,
      residualFunction,
      { stepSize: stepSizeX }
    );
    const residual = residualFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialX);
  }

  return finiteDiffPartialX(
    parameters,
    states,
    costFunction as ConstrainedCostFn,
    { stepSize: stepSizeX }
  );
}

/**
 * Computes partial derivatives using analytical functions or finite differences.
 */
function computePartialDerivatives(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointRuntimeSettings,
  objectiveKind: ObjectiveKind
): {
  dfdp: Float64Array;
  dfdx: Float64Array;
  dcdp: Matrix;
  dcdx: Matrix;
} {
  const stepSizeP = options.stepSizeP;
  const stepSizeX = options.stepSizeX;
  
  const dfdp = computeDfdp(parameters, states, costFunction, options, objectiveKind);
  const dfdx = computeDfdx(parameters, states, costFunction, options, objectiveKind);

  // Compute ∂c/∂p: needed for adjoint gradient computation (df/dp = ∂f/∂p - λ^T ∂c/∂p)
  const dcdp = options.dcdp
    ? options.dcdp(parameters, states)
    : finiteDiffConstraintPartialP(parameters, states, constraintFunction, { stepSize: stepSizeP });

  // Compute ∂c/∂x: needed to solve adjoint equation (∂c/∂x)^T λ = (∂f/∂x)^T
  const dcdx = options.dcdx
    ? options.dcdx(parameters, states)
    : finiteDiffConstraintPartialX(parameters, states, constraintFunction, { stepSize: stepSizeX });

  return { dfdp, dfdx, dcdp, dcdx };
}


/**
 * Creates a cost function wrapper for line search that updates states using linear approximation.
 * Partial derivatives are pre-computed and cached to avoid recomputation during line search.
 */
function createCostFunctionWrapper(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  options: AdjointRuntimeSettings,
  logger: Logger,
  objectiveKind: ObjectiveKind,
  cachedPartials?: { dcdx: Matrix; dcdp: Matrix }
): (params: Float64Array) => number {
  const partials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    objectiveKind
  );
  const { dcdx, dcdp } = partials;

  return (params: Float64Array): number => {
    const deltaP = subtractVectors(params, currentParameters);
    const newStates = updateStates(
      currentStates,
      dcdx,
      dcdp,
      deltaP,
      logger,
      ADJOINT_ALGORITHM_NAME,
      options.regularization
    );
    return computeCost(costFunction, params, newStates, objectiveKind);
  };
}

/**
 * Checks if two parameter arrays are equal within floating point tolerance.
 * Used to avoid redundant gradient computation when line search evaluates at the starting point.
 */
function areParametersEqual(
  parameters1: Float64Array,
  parameters2: Float64Array
): boolean {
  if (parameters1.length !== parameters2.length) {
    return false;
  }
  for (let i = 0; i < parameters1.length; i++) {
    if (Math.abs(parameters1[i] - parameters2[i]) > FLOATING_POINT_EQUALITY_TOLERANCE) {
      return false;
    }
  }
  return true;
}

/**
 * Creates a gradient function wrapper for line search.
 * For each trial parameter, updates states and computes gradient at that point.
 * Uses pre-computed currentGradient for current point to ensure consistency.
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
  cachedPartials?: { dfdp: Float64Array; dfdx: Float64Array; dcdp: Matrix; dcdx: Matrix }
): (_params: Float64Array) => Float64Array {
  // Pre-compute partial derivatives once for state updates
  const currentPartials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    objectiveKind
  );
  const { dcdx: currentDcdx, dcdp: currentDcdp } = currentPartials;

  return (trialParams: Float64Array): Float64Array => {
    // Return pre-computed gradient if parameters haven't changed to avoid redundant computation.
    // Line search evaluates gradient at the starting point for direction derivative calculation.
    if (areParametersEqual(trialParams, currentParameters)) {
      return new Float64Array(currentGradient);
    }
    
    // For different trial parameters, update states to maintain constraints and compute gradient.
    // We use linear approximation for efficiency: solving full nonlinear constraints for each trial would be too slow.
    const deltaP = subtractVectors(trialParams, currentParameters);
    const trialStates = updateStates(
      currentStates,
      currentDcdx,
      currentDcdp,
      deltaP,
      logger,
      ADJOINT_ALGORITHM_NAME,
      options.regularization
    );
    
    // Compute gradient at trial point to evaluate search direction quality in line search.
    const trialPartials = computePartialDerivatives(
      trialParams,
      trialStates,
      costFunction,
      constraintFunction,
      options,
      objectiveKind
    );
    const lambda = solveAdjointEquation(
      trialPartials.dcdx,
      trialPartials.dfdx,
      logger,
      ADJOINT_ALGORITHM_NAME,
      options.regularization
    );
    return computeAdjointGradient(trialPartials.dfdp, lambda, trialPartials.dcdp);
  };
}

/**
 * Determines the step size for gradient descent iteration.
 */
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
  cachedPartials?: { dfdp: Float64Array; dfdx: Float64Array; dcdp: Matrix; dcdx: Matrix }
): { stepSize: number; usedLineSearch: boolean } {
  if (!useLineSearch || fixedStepSize !== undefined) {
    return { stepSize: fixedStepSize ?? DEFAULT_STEP_SIZE, usedLineSearch: false };
  }

  // Pre-compute partial derivatives once and reuse in both wrappers
  const partials = cachedPartials ?? computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    options,
    objectiveKind
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

/**
 * Checks constraint violation and logs warning if needed.
 */
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

  validateInitialConditions(
    initialParameters,
    initialStates,
    constraintFunction,
    settings.constraintTolerance,
    logger,
    ADJOINT_ALGORITHM_NAME
  );

  let currentParameters = new Float64Array(initialParameters);
  let currentStates = new Float64Array(initialStates);
  const objectiveKind = resolveObjectiveKind(costFunction, currentParameters, currentStates);
  let currentCost = computeCost(costFunction, currentParameters, currentStates, objectiveKind);
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

    const partials = computePartialDerivatives(
      currentParameters,
      currentStates,
      costFunction,
      constraintFunction,
      settings,
      objectiveKind
    );
    const lambda = solveAdjointEquation(
      partials.dcdx,
      partials.dfdx,
      logger,
      ADJOINT_ALGORITHM_NAME,
      settings.regularization
    );
    const adjointGradient = computeAdjointGradient(partials.dfdp, lambda, partials.dcdp);
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
    const newStates = updateStates(
      currentStates,
      partials.dcdx,
      partials.dcdp,
      parameterStep,
      logger,
      ADJOINT_ALGORITHM_NAME,
      settings.regularization
    );
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

  const finalPartials = computePartialDerivatives(
    currentParameters,
    currentStates,
    costFunction,
    constraintFunction,
    settings,
    objectiveKind
  );
  const finalLambda = solveAdjointEquation(
    finalPartials.dcdx,
    finalPartials.dfdx,
    logger,
    ADJOINT_ALGORITHM_NAME,
    settings.regularization
  );
  const finalGradientNorm = vectorNorm(
    computeAdjointGradient(finalPartials.dfdp, finalLambda, finalPartials.dcdp)
  );
  const finalConstraintNorm = vectorNorm(
    constraintFunction(currentParameters, currentStates)
  );

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

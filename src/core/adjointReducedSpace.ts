/**
 * Shared reduced-space helpers for adjoint solvers.
 * Owns square implicit-state checks, partials, restore, and reduced (f̃, ∇f̃).
 * Does not own Armijo frozen-Jacobian line-search wrappers.
 */

import { Matrix } from 'ml-matrix';
import type {
  AdjointDerivativeOptions,
  ConstrainedCostFn,
  ConstrainedResidualFn,
  ConstraintFn,
  CostFn,
  GradientFn
} from './types.js';
import {
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX,
  finiteDiffPartialP,
  finiteDiffPartialX,
  finiteDiffResidualPartialP,
  finiteDiffResidualPartialX
} from './finiteDiff.js';
import { Logger } from './logger.js';
import {
  projectStatesToConstraints,
  solveAdjointEquation,
  updateStates,
  validateInitialConditions
} from './constrainedUtils.js';
import { float64ArrayToMatrix, subtractVectors, vectorNorm } from '../utils/matrix.js';

export const DEFAULT_ADJOINT_REGULARIZATION = 0.0;
export const DEFAULT_CONSTRAINT_TOLERANCE = 1e-6;
export const DEFAULT_STEP_SIZE_P = 1e-6;
export const DEFAULT_STEP_SIZE_X = 1e-6;
export const INITIAL_PROJECTION_ITERATIONS = 8;
export const TRIAL_PROJECTION_ITERATIONS = 3;
export const UNPROJECTABLE_TRIAL_COST = Number.POSITIVE_INFINITY;
export const RESIDUAL_COST_COEFFICIENT = 0.5;
const FLOATING_POINT_EQUALITY_TOLERANCE = 1e-15;

export type ObjectiveKind = 'cost' | 'residual';

export type AdjointReducedSpaceSettings = AdjointDerivativeOptions & {
  constraintTolerance: number;
  stepSizeP: number;
  stepSizeX: number;
  regularization: number;
};

export type ConstraintPartials = {
  dcdp: Matrix;
  dcdx: Matrix;
};

export type ReducedPartials = {
  dfdp: Float64Array;
  dfdx: Float64Array;
  dcdp: Matrix;
  dcdx: Matrix;
};

export type FeasibleReducedSpace = {
  parameters: Float64Array;
  states: Float64Array;
  objectiveKind: ObjectiveKind;
  cost: number;
  constraintNorm: number;
};

export type ReducedGradientEvaluation = {
  gradient: Float64Array;
  partials: ReducedPartials;
};

export function resolveAdjointReducedSpaceSettings(
  options: AdjointDerivativeOptions
): AdjointReducedSpaceSettings {
  return {
    ...options,
    constraintTolerance: options.constraintTolerance ?? DEFAULT_CONSTRAINT_TOLERANCE,
    stepSizeP: options.stepSizeP ?? DEFAULT_STEP_SIZE_P,
    stepSizeX: options.stepSizeX ?? DEFAULT_STEP_SIZE_X,
    regularization: options.regularization ?? DEFAULT_ADJOINT_REGULARIZATION
  };
}

export function areParametersEqual(
  leftParameters: Float64Array,
  rightParameters: Float64Array
): boolean {
  if (leftParameters.length !== rightParameters.length) {
    return false;
  }
  for (let index = 0; index < leftParameters.length; index++) {
    if (Math.abs(leftParameters[index] - rightParameters[index]) > FLOATING_POINT_EQUALITY_TOLERANCE) {
      return false;
    }
  }
  return true;
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

export function computeAdjointGradient(
  dfdp: Float64Array,
  lambda: Float64Array,
  dcdp: Matrix
): Float64Array {
  const lambdaTdcdp = float64ArrayToMatrix(lambda).transpose().mmul(dcdp);
  return subtractVectors(dfdp, rowVectorToFloat64Array(lambdaTdcdp));
}

export function assertSquareImplicitStateSystem(
  constraintCount: number,
  stateCount: number,
  parameterCount: number,
  algorithmName: string
): void {
  if (constraintCount !== stateCount) {
    throw new Error(
      `${algorithmName} requires a square implicit-state system: ` +
        `constraintCount must equal stateCount so that x(p) is locally unique. ` +
        `Got constraintCount=${constraintCount}, stateCount=${stateCount}, parameterCount=${parameterCount}.`
    );
  }
}

export function assertSquareConstraintJacobians(
  dcdx: Matrix,
  dcdp: Matrix,
  parameters: Float64Array,
  states: Float64Array,
  algorithmName: string
): void {
  if (dcdx.rows !== states.length || dcdx.columns !== states.length) {
    throw new Error(
      `${algorithmName} requires a square constraint Jacobian ∂c/∂x ` +
        `(constraintCount === stateCount). ` +
        `Got ∂c/∂x as ${dcdx.rows} × ${dcdx.columns} with stateCount=${states.length}.`
    );
  }
  if (dcdp.rows !== states.length || dcdp.columns !== parameters.length) {
    throw new Error(
      `${algorithmName} expected ∂c/∂p to be ${states.length} × ${parameters.length}. ` +
        `Got ${dcdp.rows} × ${dcdp.columns}.`
    );
  }
}

export function resolveConstraintJacobians(
  parameters: Float64Array,
  states: Float64Array,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  algorithmName: string
): ConstraintPartials {
  const dcdp = settings.dcdp
    ? settings.dcdp(parameters, states)
    : finiteDiffConstraintPartialP(parameters, states, constraintFunction, { stepSize: settings.stepSizeP });
  const dcdx = settings.dcdx
    ? settings.dcdx(parameters, states)
    : finiteDiffConstraintPartialX(parameters, states, constraintFunction, { stepSize: settings.stepSizeX });
  assertSquareConstraintJacobians(dcdx, dcdp, parameters, states, algorithmName);
  return { dcdp, dcdx };
}

export function projectAndVerify(
  parameters: Float64Array,
  states: Float64Array,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  logger: Logger,
  algorithmName: string,
  maxIterations: number
): { projectedStates: Float64Array | undefined; constraintNorm: number } {
  const projectedStates = projectStatesToConstraints(
    parameters,
    states,
    constraintFunction,
    settings.stepSizeX,
    settings.constraintTolerance,
    logger,
    algorithmName,
    maxIterations,
    {
      dcdx: settings.dcdx,
      regularization: settings.regularization
    }
  );
  const constraintNorm = vectorNorm(constraintFunction(parameters, projectedStates));
  if (constraintNorm > settings.constraintTolerance) {
    return { projectedStates: undefined, constraintNorm };
  }
  return { projectedStates, constraintNorm };
}

export function restoreFeasibleStates(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  nextParameters: Float64Array,
  dcdx: Matrix,
  dcdp: Matrix,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  logger: Logger,
  algorithmName: string
): Float64Array | undefined {
  const deltaP = subtractVectors(nextParameters, currentParameters);
  const predictedStates = updateStates(
    currentStates,
    dcdx,
    dcdp,
    deltaP,
    logger,
    algorithmName,
    settings.regularization
  );
  return projectAndVerify(
    nextParameters,
    predictedStates,
    constraintFunction,
    settings,
    logger,
    algorithmName,
    TRIAL_PROJECTION_ITERATIONS
  ).projectedStates;
}

export function resolveObjectiveKind(
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  parameters: Float64Array,
  states: Float64Array
): ObjectiveKind {
  const sample = costFunction(parameters, states);
  return sample instanceof Float64Array ? 'residual' : 'cost';
}

export function computeCost(
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

function computeGradientFromResidual(residual: Float64Array, derivativeMatrix: Matrix): Float64Array {
  const residualMatrix = float64ArrayToMatrix(residual);
  const gradientMatrix = residualMatrix.transpose().mmul(derivativeMatrix);
  return rowVectorToFloat64Array(gradientMatrix);
}

function computeDfdp(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  settings: AdjointReducedSpaceSettings,
  objectiveKind: ObjectiveKind
): Float64Array {
  if (settings.dfdp) {
    return settings.dfdp(parameters, states);
  }
  if (objectiveKind === 'residual') {
    const residualFunction = costFunction as ConstrainedResidualFn;
    const derivativeResidualPartialP = finiteDiffResidualPartialP(
      parameters,
      states,
      residualFunction,
      { stepSize: settings.stepSizeP }
    );
    const residual = residualFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialP);
  }
  return finiteDiffPartialP(
    parameters,
    states,
    costFunction as ConstrainedCostFn,
    { stepSize: settings.stepSizeP }
  );
}

function computeDfdx(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  settings: AdjointReducedSpaceSettings,
  objectiveKind: ObjectiveKind
): Float64Array {
  if (settings.dfdx) {
    return settings.dfdx(parameters, states);
  }
  if (objectiveKind === 'residual') {
    const residualFunction = costFunction as ConstrainedResidualFn;
    const derivativeResidualPartialX = finiteDiffResidualPartialX(
      parameters,
      states,
      residualFunction,
      { stepSize: settings.stepSizeX }
    );
    const residual = residualFunction(parameters, states);
    return computeGradientFromResidual(residual, derivativeResidualPartialX);
  }
  return finiteDiffPartialX(
    parameters,
    states,
    costFunction as ConstrainedCostFn,
    { stepSize: settings.stepSizeX }
  );
}

export function computePartialDerivatives(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  objectiveKind: ObjectiveKind,
  algorithmName: string
): ReducedPartials {
  const dfdp = computeDfdp(parameters, states, costFunction, settings, objectiveKind);
  const dfdx = computeDfdx(parameters, states, costFunction, settings, objectiveKind);
  const { dcdp, dcdx } = resolveConstraintJacobians(
    parameters,
    states,
    constraintFunction,
    settings,
    algorithmName
  );
  return { dfdp, dfdx, dcdp, dcdx };
}

export function computeReducedGradient(
  parameters: Float64Array,
  states: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  logger: Logger,
  algorithmName: string,
  objectiveKind: ObjectiveKind,
  cachedPartials?: ReducedPartials
): ReducedGradientEvaluation {
  const partials =
    cachedPartials ??
    computePartialDerivatives(
      parameters,
      states,
      costFunction,
      constraintFunction,
      settings,
      objectiveKind,
      algorithmName
    );
  const lambda = solveAdjointEquation(
    partials.dcdx,
    partials.dfdx,
    logger,
    algorithmName,
    settings.regularization
  );
  return {
    gradient: computeAdjointGradient(partials.dfdp, lambda, partials.dcdp),
    partials
  };
}

export function initializeFeasibleReducedSpace(
  initialParameters: Float64Array,
  initialStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  logger: Logger,
  algorithmName: string
): FeasibleReducedSpace {
  validateInitialConditions(
    initialParameters,
    initialStates,
    constraintFunction,
    settings.constraintTolerance,
    logger,
    algorithmName
  );
  const initialConstraint = constraintFunction(initialParameters, initialStates);
  assertSquareImplicitStateSystem(
    initialConstraint.length,
    initialStates.length,
    initialParameters.length,
    algorithmName
  );
  // WHY: Reject a malformed analytical Jacobian before Newton projection can explode.
  resolveConstraintJacobians(initialParameters, initialStates, constraintFunction, settings, algorithmName);

  const restoreAttempt = projectAndVerify(
    initialParameters,
    initialStates,
    constraintFunction,
    settings,
    logger,
    algorithmName,
    INITIAL_PROJECTION_ITERATIONS
  );
  if (restoreAttempt.projectedStates === undefined) {
    throw new Error(
      `Failed to restore feasible states for ${algorithmName}: ` +
        `||c(p,x)||=${restoreAttempt.constraintNorm}, tolerance=${settings.constraintTolerance}, ` +
        `parameterCount=${initialParameters.length}, stateCount=${initialStates.length}. ` +
        `Provide a projectable implicit-state guess with a locally unique x(p).`
    );
  }

  const parameters = new Float64Array(initialParameters);
  const states = new Float64Array(restoreAttempt.projectedStates);
  const objectiveKind = resolveObjectiveKind(costFunction, parameters, states);
  return {
    parameters,
    states,
    objectiveKind,
    cost: computeCost(costFunction, parameters, states, objectiveKind),
    constraintNorm: restoreAttempt.constraintNorm
  };
}

/**
 * Reduced-space (f̃, ∇f̃) for a line search that must not see frozen trial gradients.
 * Tangent prediction may reuse start-of-search ∂c/∂x; the adjoint is always recomputed
 * on the restored point. Restore failure is +∞ cost and must not yield a gradient.
 */
export function createReducedObjective(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  costFunction: ConstrainedCostFn | ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  settings: AdjointReducedSpaceSettings,
  logger: Logger,
  algorithmName: string,
  objectiveKind: ObjectiveKind,
  predictorPartials: ConstraintPartials
): { evaluateCost: CostFn; evaluateGradient: GradientFn } {
  let cachedParameters: Float64Array | undefined;
  let cachedStates: Float64Array | undefined;

  function restoreTrial(trialParameters: Float64Array): Float64Array | undefined {
    if (
      cachedParameters !== undefined &&
      areParametersEqual(trialParameters, cachedParameters)
    ) {
      return cachedStates;
    }
    const trialStates = restoreFeasibleStates(
      currentParameters,
      currentStates,
      trialParameters,
      predictorPartials.dcdx,
      predictorPartials.dcdp,
      constraintFunction,
      settings,
      logger,
      algorithmName
    );
    cachedParameters = new Float64Array(trialParameters);
    cachedStates = trialStates;
    return trialStates;
  }

  function evaluateCost(trialParameters: Float64Array): number {
    const trialStates = restoreTrial(trialParameters);
    if (trialStates === undefined) {
      return UNPROJECTABLE_TRIAL_COST;
    }
    return computeCost(costFunction, trialParameters, trialStates, objectiveKind);
  }

  function evaluateGradient(trialParameters: Float64Array): Float64Array {
    const trialStates = restoreTrial(trialParameters);
    if (trialStates === undefined) {
      throw new Error(
        `${algorithmName} evaluated the reduced gradient where state restoration failed. ` +
          `The reduced cost is undefined at this point.`
      );
    }
    return computeReducedGradient(
      trialParameters,
      trialStates,
      costFunction,
      constraintFunction,
      settings,
      logger,
      algorithmName,
      objectiveKind
    ).gradient;
  }

  return { evaluateCost, evaluateGradient };
}

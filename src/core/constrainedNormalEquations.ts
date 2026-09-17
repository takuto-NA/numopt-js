/**
 * Shared normal-equation and constrained step helpers for GN / LM.
 * Damping policy and outer loops stay in each solver.
 */

import { CholeskyDecomposition, Matrix, solve } from 'ml-matrix';
import type { ConstrainedResidualFn, ConstraintFn } from './types.js';
import {
  computeSumOfSquaredResiduals,
  float64ArrayToMatrix,
  matrixToFloat64Array,
  vectorNorm
} from '../utils/matrix.js';
import { Logger } from './logger.js';
import { projectStatesToConstraints, updateStates } from './constrainedUtils.js';
import {
  finiteDiffConstraintPartialP,
  finiteDiffConstraintPartialX
} from './finiteDiff.js';

const NEGATIVE_COEFFICIENT = -1.0;
const DEFAULT_RIDGE_REGULARIZATION = 1e-8;

export type NormalEquationsMatrices = {
  jtj: Matrix;
  jtr: Matrix;
};

export function computeNormalEquationsMatrices(
  jacobianMatrix: Matrix,
  residual: Float64Array
): NormalEquationsMatrices {
  const jacobianTranspose = jacobianMatrix.transpose();
  const jtj = jacobianTranspose.mmul(jacobianMatrix);
  const residualMatrix = float64ArrayToMatrix(residual);
  const jtr = jacobianTranspose.mmul(residualMatrix);
  return { jtj, jtr };
}

export function solveUndampedNormalEquations(
  jtj: Matrix,
  jtr: Matrix,
  ridgeRegularization: number = DEFAULT_RIDGE_REGULARIZATION
): Float64Array {
  const negativeRightHandSide = jtr.mul(NEGATIVE_COEFFICIENT);
  const jittered = jtj.add(Matrix.eye(jtj.rows, jtj.columns).mul(ridgeRegularization));

  try {
    const cholesky = new CholeskyDecomposition(jtj);
    if (cholesky.isPositiveDefinite()) {
      return matrixToFloat64Array(cholesky.solve(negativeRightHandSide));
    }
  } catch {
    // Fall through to ridge regularization
  }

  try {
    const choleskyRidge = new CholeskyDecomposition(jittered);
    if (choleskyRidge.isPositiveDefinite()) {
      return matrixToFloat64Array(choleskyRidge.solve(negativeRightHandSide));
    }
  } catch {
    // Fall through to general solver
  }

  return matrixToFloat64Array(solve(jtj, negativeRightHandSide));
}

export function solveDampedNormalEquations(
  jtj: Matrix,
  jtr: Matrix,
  currentLambda: number,
  lambdaFactor: number
): { step: Float64Array; stepNorm: number } | { newLambda: number } {
  const parameterCount = jtj.rows;
  const identity = Matrix.eye(parameterCount, parameterCount);
  const dampedHessian = jtj.add(identity.mul(currentLambda));
  const negativeJtr = jtr.mul(NEGATIVE_COEFFICIENT);

  try {
    const cholesky = new CholeskyDecomposition(dampedHessian);
    if (cholesky.isPositiveDefinite()) {
      const step = matrixToFloat64Array(cholesky.solve(negativeJtr));
      return { step, stepNorm: vectorNorm(step) };
    }
    return { newLambda: currentLambda * lambdaFactor };
  } catch {
    return { newLambda: currentLambda * lambdaFactor };
  }
}

export function checkConstraintViolation(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  iteration: number,
  logger: Logger,
  algorithmName: string
): { constraint: Float64Array; constraintNorm: number } {
  const constraint = constraintFunction(currentParameters, currentStates);
  const constraintNorm = vectorNorm(constraint);
  if (constraintNorm > constraintTolerance) {
    logger.warn(algorithmName, iteration, 'Constraint violation detected', [
      { key: '||c(p,x)||:', value: constraintNorm },
      { key: 'Tolerance:', value: constraintTolerance }
    ]);
  }
  return { constraint, constraintNorm };
}

export function applyConstrainedParameterStep(
  currentParameters: Float64Array,
  currentStates: Float64Array,
  step: Float64Array,
  constraintFunction: ConstraintFn,
  stepSizeP: number,
  stepSizeX: number,
  constraintTolerance: number,
  logger: Logger,
  algorithmName: string,
  dcdp?: (parameters: Float64Array, states: Float64Array) => Matrix,
  dcdx?: (parameters: Float64Array, states: Float64Array) => Matrix
): { newParameters: Float64Array; newStates: Float64Array } {
  const newParameters = new Float64Array(currentParameters.length);
  for (let index = 0; index < currentParameters.length; index++) {
    newParameters[index] = currentParameters[index] + step[index];
  }

  const constraintJacobianX = dcdx
    ? dcdx(currentParameters, currentStates)
    : finiteDiffConstraintPartialX(currentParameters, currentStates, constraintFunction, {
        stepSize: stepSizeX
      });
  const constraintJacobianP = dcdp
    ? dcdp(currentParameters, currentStates)
    : finiteDiffConstraintPartialP(currentParameters, currentStates, constraintFunction, {
        stepSize: stepSizeP
      });

  const predictedStates = updateStates(
    currentStates,
    constraintJacobianX,
    constraintJacobianP,
    step,
    logger,
    algorithmName
  );
  const projectedStates = projectStatesToConstraints(
    newParameters,
    predictedStates,
    constraintFunction,
    stepSizeX,
    constraintTolerance,
    logger,
    algorithmName
  );
  return { newParameters, newStates: projectedStates };
}

export function evaluateConstrainedIterationMetrics(
  parameters: Float64Array,
  states: Float64Array,
  residualFunction: ConstrainedResidualFn,
  constraintFunction: ConstraintFn,
  constraintTolerance: number,
  iteration: number,
  logger: Logger,
  algorithmName: string
): {
  residual: Float64Array;
  residualNorm: number;
  cost: number;
  constraintNorm: number;
  constraintSatisfied: boolean;
} {
  const { constraintNorm } = checkConstraintViolation(
    parameters,
    states,
    constraintFunction,
    constraintTolerance,
    iteration,
    logger,
    algorithmName
  );
  const residual = residualFunction(parameters, states);
  const residualNorm = vectorNorm(residual);
  const cost = computeSumOfSquaredResiduals(residualNorm);
  return {
    residual,
    residualNorm,
    cost,
    constraintNorm,
    constraintSatisfied: constraintNorm <= constraintTolerance
  };
}

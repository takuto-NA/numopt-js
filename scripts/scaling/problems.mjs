/**
 * Dimension-controlled implicit-state problems with independent reduced-gradient oracles.
 * Linear coupled state equations are stored densely to measure the existing solver implementation.
 */

import { Matrix } from 'ml-matrix';

const STATE_COUPLING = 0.25;
const TARGET_PARAMETER = 1;
const INITIAL_PARAMETER = 1.5;
const FIRST_DERIVATIVE_FACTOR = 2;
const INITIAL_PARAMETER_SPREAD = 0.3;
const INITIAL_PHASE_INCREMENT = (1 + Math.sqrt(5)) / 2;

export function createScalingProblem({ parameters: parameterCount, states: stateCount, family }) {
  if (!Number.isInteger(parameterCount) || parameterCount <= 0 || !Number.isInteger(stateCount) || stateCount <= 0) {
    throw new Error('Parameter and state counts must be positive integers');
  }
  if (!['easy', 'curved', 'heterogeneous'].includes(family)) throw new Error('Unknown scaling problem family');
  const nonlinear = family !== 'easy';
  const assignments = Int32Array.from({ length: stateCount }, (_, index) => index % parameterCount);
  const repetitions = new Int32Array(parameterCount);
  for (const index of assignments) repetitions[index]++;
  const weights = Float64Array.from(assignments, index => 1 / repetitions[index]);
  const stateJacobian = Matrix.eye(stateCount);
  for (let row = 1; row < stateCount; row++) stateJacobian.set(row, row - 1, -STATE_COUPLING);
  const initialParameters = family === 'heterogeneous'
    ? Float64Array.from({ length: parameterCount }, (_, index) =>
      INITIAL_PARAMETER + INITIAL_PARAMETER_SPREAD * Math.sin(index * INITIAL_PHASE_INCREMENT))
    : new Float64Array(parameterCount).fill(INITIAL_PARAMETER);
  const stateValue = value => nonlinear ? value * value : value;
  const stateDerivative = value => nonlinear ? FIRST_DERIVATIVE_FACTOR * value : 1;
  const initialStates = Float64Array.from(assignments, index => stateValue(initialParameters[index]));
  let cachedParameterHessian;
  let zeroStateHessian;
  let cachedJointHessian;

  function reducedGradient(parameters) {
    return Float64Array.from(parameters, value => {
      const error = value - TARGET_PARAMETER;
      return nonlinear ? error ** 3 + error ** 2 + 3 * error : error;
    });
  }

  function reducedCurvature(parameters) {
    return Float64Array.from(parameters, value => {
      const error = value - TARGET_PARAMETER;
      return nonlinear ? 3 * error * error + 2 * error + 3 : 1;
    });
  }

  function objective(parameters, states) {
    let cost = 0;
    for (const parameter of parameters) {
      const error = parameter - TARGET_PARAMETER;
      cost += nonlinear ? error ** 4 / 4 + error ** 3 / 3 + 3 * error * error / 2 : error * error / 2;
    }
    for (let row = 0; row < stateCount; row++) {
      cost += weights[row] * (states[row] - stateValue(parameters[assignments[row]]));
    }
    return cost;
  }

  function constraint(parameters, states) {
    const values = new Float64Array(stateCount);
    let previous = 0;
    for (let row = 0; row < stateCount; row++) {
      const current = states[row] - stateValue(parameters[assignments[row]]);
      values[row] = current - STATE_COUPLING * previous;
      previous = current;
    }
    return values;
  }

  function constraintPartialP(parameters) {
    const jacobian = Matrix.zeros(stateCount, parameterCount);
    for (let row = 0; row < stateCount; row++) {
      const column = assignments[row];
      jacobian.set(row, column, -stateDerivative(parameters[column]));
      if (row > 0) {
        const previous = assignments[row - 1];
        jacobian.set(row, previous, jacobian.get(row, previous) + STATE_COUPLING * stateDerivative(parameters[previous]));
      }
    }
    return jacobian;
  }

  const fixedConstraintPartialP = nonlinear ? undefined : constraintPartialP(initialParameters);
  const firstDerivatives = {
    dfdp(parameters) {
      const gradient = reducedGradient(parameters);
      for (let row = 0; row < stateCount; row++) gradient[assignments[row]] -= weights[row] * stateDerivative(parameters[assignments[row]]);
      return gradient;
    },
    dfdx: () => new Float64Array(weights),
    dcdp: parameters => fixedConstraintPartialP ?? constraintPartialP(parameters),
    dcdx: () => stateJacobian
  };

  function approximateParameterHessian(parameters) {
    if (!nonlinear) return cachedParameterHessian ??= Matrix.eye(parameterCount);
    const diagonal = reducedCurvature(parameters);
    for (let row = 0; row < stateCount; row++) diagonal[assignments[row]] -= FIRST_DERIVATIVE_FACTOR * weights[row];
    return Matrix.diag(Array.from(diagonal));
  }

  function lagrangianDiagonal(parameters, lambda) {
    const diagonal = reducedCurvature(parameters);
    if (nonlinear) {
      for (let row = 0; row < stateCount; row++) {
        const multiplierWeight = lambda[row] - (row + 1 < stateCount ? STATE_COUPLING * lambda[row + 1] : 0);
        diagonal[assignments[row]] += FIRST_DERIVATIVE_FACTOR * (multiplierWeight - weights[row]);
      }
    }
    return diagonal;
  }

  return {
    initialParameters, initialStates, objective, constraint, reducedGradient,
    firstDerivatives,
    approximateDerivatives: { d2fdp2: approximateParameterHessian,
      d2fdx2: () => zeroStateHessian ??= Matrix.zeros(stateCount, stateCount) },
    lagrangianHessian(parameters, _states, lambda) {
      if (!nonlinear && cachedJointHessian) return cachedJointHessian;
      const hessian = Matrix.zeros(parameterCount + stateCount, parameterCount + stateCount);
      const diagonal = lagrangianDiagonal(parameters, lambda);
      for (let index = 0; index < parameterCount; index++) hessian.set(index, index, diagonal[index]);
      if (!nonlinear) cachedJointHessian = hessian;
      return hessian;
    },
    lagrangianHessianVector(parameters, _states, lambda, direction) {
      const product = new Float64Array(parameterCount + stateCount);
      const diagonal = lagrangianDiagonal(parameters, lambda);
      for (let index = 0; index < parameterCount; index++) product[index] = diagonal[index] * direction[index];
      return product;
    },
    reference: { target: TARGET_PARAMETER, stateCoupling: STATE_COUPLING, curvatureAtSolution: nonlinear ? 3 : 1 }
  };
}

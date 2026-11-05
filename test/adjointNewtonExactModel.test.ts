/**
 * Independent full-Hessian checks with nonlinear p-x and x-x constraint curvature.
 */

import { Matrix, solve } from 'ml-matrix';
import { computeExactAdjointHessian } from '../src/core/adjointNewtonExactModel';
import type { ReducedGradientEvaluation } from '../src/core/adjointReducedSpace';

const PARAMETER_HESSIAN = new Matrix([[4, 0.5], [0.5, 3]]);
const STATE_HESSIAN = new Matrix([[2, 0.3], [0.3, 1]]);
const MIXED_HESSIAN = new Matrix([[0.7, -0.2], [0.4, 0.6]]);
const STATE_LINEAR_TERM = new Matrix([[0.8], [-0.5]]);
const parameters = new Float64Array([0.2, 0.3]);
const states = new Float64Array([(parameters[0] ** 2 + parameters[1]) / (1 + parameters[0]),
  Math.sqrt(Math.exp(parameters[1]) + parameters[0])]);
const stateGradient = MIXED_HESSIAN.transpose().mmul(Matrix.columnVector(Array.from(parameters)))
  .add(STATE_HESSIAN.mmul(Matrix.columnVector(Array.from(states)))).add(STATE_LINEAR_TERM);
const constraintPartialP = new Matrix([[states[0] - 2 * parameters[0], -1], [-1, -Math.exp(parameters[1])]]);
const constraintPartialX = Matrix.diag([1 + parameters[0], 2 * states[1]]);
const reduced: ReducedGradientEvaluation = {
  gradient: new Float64Array(2),
  lambda: new Float64Array(solve(constraintPartialX.transpose(), stateGradient).getColumn(0)),
  partials: { dfdp: new Float64Array(2), dfdx: new Float64Array(stateGradient.getColumn(0)),
    dcdp: constraintPartialP, dcdx: constraintPartialX }
};

function cost(parameters: Float64Array, states: Float64Array): number {
  const parameterColumn = Matrix.columnVector(Array.from(parameters));
  const stateColumn = Matrix.columnVector(Array.from(states));
  return parameterColumn.transpose().mmul(PARAMETER_HESSIAN).mmul(parameterColumn).get(0, 0) / 2 +
    parameterColumn.transpose().mmul(MIXED_HESSIAN).mmul(stateColumn).get(0, 0) +
    stateColumn.transpose().mmul(STATE_HESSIAN).mmul(stateColumn).get(0, 0) / 2 +
    STATE_LINEAR_TERM.transpose().mmul(stateColumn).get(0, 0);
}

function constraint(parameters: Float64Array, states: Float64Array): Float64Array {
  return new Float64Array([(1 + parameters[0]) * states[0] - parameters[0] ** 2 - parameters[1],
    states[1] ** 2 - Math.exp(parameters[1]) - parameters[0]]);
}

function jointHessian(_parameters: Float64Array, _states: Float64Array, lambda: Float64Array): Matrix {
  const hessian = Matrix.zeros(4, 4);
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 2; column++) {
      hessian.set(row, column, PARAMETER_HESSIAN.get(row, column));
      hessian.set(row + 2, column + 2, STATE_HESSIAN.get(row, column));
      hessian.set(row, column + 2, MIXED_HESSIAN.get(row, column));
      hessian.set(column + 2, row, MIXED_HESSIAN.get(row, column));
    }
  }
  hessian.set(0, 0, hessian.get(0, 0) + 2 * lambda[0]);
  hessian.set(1, 1, hessian.get(1, 1) + Math.exp(parameters[1]) * lambda[1]);
  hessian.set(0, 2, hessian.get(0, 2) - lambda[0]);
  hessian.set(2, 0, hessian.get(2, 0) - lambda[0]);
  hessian.set(3, 3, hessian.get(3, 3) - 2 * lambda[1]);
  return hessian;
}

function explicitSolutionHessian(): Matrix {
  const [first, second] = parameters;
  const exponential = Math.exp(second);
  const lastState = states[1];
  const sensitivity = new Matrix([[(first * first + 2 * first - second) / (1 + first) ** 2, 1 / (1 + first)],
    [1 / (2 * lastState), exponential / (2 * lastState)]]);
  const firstStateHessian = new Matrix([[2 * (1 + second) / (1 + first) ** 3, -1 / (1 + first) ** 2],
    [-1 / (1 + first) ** 2, 0]]);
  const lastStateHessian = new Matrix([[-1 / (4 * lastState ** 3), -exponential / (4 * lastState ** 3)],
    [-exponential / (4 * lastState ** 3), exponential / (2 * lastState) - exponential ** 2 / (4 * lastState ** 3)]]);
  return PARAMETER_HESSIAN.clone().add(MIXED_HESSIAN.mmul(sensitivity))
    .add(sensitivity.transpose().mmul(MIXED_HESSIAN.transpose()))
    .add(sensitivity.transpose().mmul(STATE_HESSIAN).mmul(sensitivity))
    .add(firstStateHessian.mul(stateGradient.get(0, 0))).add(lastStateHessian.mul(stateGradient.get(1, 0)));
}

describe('full reduced Hessian', () => {
  it.each(['analytical', 'numerical', 'hessianVector'])('matches explicit second state sensitivities in %s mode', mode => {
    const snapshot = Array.from(reduced.lambda);
    const hessian = computeExactAdjointHessian(parameters, states, cost, constraint, reduced, {
      lagrangianHessian: mode === 'analytical' ? jointHessian : undefined,
      lagrangianHessianVector: mode === 'hessianVector' ? (parameters, states, lambda, direction) =>
        new Float64Array(jointHessian(parameters, states, lambda).mmul(Matrix.columnVector(Array.from(direction))).getColumn(0)) : undefined
    });
    const expected = explicitSolutionHessian();
    for (let row = 0; row < 2; row++) {
      for (let column = 0; column < 2; column++) expect(hessian.get(row, column)).toBeCloseTo(expected.get(row, column), 4);
    }
    expect(Array.from(reduced.lambda)).toEqual(snapshot);
    expect(constraintPartialX.to2DArray()).toEqual([[1 + parameters[0], 0], [0, 2 * states[1]]]);
  });

  it('rejects a nonsymmetric projected Hessian-vector operator', () => {
    expect(() => computeExactAdjointHessian(parameters, states, cost, constraint, reduced, {
      lagrangianHessianVector: (_parameters, _states, _lambda, direction) => new Float64Array([direction[1], 0, 0, 0])
    })).toThrow(/symmetric/);
  });
});

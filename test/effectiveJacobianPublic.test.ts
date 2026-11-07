/** Public implicit derivatives against independently eliminated nonlinear states and stable rectangular solves. */

import { LuDecomposition, Matrix, QrDecomposition } from 'ml-matrix';
import { describe, expect, it, vi } from 'vitest';
import { computeEffectiveJacobian, type EffectiveJacobianOptions } from '../src/index';

const FINITE_DIFFERENCE_STEP = 1e-5;
const DERIVATIVE_TOLERANCE = 1e-8;
const NEAR_DEPENDENCE = 1e-6;

function eliminatedStates(parameters: Float64Array): Float64Array {
  return new Float64Array([parameters[0] ** 2 + parameters[1], Math.sin(parameters[0]) - 2 * parameters[1]]);
}

function residual(parameters: Float64Array, states: Float64Array): Float64Array {
  return new Float64Array([states[0] + parameters[0], states[1] - parameters[1], states[0] / 2 + states[1]]);
}

function nonlinearFixture(redundant: boolean) {
  const coefficients = redundant ? new Matrix([[1, 0], [0, 1], [2, -3], [-1, 2]]) : Matrix.eye(2);
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array => {
    const exact = eliminatedStates(parameters);
    return Float64Array.from(coefficients.mmul(Matrix.columnVector(states.map((value, index) => value - exact[index]))).to1DArray());
  };
  const derivatives: EffectiveJacobianOptions = {
    drdp: () => new Matrix([[1, 0], [0, -1], [0, 0]]),
    drdx: () => new Matrix([[1, 0], [0, 1], [0.5, 1]]),
    dcdp: parameters => coefficients.mmul(new Matrix([[-2 * parameters[0], -1], [-Math.cos(parameters[0]), 2]])),
    dcdx: () => coefficients
  };
  return { constraint, coefficients, derivatives };
}

describe('public effective Jacobian', () => {
  it('does not mutate caller-owned derivative matrices', () => {
    const fixture = nonlinearFixture(true);
    const parameters = new Float64Array([0.7, -0.3]);
    const states = eliminatedStates(parameters);
    const matrices = {
      drdp: fixture.derivatives.drdp!(parameters, states), drdx: fixture.derivatives.drdx!(parameters, states),
      dcdp: fixture.derivatives.dcdp!(parameters, states), dcdx: fixture.derivatives.dcdx!(parameters, states)
    };
    const originals = Object.values(matrices).map(matrix => matrix.to2DArray());
    computeEffectiveJacobian(parameters, states, residual, fixture.constraint, {
      drdp: () => matrices.drdp, drdx: () => matrices.drdx, dcdp: () => matrices.dcdp, dcdx: () => matrices.dcdx
    });
    Object.values(matrices).forEach((matrix, index) => expect(matrix.to2DArray()).toEqual(originals[index]));
  });

  it.each([false, true])('matches a fully eliminated nonlinear trajectory (redundant=%s)', redundant => {
    const { constraint, coefficients, derivatives } = nonlinearFixture(redundant);
    const parameters = new Float64Array([0.7, -0.3]);
    const states = eliminatedStates(parameters);
    const original = coefficients.to2DArray();
    const actual = computeEffectiveJacobian(parameters, states, residual, constraint, derivatives);
    const numerical = computeEffectiveJacobian(parameters, states, residual, constraint);
    for (let column = 0; column < parameters.length; column++) {
      const plus = parameters.slice();
      const minus = parameters.slice();
      plus[column] += FINITE_DIFFERENCE_STEP;
      minus[column] -= FINITE_DIFFERENCE_STEP;
      const forward = residual(plus, eliminatedStates(plus));
      const backward = residual(minus, eliminatedStates(minus));
      forward.forEach((value, row) => {
        const expected = (value - backward[row]) / (2 * FINITE_DIFFERENCE_STEP);
        expect(Math.abs(actual.get(row, column) - expected)).toBeLessThan(DERIVATIVE_TOLERANCE);
        expect(Math.abs(numerical.get(row, column) - expected)).toBeLessThan(DERIVATIVE_TOLERANCE);
      });
    }
    expect(coefficients.to2DArray()).toEqual(original);
  });

  it.each([false, true])('solves every sensitivity column in one decomposition solve (redundant=%s)', redundant => {
    const fixture = nonlinearFixture(redundant);
    const parameters = new Float64Array([0.7, -0.3]);
    const prototype = redundant ? QrDecomposition.prototype : LuDecomposition.prototype;
    const spy = vi.spyOn(prototype, 'solve');
    try {
      computeEffectiveJacobian(parameters, eliminatedStates(parameters), residual, fixture.constraint, fixture.derivatives);
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0].columns).toBe(parameters.length);
    } finally {
      spy.mockRestore();
    }
  });

  it('preserves sensitivities of nearly dependent redundant constraints without normal equations', () => {
    const coefficients = new Matrix([[1, 1], [1, 1 + NEAR_DEPENDENCE], [1, 1 - NEAR_DEPENDENCE]]);
    const expected = new Matrix([[2, -1], [-3, 4]]);
    const parameterPartial = coefficients.mmul(expected).mul(-1);
    const parameters = new Float64Array([0, 0]);
    const constraint = (design: Float64Array, states: Float64Array): Float64Array => Float64Array.from(
      coefficients.mmul(Matrix.columnVector(states)).add(parameterPartial.mmul(Matrix.columnVector(design))).to1DArray());
    const actual = computeEffectiveJacobian(parameters, parameters, (_design, states) => states, constraint, {
      drdp: () => Matrix.zeros(2, 2), drdx: () => Matrix.eye(2),
      dcdp: () => parameterPartial, dcdx: () => coefficients
    });
    actual.to1DArray().forEach((value, index) => expect(Math.abs(value - expected.to1DArray()[index])).toBeLessThan(DERIVATIVE_TOLERANCE));
  });

  it('retains the documented minimum-norm linearization for underdetermined state systems', () => {
    const actual = computeEffectiveJacobian(new Float64Array([1]), new Float64Array([0.5, 0.5]),
      (_parameters, states) => states,
      (parameters, states) => new Float64Array([states[0] + states[1] - parameters[0]]));
    expect(actual.get(0, 0)).toBeCloseTo(0.5, 8);
    expect(actual.get(1, 0)).toBeCloseTo(0.5, 8);
  });

  it.each(['drdp', 'drdx', 'dcdp', 'dcdx'] as const)('rejects malformed %s dimensions before solving', key => {
    const fixture = nonlinearFixture(true);
    const parameters = new Float64Array([0.7, -0.3]);
    expect(() => computeEffectiveJacobian(parameters, eliminatedStates(parameters), residual, fixture.constraint,
      { ...fixture.derivatives, [key]: () => Matrix.zeros(1, 1) })).toThrow(/must be/);
  });
});

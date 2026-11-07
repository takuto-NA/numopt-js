/** Regression checks for reusable normal equations and rejected LM trial damping. */

import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';
import { solveDampedNormalEquations, solveUndampedNormalEquations } from '../src/core/constrainedNormalEquations';
import { constrainedLevenbergMarquardt, levenbergMarquardt } from '../src/index';

const LAMBDA_FACTOR = 10;
const RIDGE = 1e-4;
const DECIMAL_PRECISION = 12;

describe('constrained normal equation solves', () => {
  it('repeats the same LM solve without changing its inputs or step direction', () => {
    const hessian = new Matrix([[1]]);
    const gradient = new Matrix([[1]]);
    for (const damping of [1, 1, 10, 1]) {
      const result = solveDampedNormalEquations(hessian, gradient, damping, LAMBDA_FACTOR);
      expect('step' in result).toBe(true);
      if ('step' in result) expect(result.step[0]).toBeCloseTo(-1 / (1 + damping), DECIMAL_PRECISION);
      expect(hessian.to2DArray()).toEqual([[1]]);
      expect(gradient.to2DArray()).toEqual([[1]]);
    }
  });

  it('solves an undamped positive-definite model before applying a ridge', () => {
    const hessian = new Matrix([[2]]);
    const gradient = new Matrix([[3]]);
    const result = solveUndampedNormalEquations(hessian, gradient, RIDGE);
    expect(result[0]).toBeCloseTo(-1.5, DECIMAL_PRECISION);
    expect(hessian.to2DArray()).toEqual([[2]]);
    expect(gradient.to2DArray()).toEqual([[3]]);
  });

  it('uses a ridge only for a singular model and leaves it reusable', () => {
    const hessian = new Matrix([[1, 1], [1, 1]]);
    const gradient = new Matrix([[2], [2]]);
    const original = hessian.to2DArray();
    for (let iteration = 0; iteration < 2; iteration++) {
      const result = solveUndampedNormalEquations(hessian, gradient, RIDGE);
      result.forEach(value => expect(value).toBeCloseTo(-2 / (2 + RIDGE), 8));
      expect(hessian.to2DArray()).toEqual(original);
      expect(gradient.to2DArray()).toEqual([[2], [2]]);
    }
  });

  it.each([false, true])('rejected nonlinear LM trials retain the descent direction (constrained=%s)', constrained => {
    const initial = new Float64Array([0.1]);
    const visited: number[] = [];
    const evaluated: number[] = [];
    const residual = (parameters: Float64Array): Float64Array => {
      evaluated.push(parameters[0]);
      return new Float64Array([parameters[0] ** 2 - 1]);
    };
    const options = { lambdaInitial: 1e-6, tolGradient: 1e-10, tolStep: 1e-12, tolResidual: 1e-10,
      onIteration: (_iteration: number, _cost: number, parameters: Float64Array) => { visited.push(parameters[0]); } };
    const result = constrained
      ? constrainedLevenbergMarquardt(initial, initial,
        (_parameters, states) => residual(states),
        (parameters, states) => new Float64Array([states[0] - parameters[0]]), {
          ...options,
          drdp: () => Matrix.zeros(1, 1), drdx: (_parameters, states) => new Matrix([[2 * states[0]]]),
          dcdp: () => new Matrix([[-1]]), dcdx: () => Matrix.eye(1)
        })
      : levenbergMarquardt(initial, residual,
        { ...options, jacobian: parameters => new Matrix([[2 * parameters[0]]]) });
    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(1, 8);
    expect(visited.every(value => value > 0)).toBe(true);
    expect(evaluated.some(value => value > 2)).toBe(true);
  });
});

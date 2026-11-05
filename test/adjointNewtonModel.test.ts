/**
 * Mathematical checks of the partial-Hessian model, projected numerical curvature, and immutable solves.
 */

import { Matrix } from 'ml-matrix';
import { computeApproximateAdjointHessian, computeSafeguardedNewtonDirection } from '../src/core/adjointNewtonModel';
import { solveLeastSquares } from '../src/core/constrainedUtils';
import { Logger } from '../src/core/logger';
import type { ReducedPartials } from '../src/core/adjointReducedSpace';

const PARAMETER_HESSIAN = new Matrix([[4, 1], [1, 2]]);
const STATE_HESSIAN = new Matrix([[3, 1], [1, 5]]);
const EXPECTED_APPROXIMATION = [[10, -7], [-7, 71]];
const partials: ReducedPartials = {
  dfdp: new Float64Array(2), dfdx: new Float64Array(2),
  dcdp: new Matrix([[-1, -2], [1, -3]]), dcdx: Matrix.eye(2)
};
const parameters = new Float64Array([0.4, -0.3]);
const states = new Float64Array([-0.2, -1.3]);

function coupledCost(parameters: Float64Array, states: Float64Array): number {
  const [firstParameter, secondParameter] = parameters;
  const [firstState, secondState] = states;
  const parameterCost = (4 * firstParameter ** 2 + 2 * firstParameter * secondParameter + 2 * secondParameter ** 2) / 2;
  const stateCost = (3 * firstState ** 2 + 2 * firstState * secondState + 5 * secondState ** 2) / 2;
  // These mixed p-x terms must not enter either partial Hessian block.
  return parameterCost + stateCost + firstParameter * firstState - 2 * secondParameter * secondState;
}

describe('Newton-adjoint Hessian model', () => {
  it.each(['analytical', 'numerical', 'parameterOnly', 'stateOnly'])('matches f_pp + S^T f_xx S in %s mode', mode => {
    const hessian = computeApproximateAdjointHessian(parameters, states, coupledCost, partials, {
      d2fdp2: mode === 'analytical' || mode === 'parameterOnly' ? () => PARAMETER_HESSIAN : undefined,
      d2fdx2: mode === 'analytical' || mode === 'stateOnly' ? () => STATE_HESSIAN : undefined
    });
    hessian.to2DArray().forEach((row, rowIndex) => row.forEach((value, column) =>
      expect(value).toBeCloseTo(EXPECTED_APPROXIMATION[rowIndex][column], 4)));
    expect(PARAMETER_HESSIAN.to2DArray()).toEqual([[4, 1], [1, 2]]);
    expect(STATE_HESSIAN.to2DArray()).toEqual([[3, 1], [1, 5]]);
    expect(partials.dcdx.to2DArray()).toEqual([[1, 0], [0, 1]]);
  });

  it('numerical state curvature holds p fixed and omits the curvature of x(p)', () => {
    const hessian = computeApproximateAdjointHessian(new Float64Array([0.5]), new Float64Array([0.25]),
      (_parameters, states) => (states[0] - 2) ** 2 / 2, {
        dfdp: new Float64Array([0]), dfdx: new Float64Array([-1.75]),
        dcdp: new Matrix([[-1]]), dcdx: new Matrix([[1]])
      }, {});
    // S=1 and f_xx=1. The full reduced Hessian would instead be 6p²-4 = -2.5.
    expect(hessian.get(0, 0)).toBeCloseTo(1, 5);
  });

  it('requires the same number of objective samples with one parameter and many states', () => {
    const counts = [2, 40].map(stateCount => {
      let evaluations = 0;
      const hessian = computeApproximateAdjointHessian(new Float64Array([0]), new Float64Array(stateCount),
        (_parameters, states) => { evaluations++; return states[0] ** 2; }, {
          dfdp: new Float64Array([0]), dfdx: new Float64Array(stateCount),
          dcdp: Matrix.ones(stateCount, 1).mul(-1), dcdx: Matrix.eye(stateCount)
        }, {});
      expect(hessian.get(0, 0)).toBeCloseTo(2, 6);
      return evaluations;
    });
    expect(counts[0]).toBe(counts[1]);
    expect(counts[0]).toBeLessThan(10);
  });

  it('rejects an asymmetric analytical Hessian rather than silently changing its meaning', () => {
    expect(() => computeApproximateAdjointHessian(parameters, states, coupledCost, partials,
      { d2fdp2: () => new Matrix([[1, 2], [0, 1]]) })).toThrow(/symmetric/);
  });

  it('explains non-finite finite-difference samples', () => {
    expect(() => computeApproximateAdjointHessian(parameters, states, () => Number.NaN, partials, {}))
      .toThrow(/Non-finite partial Hessian sample/);
  });

  it('solves a positive definite model without damping', () => {
    const result = computeSafeguardedNewtonDirection(new Matrix([[2, 0], [0, 4]]), new Float64Array([4, 8]), 0);
    result.direction.forEach(value => expect(value).toBeCloseTo(-2, 12));
    expect(result.damping).toBe(0);
    expect(result.usedGradientFallback).toBe(false);
  });

  it('falls back to steepest descent if shifted models cannot be solved finitely', () => {
    const result = computeSafeguardedNewtonDirection(Matrix.ones(3, 3).mul(1e308), new Float64Array([1, 2, 3]), 0);
    expect(Array.from(result.direction)).toEqual([-1, -2, -3]);
    expect(result.usedGradientFallback).toBe(true);
  });

  it('shared regularized solves do not mutate the caller Jacobian or accumulate shifts', () => {
    const jacobian = new Matrix([[2]]);
    const rhs = new Matrix([[4]]);
    const logger = new Logger(undefined, false);
    const first = solveLeastSquares(jacobian, rhs, logger, 'test', 1);
    const second = solveLeastSquares(jacobian, rhs, logger, 'test', 1);
    expect(jacobian.get(0, 0)).toBe(2);
    expect(rhs.get(0, 0)).toBe(4);
    expect(first[0]).toBeCloseTo(4 / 3, 12);
    expect(second[0]).toBe(first[0]);
  });
});

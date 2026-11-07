/** Failure and stagnation contracts for line searches and their quasi-Newton callers. */

import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';
import { adjointBfgs, adjointLbfgs, backtrackingLineSearch, bfgs, formatResult, lbfgs, strongWolfeLineSearch } from '../src/index';

const TINY_FIXED_STEP = 1e-12;
const GRADIENT_TOLERANCE = 1e-6;
const INITIAL_PARAMETERS = new Float64Array([1]);
const DESCENT_DIRECTION = new Float64Array([-1]);
const cost = (parameters: Float64Array): number => parameters[0] ** 2 / 2;
const gradient = (parameters: Float64Array): Float64Array => new Float64Array([parameters[0]]);

describe('line search exhaustion', () => {
  it('returns zero when bracketing exhausts its budget before satisfying curvature', () => {
    const result = strongWolfeLineSearch(cost, gradient, INITIAL_PARAMETERS, DESCENT_DIRECTION,
      { initialStepSize: 0.01, maxIterations: 1, wolfeC2: 0.1 });
    expect(result).toBe(0);
  });

  it('returns zero when zoom found decrease but no Strong Wolfe step', () => {
    const result = strongWolfeLineSearch(cost, gradient, INITIAL_PARAMETERS, DESCENT_DIRECTION,
      { initialStepSize: 8, maxZoomIterations: 3, wolfeC2: 0.01 });
    // Use a minimum away from every bisection sample so a positive Armijo bound cannot certify curvature.
    const shifted = strongWolfeLineSearch(p => (p[0] - 0.3) ** 2 / 2,
      p => new Float64Array([p[0] - 0.3]), INITIAL_PARAMETERS, DESCENT_DIRECTION,
      { initialStepSize: 8, maxZoomIterations: 4, wolfeC2: 0.01 });
    expect(result).toBeGreaterThan(0);
    expect(shifted).toBe(0);
  });

  it('does not return an untested Armijo step after its budget is exhausted', () => {
    expect(backtrackingLineSearch(cost, gradient, INITIAL_PARAMETERS, DESCENT_DIRECTION,
      { initialStepSize: 8, maxIterations: 1 })).toBe(0);
  });

  it.each([-1e-30, 0, Number.NaN, Number.POSITIVE_INFINITY])('does not accept an invalid Armijo initial step (%s)', initialStepSize => {
    expect(backtrackingLineSearch(cost, gradient, INITIAL_PARAMETERS, DESCENT_DIRECTION, { initialStepSize })).toBe(0);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('rejects a non-finite initial cost (%s)', value => {
    expect(strongWolfeLineSearch(() => value, gradient, INITIAL_PARAMETERS, DESCENT_DIRECTION)).toBe(0);
  });

  it('returns only steps that actually satisfy both Wolfe conditions over varied scales and budgets', () => {
    const curvatureCoefficient = 0.1;
    const armijoCoefficient = 1e-4;
    for (const scale of [1e-6, 1, 1e6]) {
      for (const initialStepSize of [0.01, 1, 100]) {
        for (const budget of [1, 3, 25]) {
          const scaledCost = (parameters: Float64Array): number => scale * cost(parameters);
          const scaledGradient = (parameters: Float64Array): Float64Array => new Float64Array([scale * parameters[0]]);
          const step = strongWolfeLineSearch(scaledCost, scaledGradient, INITIAL_PARAMETERS, DESCENT_DIRECTION,
            { initialStepSize, maxIterations: budget, maxZoomIterations: budget, wolfeC1: armijoCoefficient, wolfeC2: curvatureCoefficient });
          if (step === 0) continue;
          const point = new Float64Array([1 - step]);
          expect(scaledCost(point)).toBeLessThanOrEqual(scaledCost(INITIAL_PARAMETERS) - armijoCoefficient * step * scale);
          expect(Math.abs(scaledGradient(point)[0])).toBeLessThanOrEqual(curvatureCoefficient * scale);
        }
      }
    }
  });

  it.each([{ name: 'BFGS', solver: bfgs }, { name: 'L-BFGS', solver: lbfgs }])('$name stops on a failed search', ({ solver }) => {
    const result = solver(INITIAL_PARAMETERS, cost, gradient,
      { lineSearchOptions: { initialStepSize: 0.01, maxIterations: 1, wolfeC2: 0.1 } });
    expect(result.converged).toBe(false);
    expect(result.finalParameters[0]).toBe(INITIAL_PARAMETERS[0]);
    expect(result.finalCost).toBe(cost(INITIAL_PARAMETERS));
  });
});

describe('adjoint BFGS stagnation', () => {
  const constraint = (parameters: Float64Array, states: Float64Array): Float64Array =>
    new Float64Array([states[0] - parameters[0]]);
  const derivatives = {
    dfdp: gradient, dfdx: () => new Float64Array([0]), dcdp: () => new Matrix([[-1]]), dcdx: () => Matrix.eye(1)
  };

  it('does not certify a large remaining gradient merely because the step is tiny', () => {
    const result = adjointBfgs(INITIAL_PARAMETERS, INITIAL_PARAMETERS, cost, constraint,
      { ...derivatives, useLineSearch: false, stepSize: TINY_FIXED_STEP, tolerance: GRADIENT_TOLERANCE,
        stepTolerance: 1e-10 });
    expect(result.converged).toBe(false);
    expect(result.finalGradientNorm).toBeGreaterThan(GRADIENT_TOLERANCE);
    expect(result.finalCost).toBeCloseTo(cost(result.finalParameters), 14);
    expect(result.finalStates[0]).toBeCloseTo(result.finalParameters[0], 14);
    expect(result.stopReason).toBe('stepStagnation');
    expect(formatResult(result)).toContain('Stop reason: stepStagnation');
  });

  it('reports a verified new point if a small step actually reaches stationarity', () => {
    const initial = new Float64Array([1e-4]);
    const result = adjointBfgs(initial, initial, cost, constraint,
      { ...derivatives, useLineSearch: false, stepSize: 1, tolerance: 1e-8 });
    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(0, 14);
    expect(result.stopReason).toBe('gradientTolerance');
  });

  it('checks the current point even with a zero iteration budget', () => {
    const stationary = new Float64Array([0]);
    const result = adjointBfgs(stationary, stationary, cost, constraint, { ...derivatives, maxIterations: 0 });
    expect(result.converged).toBe(true);
    expect(result.stopReason).toBe('gradientTolerance');
    expect(result.iterations).toBe(0);
  });

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY])('rejects an invalid step tolerance (%s)', stepTolerance => {
    expect(() => adjointBfgs(INITIAL_PARAMETERS, INITIAL_PARAMETERS, cost, constraint, { ...derivatives, stepTolerance }))
      .toThrow(/stepTolerance/);
  });

  it.each([adjointBfgs, adjointLbfgs])('reports line-search failure on the last feasible iterate', solver => {
    const result = solver(INITIAL_PARAMETERS, INITIAL_PARAMETERS, cost, constraint,
      { ...derivatives, lineSearchOptions: { initialStepSize: 0.01, maxIterations: 1, wolfeC2: 0.1 } });
    expect(result.converged).toBe(false);
    expect(result.finalParameters[0]).toBe(INITIAL_PARAMETERS[0]);
    expect(result.stopReason).toBe('lineSearchFailure');
  });
});

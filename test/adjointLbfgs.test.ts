/** Tests feasible adjoint L-BFGS, bounded history, and truthful termination. */
import { Matrix } from 'ml-matrix';
import { adjointLbfgs } from '../src/index.js';

const objective = (parameters: Float64Array, states: Float64Array) =>
  (states[0] - 4) ** 2 + 0.1 * (parameters[0] - 2) ** 2;
const constraint = (parameters: Float64Array, states: Float64Array) =>
  new Float64Array([states[0] - parameters[0] ** 2]);
const derivatives = {
  dfdp: (parameters: Float64Array) => new Float64Array([0.2 * (parameters[0] - 2)]),
  dfdx: (_parameters: Float64Array, states: Float64Array) => new Float64Array([2 * (states[0] - 4)]),
  dcdp: (parameters: Float64Array) => new Matrix([[-2 * parameters[0]]]),
  dcdx: () => Matrix.eye(1)
};

describe('adjoint L-BFGS', () => {
  it('uses a dimensionless curvature threshold for objectives with large units', () => {
    const scale = 1e12;
    const result = adjointLbfgs(new Float64Array([2, 2]), new Float64Array([4]),
      parameters => scale * ((parameters[0] - 1) ** 2 + 2 * (parameters[1] - 1) ** 2) / 2,
      (parameters, states) => new Float64Array([states[0] - parameters[0] - parameters[1]]), {
        dfdp: parameters => new Float64Array([scale * (parameters[0] - 1), 2 * scale * (parameters[1] - 1)]),
        dfdx: () => new Float64Array([0]), dcdp: () => new Matrix([[-1, -1]]), dcdx: () => Matrix.eye(1),
        maxIterations: 120, tolerance: 1e-10 * scale, constraintTolerance: 1e-9,
        lineSearchOptions: { maxIterations: 60, maxZoomIterations: 60 }
      });
    expect(result.converged).toBe(true);
    expect(result.maxHistoryPairs).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThanOrEqual(10);
    result.finalParameters.forEach(value => expect(value).toBeCloseTo(1, 10));
  });

  it.each([1, 5, 10, 20])('restores nonlinear states and converges with historySize=%i', historySize => {
    const initialParameters = new Float64Array([1.4]);
    const initialStates = new Float64Array([-10]);
    const result = adjointLbfgs(initialParameters, initialStates, objective, constraint, {
      ...derivatives, historySize, tolerance: 1e-9, constraintTolerance: 1e-12
    });
    expect(result.converged).toBe(true);
    expect(result.stopReason).toBe('gradientTolerance');
    expect(result.finalParameters[0]).toBeCloseTo(2, 8);
    expect(result.finalStates[0]).toBeCloseTo(4, 8);
    expect(result.finalGradientNorm).toBeLessThanOrEqual(1e-9);
    expect(result.finalConstraintNorm).toBeLessThanOrEqual(1e-12);
    expect(result.maxHistoryPairs).toBeLessThanOrEqual(historySize);
    expect(initialParameters[0]).toBe(1.4);
    expect(initialStates[0]).toBe(-10);
  });

  it('supports residual objectives and numerical partial derivatives', () => {
    const result = adjointLbfgs(new Float64Array([1.4]), new Float64Array([1.96]),
      (parameters, states) => new Float64Array([states[0] - 4, parameters[0] - 2]), constraint,
      { tolerance: 1e-7, constraintTolerance: 1e-12 });
    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(2, 6);
  });

  it('converges on a coupled Rosenbrock problem with a short memory', () => {
    const result = adjointLbfgs(new Float64Array([-1.2, 1]), new Float64Array([0]),
      (parameters, states) => 100 * (parameters[1] - parameters[0] ** 2) ** 2 +
        (1 - parameters[0]) ** 2 + (states[0] - parameters[0] - parameters[1]) ** 2,
      (parameters, states) => new Float64Array([states[0] - parameters[0] - parameters[1]]),
      { historySize: 3, tolerance: 1e-6, constraintTolerance: 1e-12 });
    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(1, 4);
    expect(result.finalParameters[1]).toBeCloseTo(1, 4);
    expect(result.maxHistoryPairs).toBe(3);
  });

  it('does not call a tiny fixed step convergence', () => {
    const result = adjointLbfgs(new Float64Array([1]), new Float64Array([1]),
      parameters => parameters[0] ** 2, (parameters, states) => new Float64Array([states[0] - parameters[0]]),
      { useLineSearch: false, stepSize: 1e-30 });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('stepStagnation');
    expect(result.finalGradientNorm).toBeGreaterThan(1);
  });

  it('rejects an exhausted line search that returns an increasing trial', () => {
    const result = adjointLbfgs(new Float64Array([1]), new Float64Array([1]),
      parameters => parameters[0] ** 2, (parameters, states) => new Float64Array([states[0] - parameters[0]]),
      { lineSearchOptions: { initialStepSize: 100, maxIterations: 0 } });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('lineSearchFailure');
    expect(result.finalParameters[0]).toBe(1);
  });

  it('reports max iterations and accepts an initially optimal feasible point', () => {
    const unfinished = adjointLbfgs(new Float64Array([1.4]), new Float64Array([1.96]),
      objective, constraint, { ...derivatives, maxIterations: 0 });
    expect(unfinished.stopReason).toBe('maxIterations');
    expect(unfinished.converged).toBe(false);
    const optimal = adjointLbfgs(new Float64Array([2]), new Float64Array([4]),
      objective, constraint, { ...derivatives });
    expect(optimal.converged).toBe(true);
    expect(optimal.iterations).toBe(0);
  });

  it('rejects a step outside the implicit state manifold', () => {
    const result = adjointLbfgs(new Float64Array([1]), new Float64Array([1]),
      parameters => parameters[0] ** 2,
      (parameters, states) => new Float64Array([Math.exp(states[0]) - parameters[0] * Math.E]),
      { useLineSearch: false, stepSize: 1 });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('stateRestorationFailure');
    expect(result.finalParameters[0]).toBe(1);
  });

  it.each([0, -1, 1.5, NaN, Infinity])('rejects invalid historySize=%s', historySize => {
    expect(() => adjointLbfgs(new Float64Array([1]), new Float64Array([1]), objective, constraint,
      { historySize })).toThrow('historySize');
  });
});

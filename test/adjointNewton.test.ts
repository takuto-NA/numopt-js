/**
 * Newton-adjoint contracts and scenarios: exact-model speedup, omitted terms, and safeguard failures.
 */

import { Matrix } from 'ml-matrix';
import { adjointBfgs, adjointNewton, formatResult } from '../src/index';
import { createNewtonAdjointProblems } from '../examples/adjoint-newton-problems';
import type { AdjointNewtonOptions, ConstrainedCostFn } from '../src/core/types';

const SOLUTION_TOLERANCE = 1e-6;
const simpleCost: ConstrainedCostFn = (parameters, states) => parameters[0] ** 2 + states[0] ** 2;
const simpleConstraint = (parameters: Float64Array, states: Float64Array) => new Float64Array([parameters[0] + states[0] - 1]);
const simpleDerivatives: AdjointNewtonOptions = {
  dfdp: parameters => new Float64Array([2 * parameters[0]]),
  dfdx: (_parameters, states) => new Float64Array([2 * states[0]]),
  dcdp: () => new Matrix([[1]]), dcdx: () => new Matrix([[1]]),
  d2fdp2: () => new Matrix([[2]]), d2fdx2: () => new Matrix([[2]])
};

describe('adjointNewton', () => {
  it.each(['analytical', 'numerical'])('solves the quadratic with %s partial Hessians', mode => {
    const parameters = new Float64Array([2]);
    const states = new Float64Array([-1]);
    const result = adjointNewton(parameters, states, simpleCost, simpleConstraint,
      mode === 'analytical' ? simpleDerivatives : {});
    expect(result.converged).toBe(true);
    expect(result.stopReason).toBe('gradientTolerance');
    expect(result.finalParameters[0]).toBeCloseTo(0.5, 6);
    expect(result.finalStates[0]).toBeCloseTo(0.5, 6);
    expect(result.finalConstraintNorm).toBeLessThan(SOLUTION_TOLERANCE);
    expect(result.finalGradientNorm).toBeLessThan(SOLUTION_TOLERANCE);
    expect(result.iterations).toBe(1);
    expect(parameters[0]).toBe(2);
    expect(states[0]).toBe(-1);
    expect(formatResult(result)).toContain('Optimized states');
    expect(formatResult(result)).toContain('Stop reason: gradientTolerance');
  });

  for (const problem of createNewtonAdjointProblems()) {
    it(`solves ${problem.name} in analytical and numerical modes`, () => {
      for (const derivatives of [problem.derivatives, {}]) {
        const result = adjointNewton(problem.parameters, problem.states, problem.cost, problem.constraint,
          { ...derivatives, maxIterations: 300 });
        expect(result.converged).toBe(true);
        expect(result.finalGradientNorm).toBeLessThanOrEqual(SOLUTION_TOLERANCE);
        expect(result.finalConstraintNorm).toBeLessThanOrEqual(SOLUTION_TOLERANCE);
        result.finalParameters.forEach((value, index) => expect(value).toBeCloseTo(problem.expected[index], 5));
      }
    });
  }

  it('needs fewer iterations than BFGS when the section-12 model is exact', () => {
    const problem = createNewtonAdjointProblems()[0];
    const newton = adjointNewton(problem.parameters, problem.states, problem.cost, problem.constraint, problem.derivatives);
    const bfgs = adjointBfgs(problem.parameters, problem.states, problem.cost, problem.constraint, problem.derivatives);
    expect(newton.iterations).toBe(1);
    expect(newton.iterations).toBeLessThan(bfgs.iterations);
  });

  it('exposes linear rather than quadratic convergence when mixed partials are omitted', () => {
    const problem = createNewtonAdjointProblems()[3];
    const errors: number[] = [];
    const result = adjointNewton(problem.parameters, problem.states, problem.cost, problem.constraint, {
      ...problem.derivatives, onIteration: (_iteration, _cost, parameters) => errors.push(Math.abs(parameters[0] - 1))
    });
    const bfgs = adjointBfgs(problem.parameters, problem.states, problem.cost, problem.constraint, problem.derivatives);
    expect(result.converged).toBe(true);
    expect(errors[1] / errors[0]).toBeCloseTo(2 / 3, 6);
    expect(errors[2] / errors[1]).toBeCloseTo(2 / 3, 6);
    expect(result.iterations).toBeGreaterThan(bfgs.iterations);
  });

  it('accepts residual input and interprets partial Hessians as derivatives of the scalar cost', () => {
    const problem = createNewtonAdjointProblems()[0];
    const result = adjointNewton(problem.parameters, problem.states, problem.residual!, problem.constraint, problem.derivatives);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(1);
  });

  it('restores a mildly infeasible start and isolates callback mutations', () => {
    const problem = createNewtonAdjointProblems()[2];
    const result = adjointNewton(problem.parameters, new Float64Array([1.3]), problem.cost, problem.constraint, {
      ...problem.derivatives,
      onIteration: (_iteration, _cost, parameters) => { parameters[0] = Number.NaN; }
    });
    expect(result.converged).toBe(true);
    expect(result.finalConstraintNorm).toBeLessThan(SOLUTION_TOLERANCE);
  });

  it('damps negative curvature and makes progress on a bounded nonconvex objective', () => {
    const result = adjointNewton(new Float64Array([0.1]), new Float64Array([0]),
      parameters => parameters[0] ** 4 / 4 - parameters[0] ** 2 / 2,
      (_parameters, states) => new Float64Array([states[0]]), {
        dfdp: parameters => new Float64Array([parameters[0] ** 3 - parameters[0]]),
        dfdx: () => new Float64Array([0]), dcdp: () => new Matrix([[0]]), dcdx: () => new Matrix([[1]]),
        d2fdp2: parameters => new Matrix([[3 * parameters[0] ** 2 - 1]]), d2fdx2: () => new Matrix([[0]])
      });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0])).toBeCloseTo(1, 5);
    expect(result.dampedSteps).toBeGreaterThan(0);
    expect(result.finalCost).toBeCloseTo(-0.25, 6);
  });

  it('does not accept an exhausted Armijo search', () => {
    const result = adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint, {
      ...simpleDerivatives, lineSearchOptions: { initialStepSize: 10, maxIterations: 1 }
    });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('lineSearchFailed');
    expect(result.finalParameters[0]).toBe(2);
  });

  it('does not claim convergence when a step rounds to zero', () => {
    const result = adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint, {
      ...simpleDerivatives, lineSearchOptions: { initialStepSize: 1e-20 }
    });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('stepStagnation');
    expect(result.finalGradientNorm).toBeGreaterThan(1);
  });

  it('reports cost stagnation instead of cycling when objective changes round away', () => {
    const problem = createNewtonAdjointProblems()[3];
    const result = adjointNewton(problem.parameters, problem.states,
      (parameters, states) => 1e16 + problem.cost(parameters, states), problem.constraint, problem.derivatives);
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('costStagnation');
    expect(result.iterations).toBeLessThan(10);
  });

  it('rejects undefined trial costs without reporting convergence', () => {
    const result = adjointNewton(new Float64Array([2]), new Float64Array([-1]),
      (parameters, states) => parameters[0] === 2 ? simpleCost(parameters, states) : Number.POSITIVE_INFINITY,
      simpleConstraint, { ...simpleDerivatives, lineSearchOptions: { maxIterations: 5 } });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('lineSearchFailed');
    expect(result.finalParameters[0]).toBe(2);
  });

  it('backtracks when the proposed parameter has no real implicit state', () => {
    const result = adjointNewton(new Float64Array([1]), new Float64Array([1]),
      parameters => (parameters[0] + 1) ** 2,
      (parameters, states) => new Float64Array([states[0] ** 2 - parameters[0]]), {
        maxIterations: 1,
        dfdp: parameters => new Float64Array([2 * (parameters[0] + 1)]), dfdx: () => new Float64Array([0]),
        dcdp: () => new Matrix([[-1]]), dcdx: (_parameters, states) => new Matrix([[2 * states[0]]]),
        d2fdp2: () => new Matrix([[2]]), d2fdx2: () => new Matrix([[0]])
      });
    expect(result.converged).toBe(false);
    expect(result.finalParameters[0]).toBeGreaterThan(0);
    expect(result.finalParameters[0]).toBeLessThan(1);
    expect(result.finalConstraintNorm).toBeLessThan(SOLUTION_TOLERANCE);
  });

  it('returns final diagnostics at the iteration limit', () => {
    const result = adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint,
      { ...simpleDerivatives, maxIterations: 0 });
    expect(result.converged).toBe(false);
    expect(result.stopReason).toBe('maxIterations');
    expect(result.hessianEvaluations).toBe(0);
    expect(result.usedLineSearch).toBe(false);
  });

  it('rejects non-square implicit states and unprojectable starts', () => {
    expect(() => adjointNewton(new Float64Array([1]), new Float64Array([1, 1]), simpleCost, simpleConstraint)).toThrow(/square/);
    const circle = createNewtonAdjointProblems()[2];
    expect(() => adjointNewton(new Float64Array([1.5]), circle.states, circle.cost, circle.constraint)).toThrow(/restore feasible/);
  });

  it('rejects non-finite initial constraint values', () => {
    expect(() => adjointNewton(new Float64Array([1]), new Float64Array([1]), simpleCost,
      () => new Float64Array([Number.NaN]), simpleDerivatives)).toThrow(/restore feasible/);
  });

  it('keeps caller-owned analytical Hessians and Jacobians unchanged across iterations', () => {
    const jacobian = new Matrix([[1]]);
    const hessian = new Matrix([[2]]);
    const result = adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint, {
      ...simpleDerivatives, dcdp: () => jacobian, dcdx: () => jacobian,
      d2fdp2: () => hessian, d2fdx2: () => hessian, hessianDamping: 0.1
    });
    expect(result.converged).toBe(true);
    expect(jacobian.get(0, 0)).toBe(1);
    expect(hessian.get(0, 0)).toBe(2);
    expect(result.dampedSteps).toBeGreaterThan(0);
  });

  it('rejects singular state sensitivities instead of claiming a locally unique state', () => {
    expect(() => adjointNewton(new Float64Array([2]), new Float64Array([0]), simpleCost,
      () => new Float64Array([0]), { ...simpleDerivatives, dcdx: () => new Matrix([[0]]) }))
      .toThrow(/nonsingular implicit-state Jacobian/);
  });

  it.each([
    { hessianStepSizeP: 0 }, { hessianStepSizeX: Number.NaN }, { hessianDamping: -1 },
    { regularization: -1 }, { maxIterations: 0.5 }, { tolerance: 0 },
    { lineSearchOptions: { contractionFactor: 1 } }, { lineSearchOptions: { armijoParameter: 1 } }
  ])('rejects invalid options %j', options => {
    expect(() => adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint, options)).toThrow();
  });

  it.each([new Matrix([[1, 0]]), new Matrix([[Number.NaN]])])('rejects malformed partial Hessians', hessian => {
    expect(() => adjointNewton(new Float64Array([2]), new Float64Array([-1]), simpleCost, simpleConstraint,
      { ...simpleDerivatives, d2fdp2: () => hessian })).toThrow(/d2fdp2/);
  });
});

/**
 * Full Newton-adjoint scenarios: omitted terms, local quadratic convergence, and analytical action contracts.
 */

import { Matrix } from 'ml-matrix';
import { adjointNewton, adjointNewtonExact } from '../src/index';
import { createNewtonAdjointProblems, createQuadraticConvergenceProblem } from '../examples/adjoint-newton-problems';
import { solveLeastSquares } from '../src/core/constrainedUtils';
import { Logger } from '../src/core/logger';

const ACCURATE_OPTIONS = { tolerance: 1e-9, constraintTolerance: 1e-12, maxIterations: 100 };

describe('adjointNewtonExact', () => {
  for (const problem of createNewtonAdjointProblems()) {
    it(`solves ${problem.name} with analytical and numerical full curvature`, () => {
      for (const analytical of [true, false]) {
        const result = adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint, {
          ...ACCURATE_OPTIONS,
          ...(analytical ? { ...problem.derivatives, lagrangianHessian: problem.exactDerivatives.lagrangianHessian } : {})
        });
        expect(result.converged).toBe(true);
        expect(result.finalGradientNorm).toBeLessThanOrEqual(ACCURATE_OPTIONS.tolerance);
        expect(result.finalConstraintNorm).toBeLessThanOrEqual(ACCURATE_OPTIONS.constraintTolerance);
        result.finalParameters.forEach((value, index) => expect(value).toBeCloseTo(problem.expected[index], 7));
      }
    });
  }

  it.each([3, 4])('resolves omitted-curvature counterexample %i in one undamped step', index => {
    const problem = createNewtonAdjointProblems()[index];
    const exact = adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint,
      { ...problem.derivatives, ...problem.exactDerivatives, ...ACCURATE_OPTIONS });
    const approximate = adjointNewton(problem.parameters, problem.states, problem.cost, problem.constraint,
      { ...problem.derivatives, ...ACCURATE_OPTIONS });
    expect(exact.converged).toBe(true);
    expect(exact.iterations).toBe(1);
    expect(exact.dampedSteps).toBe(0);
    expect(approximate.iterations).toBeGreaterThan(exact.iterations);
  });

  it('shows local quadratic convergence with a nonzero adjoint and no damping', () => {
    const problem = createQuadraticConvergenceProblem();
    const errors: number[] = [];
    const exact = adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint, {
      ...problem.derivatives, ...problem.exactDerivatives,
      tolerance: 1e-10, constraintTolerance: 1e-14,
      onIteration: (_iteration, _cost, parameters) => errors.push(Math.abs(parameters[0] - 1))
    });
    expect(exact.converged).toBe(true);
    expect(exact.dampedSteps).toBe(0);
    expect(errors.length).toBeGreaterThanOrEqual(5);
    for (let index = 1; index < 4; index++) {
      expect(errors[index + 1] / errors[index] ** 2).toBeGreaterThan(0.2);
      expect(errors[index + 1] / errors[index] ** 2).toBeLessThan(0.6);
    }
  });

  it('accepts an analytical Hessian-vector callback without allocating a joint Hessian', () => {
    const problem = createNewtonAdjointProblems()[1];
    const result = adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint, {
      ...problem.derivatives, lagrangianHessianVector: problem.exactDerivatives.lagrangianHessianVector, ...ACCURATE_OPTIONS
    });
    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(1);
  });

  it('supports residual input with the Lagrangian Hessian of its scalar cost', () => {
    const problem = createNewtonAdjointProblems()[0];
    const result = adjointNewtonExact(problem.parameters, problem.states, problem.residual!, problem.constraint,
      { ...problem.derivatives, ...problem.exactDerivatives });
    expect(result.converged).toBe(true);
  });

  it('rejects competing analytical curvature callbacks and malformed Hessian products', () => {
    const problem = createNewtonAdjointProblems()[1];
    expect(() => adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint,
      { ...problem.derivatives, ...problem.exactDerivatives })).toThrow(/only one/);
    expect(() => adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint,
      { ...problem.derivatives, lagrangianHessianVector: () => new Float64Array([1]) })).toThrow(/finite entries/);
    expect(() => adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint,
      { ...problem.derivatives, lagrangianHessian: () => new Matrix([[1]]) })).toThrow(/lagrangianHessian/);
  });

  it('rejects invalid finite-difference step settings', () => {
    const problem = createNewtonAdjointProblems()[0];
    expect(() => adjointNewtonExact(problem.parameters, problem.states, problem.cost, problem.constraint,
      { hessianStepSize: 0 })).toThrow(/hessianStepSize/);
  });

  it('safeguards indefinite exact curvature away from a local minimum', () => {
    const result = adjointNewtonExact(new Float64Array([0.1]), new Float64Array([0]),
      parameters => parameters[0] ** 4 / 4 - parameters[0] ** 2 / 2,
      (_parameters, states) => new Float64Array([states[0]]), {
        dfdp: parameters => new Float64Array([parameters[0] ** 3 - parameters[0]]),
        dfdx: () => new Float64Array([0]), dcdp: () => new Matrix([[0]]), dcdx: () => new Matrix([[1]]),
        lagrangianHessian: parameters => new Matrix([[3 * parameters[0] ** 2 - 1, 0], [0, 0]])
      });
    expect(result.converged).toBe(true);
    expect(result.dampedSteps).toBeGreaterThan(0);
    expect(result.finalCost).toBeCloseTo(-0.25, 6);
  });

  it('does not introduce an unconditional bias into nonsingular state and adjoint solves', () => {
    const solution = solveLeastSquares(new Matrix([[1]]), new Matrix([[1]]), new Logger(undefined, false));
    expect(solution[0]).toBeCloseTo(1, 14);
    const singularFallback = solveLeastSquares(new Matrix([[0]]), new Matrix([[1]]), new Logger(undefined, false));
    expect(Number.isFinite(singularFallback[0])).toBe(true);
  });
});

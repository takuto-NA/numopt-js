import { Matrix } from 'ml-matrix';
import { adjointBfgs } from '../src/core/adjointBfgs';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';

describe('Adjoint BFGS', () => {
  const simpleCost: ConstrainedCostFn = (parameters, states) => {
    return parameters[0] * parameters[0] + states[0] * states[0];
  };

  const simpleConstraint: ConstraintFn = (parameters, states) => {
    return new Float64Array([parameters[0] + states[0] - 1.0]);
  };

  it('converges to the simple equality minimum with finite differences', () => {
    const result = adjointBfgs(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalCost - 0.5)).toBeLessThan(1e-3);
    expect(vectorNorm(simpleConstraint(result.finalParameters, result.finalStates))).toBeLessThan(1e-3);
    expect(result.finalConstraintNorm).toBeDefined();
    expect(result.usedLineSearch).toBe(true);
  });

  it('converges with analytical reduced-space derivatives', () => {
    const result = adjointBfgs(
      new Float64Array([1.0]),
      new Float64Array([0.0]),
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6,
        dfdp: (parameters) => new Float64Array([2 * parameters[0]]),
        dfdx: (_parameters, states) => new Float64Array([2 * states[0]]),
        dcdp: () => new Matrix([[1]]),
        dcdx: () => new Matrix([[1]])
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-3);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-3);
  });

  it('accepts options.regularization on the simple problem', () => {
    const result = adjointBfgs(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      simpleCost,
      simpleConstraint,
      {
        maxIterations: 200,
        tolerance: 1e-4,
        regularization: 1e-6
      }
    );

    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(result.finalConstraintNorm).toBeLessThan(1e-3);
  });

  it('rejects an overdetermined constraint Jacobian', () => {
    const overdeterminedConstraint: ConstraintFn = (parameters, states) => {
      return new Float64Array([parameters[0] + states[0] - 1.0, 2.0 * parameters[0] + states[0] - 1.5]);
    };

    expect(() =>
      adjointBfgs(
        new Float64Array([0.5]),
        new Float64Array([0.5]),
        simpleCost,
        overdeterminedConstraint
      )
    ).toThrow(/square implicit-state system/);
  });

  it('rejects an underdetermined constraint Jacobian', () => {
    const underdeterminedCost: ConstrainedCostFn = (parameters, states) => {
      return parameters[0] * parameters[0] + states[0] * states[0] + states[1] * states[1];
    };
    const underdeterminedConstraint: ConstraintFn = (parameters, states) => {
      return new Float64Array([parameters[0] + states[0] + states[1] - 1.0]);
    };

    expect(() =>
      adjointBfgs(
        new Float64Array([1.0]),
        new Float64Array([0.0, 0.0]),
        underdeterminedCost,
        underdeterminedConstraint
      )
    ).toThrow(/square implicit-state system/);
  });

  it('converges with a fixed step when line search is disabled', () => {
    const result = adjointBfgs(
      new Float64Array([3.0]),
      new Float64Array([-2.0]),
      simpleCost,
      simpleConstraint,
      {
        stepSize: 0.1,
        useLineSearch: false,
        maxIterations: 1000,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(result.usedLineSearch).toBe(false);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
  });

  it('rejects a non-square analytical ∂c/∂x before projecting', () => {
    expect(() =>
      adjointBfgs(new Float64Array([0.5]), new Float64Array([0.5]), simpleCost, simpleConstraint, {
        dcdx: () => new Matrix([[1, 0]])
      })
    ).toThrow(/square constraint Jacobian/);
  });

  it('rejects an initial point with no real implicit state', () => {
    const circleCost: ConstrainedCostFn = (parameters, states) =>
      (parameters[0] - 1) ** 2 + (states[0] - 1) ** 2;
    const circleConstraint: ConstraintFn = (parameters, states) =>
      new Float64Array([parameters[0] * parameters[0] + states[0] * states[0] - 2.0]);

    expect(() =>
      adjointBfgs(new Float64Array([1.5]), new Float64Array([0.5]), circleCost, circleConstraint, {
        constraintTolerance: 1e-6
      })
    ).toThrow(/Failed to restore feasible states/);
  });

  it('does not report convergence when every trial cost is undefined', () => {
    const start = 1.0;
    const onlyDefinedAtStart: ConstrainedCostFn = (parameters, states) => {
      if (Math.abs(parameters[0] - start) > 1e-15) {
        return Number.POSITIVE_INFINITY;
      }
      return parameters[0] * parameters[0] + states[0] * states[0];
    };

    const result = adjointBfgs(
      new Float64Array([start]),
      new Float64Array([0.0]),
      onlyDefinedAtStart,
      simpleConstraint,
      { maxIterations: 10, tolerance: 1e-12 }
    );

    expect(result.converged).toBe(false);
    expect(result.usedLineSearch).toBe(true);
    expect(result.finalParameters[0]).toBe(start);
    expect(result.iterations).toBeLessThan(10);
  });

  it('does not claim line search was used when a zero fixed step stops the loop', () => {
    const result = adjointBfgs(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      simpleCost,
      simpleConstraint,
      {
        useLineSearch: false,
        stepSize: 0,
        maxIterations: 5,
        tolerance: 1e-12
      }
    );

    expect(result.converged).toBe(false);
    expect(result.usedLineSearch).toBe(false);
    expect(result.finalParameters[0]).toBe(2.0);
  });
});

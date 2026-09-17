/**
 * High-signal reduced-space cases that the adjointBfgs contract suite does not cover:
 * nonlinear circle, implicit chain (n=20, same as the paper example), two-parameter
 * affine manifold, residual API, and the public package entry.
 */

import { Matrix } from 'ml-matrix';
import { adjointBfgs, adjointGradientDescent, constrainedGaussNewton } from '../src/index';
import type { ConstrainedCostFn, ConstraintFn } from '../src/core/types';
import { vectorNorm } from '../src/utils/matrix';
import {
  CONSTRAINED_LS_PARAMETER_TOLERANCE,
  CONSTRAINED_LS_TARGET_PARAMETER,
  CONSTRAINED_LS_TARGET_STATE,
  constrainedLeastSquaresConstraint,
  constrainedLeastSquaresResidual,
  createConstrainedLeastSquaresInitial
} from './fixtures/constrainedLeastSquares';

const SHARED_MAX_ITERATIONS = 300;
const SHARED_TOLERANCE = 1e-6;
const SHARED_CONSTRAINT_TOLERANCE = 1e-6;

describe('Adjoint BFGS scenarios', () => {
  it('solves the nonlinear circle with fewer iterations than adjoint gradient descent', () => {
    const circleCost: ConstrainedCostFn = (parameters, states) =>
      (parameters[0] - 1) ** 2 + (states[0] - 1) ** 2;
    const circleConstraint: ConstraintFn = (parameters, states) =>
      new Float64Array([parameters[0] * parameters[0] + states[0] * states[0] - 2.0]);
    const initialParameter = 0.2;
    const initialState = Math.sqrt(2 - initialParameter * initialParameter);
    const sharedOptions = {
      maxIterations: SHARED_MAX_ITERATIONS,
      tolerance: SHARED_TOLERANCE,
      constraintTolerance: SHARED_CONSTRAINT_TOLERANCE,
      useLineSearch: true,
      dfdp: (parameters: Float64Array) => new Float64Array([2 * (parameters[0] - 1)]),
      dfdx: (_parameters: Float64Array, states: Float64Array) => new Float64Array([2 * (states[0] - 1)]),
      dcdp: (parameters: Float64Array) => new Matrix([[2 * parameters[0]]]),
      dcdx: (_parameters: Float64Array, states: Float64Array) => new Matrix([[2 * states[0]]])
    };

    const gradientDescentResult = adjointGradientDescent(
      new Float64Array([initialParameter]),
      new Float64Array([initialState]),
      circleCost,
      circleConstraint,
      sharedOptions
    );
    const bfgsResult = adjointBfgs(
      new Float64Array([initialParameter]),
      new Float64Array([initialState]),
      circleCost,
      circleConstraint,
      sharedOptions
    );

    expect(bfgsResult.converged).toBe(true);
    expect(bfgsResult.finalParameters[0]).toBeCloseTo(1, 2);
    expect(bfgsResult.finalStates[0]).toBeCloseTo(1, 2);
    expect(bfgsResult.finalConstraintNorm).toBeLessThan(SHARED_CONSTRAINT_TOLERANCE);
    expect(gradientDescentResult.converged).toBe(true);
    expect(bfgsResult.iterations).toBeLessThan(gradientDescentResult.iterations);
  });

  it('solves an implicit chain with one parameter and fewer iterations than adjoint GD', () => {
    const stateCount = 20;
    const targetEnd = 1.0;
    const expectedParameter = targetEnd / stateCount;
    const chainCost: ConstrainedCostFn = (_parameters, states) => (states[stateCount - 1] - targetEnd) ** 2;
    const chainConstraint: ConstraintFn = (parameters, states) => {
      const constraint = new Float64Array(stateCount);
      constraint[0] = states[0] - parameters[0];
      for (let index = 1; index < stateCount; index++) {
        constraint[index] = states[index] - states[index - 1] - parameters[0];
      }
      return constraint;
    };
    const initialParameter = 0.2;
    const initialStates = new Float64Array(stateCount);
    for (let index = 0; index < stateCount; index++) {
      initialStates[index] = (index + 1) * initialParameter;
    }

    const dcdp = () => new Matrix(Array.from({ length: stateCount }, () => [-1]));
    const dcdx = () => {
      const rows = Array.from({ length: stateCount }, (_, row) => {
        const values = new Array<number>(stateCount).fill(0);
        values[row] = 1;
        if (row > 0) {
          values[row - 1] = -1;
        }
        return values;
      });
      return new Matrix(rows);
    };
    const sharedOptions = {
      maxIterations: 200,
      tolerance: 1e-8,
      constraintTolerance: 1e-8,
      useLineSearch: true,
      dfdp: () => new Float64Array([0]),
      dfdx: (_parameters: Float64Array, states: Float64Array) => {
        const gradient = new Float64Array(stateCount);
        gradient[stateCount - 1] = 2 * (states[stateCount - 1] - targetEnd);
        return gradient;
      },
      dcdp,
      dcdx
    };

    const gradientDescentResult = adjointGradientDescent(
      new Float64Array([initialParameter]),
      new Float64Array(initialStates),
      chainCost,
      chainConstraint,
      sharedOptions
    );
    const bfgsResult = adjointBfgs(
      new Float64Array([initialParameter]),
      new Float64Array(initialStates),
      chainCost,
      chainConstraint,
      sharedOptions
    );

    expect(bfgsResult.converged).toBe(true);
    expect(bfgsResult.finalParameters[0]).toBeCloseTo(expectedParameter, 3);
    expect(bfgsResult.finalParameters.length).toBe(1);
    expect(bfgsResult.finalStates.length).toBe(stateCount);
    expect(bfgsResult.finalConstraintNorm).toBeLessThan(1e-6);
    expect(gradientDescentResult.converged).toBe(true);
    expect(bfgsResult.iterations).toBeLessThan(gradientDescentResult.iterations);
  });

  it('solves a two-parameter affine implicit-state problem with fewer iterations than adjoint GD', () => {
    const cost2D: ConstrainedCostFn = (parameters, states) =>
      (parameters[0] - 1) ** 2 + (parameters[1] - 2) ** 2 + states[0] * states[0] + states[1] * states[1];
    const constraint2D: ConstraintFn = (parameters, states) =>
      new Float64Array([parameters[0] + states[0] - 1.0, parameters[1] + states[1] - 2.0]);
    const sharedOptions = {
      maxIterations: 200,
      tolerance: 1e-6,
      useLineSearch: true,
      dfdp: (parameters: Float64Array) =>
        new Float64Array([2 * (parameters[0] - 1), 2 * (parameters[1] - 2)]),
      dfdx: (_parameters: Float64Array, states: Float64Array) =>
        new Float64Array([2 * states[0], 2 * states[1]]),
      dcdp: () => new Matrix([[1, 0], [0, 1]]),
      dcdx: () => new Matrix([[1, 0], [0, 1]])
    };

    const gradientDescentResult = adjointGradientDescent(
      new Float64Array([3.0, 4.0]),
      new Float64Array([-2.0, -2.0]),
      cost2D,
      constraint2D,
      sharedOptions
    );
    const bfgsResult = adjointBfgs(
      new Float64Array([3.0, 4.0]),
      new Float64Array([-2.0, -2.0]),
      cost2D,
      constraint2D,
      sharedOptions
    );

    expect(bfgsResult.converged).toBe(true);
    expect(bfgsResult.finalParameters[0]).toBeCloseTo(1, 2);
    expect(bfgsResult.finalParameters[1]).toBeCloseTo(2, 2);
    expect(bfgsResult.finalStates[0]).toBeCloseTo(0, 2);
    expect(bfgsResult.finalStates[1]).toBeCloseTo(0, 2);
    expect(bfgsResult.finalConstraintNorm).toBeLessThan(1e-6);
    expect(gradientDescentResult.converged).toBe(true);
    expect(bfgsResult.iterations).toBeLessThan(gradientDescentResult.iterations);
  });

  it('converges on residual least squares without needing to beat constrained GN', () => {
    const initial = createConstrainedLeastSquaresInitial();
    const bfgsResult = adjointBfgs(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );
    const gaussNewtonResult = constrainedGaussNewton(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(bfgsResult.converged).toBe(true);
    expect(Math.abs(bfgsResult.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(Math.abs(bfgsResult.finalStates[0] - CONSTRAINED_LS_TARGET_STATE)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(gaussNewtonResult.converged).toBe(true);
    expect(
      vectorNorm(
        constrainedLeastSquaresConstraint(bfgsResult.finalParameters, bfgsResult.finalStates)
      )
    ).toBeLessThan(1e-3);
  });

  it('is reachable from the public package entry', () => {
    const result = adjointBfgs(
      new Float64Array([2.0]),
      new Float64Array([-1.0]),
      (parameters, states) => parameters[0] * parameters[0] + states[0] * states[0],
      (parameters, states) => new Float64Array([parameters[0] + states[0] - 1.0]),
      { maxIterations: 100, tolerance: 1e-6 }
    );

    expect(result.converged).toBe(true);
    expect(result.finalParameters[0]).toBeCloseTo(0.5, 2);
    expect(result.finalStates[0]).toBeCloseTo(0.5, 2);
  });
});

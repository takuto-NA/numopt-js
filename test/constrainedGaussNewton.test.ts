import { constrainedGaussNewton } from '../src/core/constrainedGaussNewton';
import { vectorNorm } from '../src/utils/matrix';
import {
  CONSTRAINED_LS_CONSTRAINT_TOLERANCE,
  CONSTRAINED_LS_COST_TOLERANCE,
  CONSTRAINED_LS_PARAMETER_TOLERANCE,
  CONSTRAINED_LS_TARGET_PARAMETER,
  CONSTRAINED_LS_TARGET_STATE,
  constrainedLeastSquaresConstraint,
  constrainedLeastSquaresResidual,
  createConstrainedLeastSquaresAnalyticalDerivatives,
  createConstrainedLeastSquaresInitial,
  halfSquaredResidualNorm
} from './fixtures/constrainedLeastSquares';

const MAX_ITERATIONS_SHORT = 3;
const STRICT_TOLERANCE = 1e-12;

describe('Constrained Gauss-Newton Method', () => {
  it('should converge for simple constrained least squares', () => {
    const initial = createConstrainedLeastSquaresInitial();

    const result = constrainedGaussNewton(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(Math.abs(result.finalStates[0] - CONSTRAINED_LS_TARGET_STATE)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(result.finalCost).toBeLessThan(CONSTRAINED_LS_COST_TOLERANCE);
    expect(
      vectorNorm(
        constrainedLeastSquaresConstraint(result.finalParameters, result.finalStates)
      )
    ).toBeLessThan(CONSTRAINED_LS_CONSTRAINT_TOLERANCE);
  });

  it('should work with analytical derivatives', () => {
    const initial = { parameters: new Float64Array([1.0]), states: new Float64Array([0.0]) };
    const derivatives = createConstrainedLeastSquaresAnalyticalDerivatives();

    const result = constrainedGaussNewton(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolerance: 1e-6,
        ...derivatives
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      1e-2
    );
  });

  it('should reduce constraint violation from an infeasible start', () => {
    const initialParameters = new Float64Array([1.0]);
    const initialStates = new Float64Array([1.0]);
    const initialConstraintNorm = vectorNorm(
      constrainedLeastSquaresConstraint(initialParameters, initialStates)
    );

    const result = constrainedGaussNewton(
      initialParameters,
      initialStates,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 200,
        tolerance: 1e-4,
        constraintTolerance: 1e-6
      }
    );

    expect(result.finalConstraintNorm).toBeLessThan(initialConstraintNorm);
    expect(result.finalConstraintNorm).toBeLessThan(1e-2);
  });

  it('should call onIteration for each completed iteration starting at zero', () => {
    const initial = createConstrainedLeastSquaresInitial();
    const iterations: number[] = [];

    const result = constrainedGaussNewton(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 10,
        tolerance: 1e-6,
        onIteration: (iteration, cost, parameters) => {
          iterations.push(iteration);
          expect(cost).toBeGreaterThanOrEqual(0);
          expect(parameters).toBeInstanceOf(Float64Array);
        }
      }
    );

    const expectedIterations = Array.from({ length: result.iterations }, (_, index) => index);
    expect(iterations).toEqual(expectedIterations);
  });

  it('should stop at maxIterations when tolerances are unreachable', () => {
    const initial = { parameters: new Float64Array([10.0]), states: new Float64Array([-9.0]) };
    const initialCost = halfSquaredResidualNorm(initial.parameters, initial.states);

    const result = constrainedGaussNewton(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: MAX_ITERATIONS_SHORT,
        tolerance: STRICT_TOLERANCE
      }
    );

    expect(result.iterations).toBeLessThanOrEqual(MAX_ITERATIONS_SHORT);
    if (!result.converged) {
      expect(result.iterations).toBe(MAX_ITERATIONS_SHORT);
    }
    expect(result.finalCost).toBeLessThanOrEqual(initialCost);
  });

  it('should converge for 2D constrained least squares', () => {
    const residual2D = (parameters: Float64Array, states: Float64Array) => {
      return new Float64Array([
        parameters[0] - 0.5,
        parameters[1] - 0.5,
        states[0] - 0.5,
        states[1] - 0.5
      ]);
    };
    const constraint2D = (parameters: Float64Array, states: Float64Array) => {
      return new Float64Array([
        parameters[0] + states[0] - 1.0,
        parameters[1] + states[1] - 1.0
      ]);
    };

    const result = constrainedGaussNewton(
      new Float64Array([2.0, 2.0]),
      new Float64Array([-1.0, -1.0]),
      residual2D,
      constraint2D,
      {
        maxIterations: 200,
        tolerance: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalParameters[1] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(1e-2);
    expect(Math.abs(result.finalStates[1] - 0.5)).toBeLessThan(1e-2);
    expect(vectorNorm(constraint2D(result.finalParameters, result.finalStates))).toBeLessThan(
      CONSTRAINED_LS_CONSTRAINT_TOLERANCE
    );
  });
});

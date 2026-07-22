import { constrainedLevenbergMarquardt } from '../src/core/constrainedLevenbergMarquardt';
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

describe('Constrained Levenberg-Marquardt Method', () => {
  it('should converge for simple constrained least squares', () => {
    const initial = createConstrainedLeastSquaresInitial();

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolGradient: 1e-6
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
    expect(result.finalLambda).toBeGreaterThan(0);
    expect(
      vectorNorm(
        constrainedLeastSquaresConstraint(result.finalParameters, result.finalStates)
      )
    ).toBeLessThan(CONSTRAINED_LS_CONSTRAINT_TOLERANCE);
  });

  it('should work with analytical derivatives', () => {
    const initial = { parameters: new Float64Array([1.0]), states: new Float64Array([0.0]) };
    const derivatives = createConstrainedLeastSquaresAnalyticalDerivatives();

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        tolGradient: 1e-6,
        ...derivatives
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      1e-2
    );
  });

  it('should handle lambda updates correctly', () => {
    const initial = { parameters: new Float64Array([3.0]), states: new Float64Array([-2.0]) };

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 100,
        lambdaInitial: 1e-3,
        lambdaFactor: 10.0,
        tolGradient: 1e-6
      }
    );

    expect(result.converged).toBe(true);
    expect(result.finalLambda).toBeGreaterThan(0);
    expect(result.finalLambda).toBeLessThan(1e3);
  });

  it('should call onIteration starting from iteration zero', () => {
    const initial = createConstrainedLeastSquaresInitial();
    const iterations: number[] = [];
    const firstParams: number[] = [];

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 10,
        tolGradient: 1e-6,
        onIteration: (iteration, cost, parameters) => {
          iterations.push(iteration);
          expect(cost).toBeGreaterThanOrEqual(0);
          if (iteration === 0) {
            firstParams.push(parameters[0]);
          }
        }
      }
    );

    const expectedIterations = Array.from({ length: result.iterations }, (_, index) => index);
    expect(iterations).toEqual(expectedIterations);
    expect(firstParams[0]).toBe(initial.parameters[0]);
  });

  it('should return a non-worsening best cost when max iterations is reached', () => {
    const initial = { parameters: new Float64Array([10.0]), states: new Float64Array([-9.0]) };
    const initialCost = halfSquaredResidualNorm(initial.parameters, initial.states);

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: MAX_ITERATIONS_SHORT,
        tolGradient: STRICT_TOLERANCE,
        tolStep: STRICT_TOLERANCE,
        tolResidual: STRICT_TOLERANCE
      }
    );

    expect(result.iterations).toBeLessThanOrEqual(MAX_ITERATIONS_SHORT);
    if (!result.converged) {
      expect(result.iterations).toBe(MAX_ITERATIONS_SHORT);
    }
    expect(result.finalCost).toBeLessThanOrEqual(initialCost);
  });
});

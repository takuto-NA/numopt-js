import { Matrix } from 'ml-matrix';
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
const OPTIMUM_START_TOLERANCE = 1e-8;
const DEFAULT_LAMBDA_INITIAL = 1e-3;
const CONSTANT_RESIDUAL_VALUE = 1;
const TIGHT_RESIDUAL_AFTER_TRUE_SOLVE = 1e-6;

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

  it('converges immediately when started at the constrained least-squares optimum', () => {
    const derivatives = createConstrainedLeastSquaresAnalyticalDerivatives();

    const result = constrainedLevenbergMarquardt(
      new Float64Array([CONSTRAINED_LS_TARGET_PARAMETER]),
      new Float64Array([CONSTRAINED_LS_TARGET_STATE]),
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 10,
        tolGradient: OPTIMUM_START_TOLERANCE,
        ...derivatives
      }
    );

    expect(result.converged).toBe(true);
    expect(result.iterations).toBe(1);
    expect(result.finalGradientNorm).toBe(0);
    expect(result.finalLambda).toBe(DEFAULT_LAMBDA_INITIAL);
    expect(result.finalCost).toBe(0);
  });

  it('stops without moving when every trial cost is unchanged', () => {
    const derivatives = createConstrainedLeastSquaresAnalyticalDerivatives();
    const constantResidual = () => new Float64Array([CONSTANT_RESIDUAL_VALUE, 0]);
    const initial = { parameters: new Float64Array([2.0]), states: new Float64Array([-1.0]) };
    const lambdaInitial = DEFAULT_LAMBDA_INITIAL;

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constantResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 3,
        lambdaInitial,
        lambdaFactor: 10,
        tolGradient: 1e-12,
        tolStep: 1e-12,
        tolResidual: 1e-12,
        ...derivatives
      }
    );

    // WHY: equal cost is a reject. Inner λ grows to the stop threshold, but
    // shouldStop does not write that λ back onto the returned result.
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(1);
    expect(result.finalParameters[0]).toBe(initial.parameters[0]);
    expect(result.finalStates[0]).toBe(initial.states[0]);
    expect(result.finalLambda).toBe(lambdaInitial);
    expect(result.finalCost).toBe(CONSTANT_RESIDUAL_VALUE * CONSTANT_RESIDUAL_VALUE);
  });

  it('uses CommonOptimizationOptions.tolerance when LM-specific tols are omitted', () => {
    const initial = createConstrainedLeastSquaresInitial();
    const fallbackTolerance = 1e-6;

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 20,
        tolerance: fallbackTolerance
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(result.finalResidualNorm).toBeLessThan(TIGHT_RESIDUAL_AFTER_TRUE_SOLVE);
  });

  it('lets an explicit tolResidual win over a loose tolerance fallback', () => {
    const initial = createConstrainedLeastSquaresInitial();
    const looseFallbackTolerance = 10;
    const explicitResidualTolerance = 1e-8;
    const tightUnspecifiedTol = 1e-12;

    const result = constrainedLevenbergMarquardt(
      initial.parameters,
      initial.states,
      constrainedLeastSquaresResidual,
      constrainedLeastSquaresConstraint,
      {
        maxIterations: 20,
        tolerance: looseFallbackTolerance,
        // WHY: unspecified tols also fall back to `tolerance`. Pin them so a
        // loose fallback cannot stop on gradient or step before residual is tested.
        tolGradient: tightUnspecifiedTol,
        tolStep: tightUnspecifiedTol,
        tolResidual: explicitResidualTolerance
      }
    );

    expect(result.converged).toBe(true);
    expect(Math.abs(result.finalParameters[0] - CONSTRAINED_LS_TARGET_PARAMETER)).toBeLessThan(
      CONSTRAINED_LS_PARAMETER_TOLERANCE
    );
    expect(result.finalResidualNorm).toBeLessThan(TIGHT_RESIDUAL_AFTER_TRUE_SOLVE);
  });
});

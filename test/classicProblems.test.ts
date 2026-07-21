/**
 * Classic problem suite: keeps high-signal integration coverage that examples alone cannot provide.
 */

import { gradientDescent } from '../src/core/gradientDescent';
import { levenbergMarquardt } from '../src/core/levenbergMarquardt';
import { constrainedLevenbergMarquardt } from '../src/core/constrainedLevenbergMarquardt';
import { vectorNorm } from '../src/utils/matrix';
import type {
  CostFn,
  GradientFn,
  ResidualFn,
  ConstrainedResidualFn,
  ConstraintFn
} from '../src/core/types';

const ROSENBROCK_PARAMETER_TOLERANCE = 0.1;
const ROSENBROCK_MAX_ITERATIONS = 10000;
const ROSENBROCK_TOLERANCE = 1e-4;

const CURVE_FIT_MAX_COST = 1.0;
const LINEAR_SLOPE_TOLERANCE = 0.1;
const LINEAR_MAX_COST = 0.1;

const CONSTRAINT_NORM_TOLERANCE = 1e-2;
const CONSTRAINED_MAX_COST = 1.0;

describe('Classic problem suite', () => {
  describe('Rosenbrock via gradient descent', () => {
    const rosenbrockCost: CostFn = (params: Float64Array) => {
      const x = params[0];
      const y = params[1];
      const term1 = 1 - x;
      const term2 = y - x * x;
      return term1 * term1 + 100 * term2 * term2;
    };

    const rosenbrockGradient: GradientFn = (params: Float64Array) => {
      const x = params[0];
      const y = params[1];
      const dx = -2 * (1 - x) - 400 * x * (y - x * x);
      const dy = 200 * (y - x * x);
      return new Float64Array([dx, dy]);
    };

    it('should converge near the global minimum', () => {
      const result = gradientDescent(
        new Float64Array([-1.0, 1.0]),
        rosenbrockCost,
        rosenbrockGradient,
        {
          maxIterations: ROSENBROCK_MAX_ITERATIONS,
          tolerance: ROSENBROCK_TOLERANCE,
          useLineSearch: true
        }
      );

      expect(result.converged).toBe(true);
      expect(Math.abs(result.finalParameters[0] - 1.0)).toBeLessThan(ROSENBROCK_PARAMETER_TOLERANCE);
      expect(Math.abs(result.finalParameters[1] - 1.0)).toBeLessThan(ROSENBROCK_PARAMETER_TOLERANCE);
    });
  });

  describe('Curve fitting via Levenberg-Marquardt', () => {
    it('should fit an exponential curve', () => {
      const xData = new Float64Array([0, 1, 2, 3, 4]);
      const yData = new Float64Array([1.0, 2.7, 7.4, 20.1, 54.6]);

      const exponentialResidual: ResidualFn = (params: Float64Array) => {
        const a = params[0];
        const b = params[1];
        const residuals = new Float64Array(xData.length);
        for (let i = 0; i < xData.length; i++) {
          residuals[i] = a * Math.exp(b * xData[i]) - yData[i];
        }
        return residuals;
      };

      const result = levenbergMarquardt(new Float64Array([1.0, 1.0]), exponentialResidual, {
        useNumericJacobian: true,
        maxIterations: 100,
        tolGradient: 1e-6
      });

      expect(result.converged).toBe(true);
      expect(result.finalCost).toBeLessThan(CURVE_FIT_MAX_COST);
    });

    it('should fit a linear model', () => {
      const xData = new Float64Array([1, 2, 3, 4, 5]);
      const yData = new Float64Array([2.1, 3.9, 6.1, 8.0, 9.9]);

      const linearResidual: ResidualFn = (params: Float64Array) => {
        const a = params[0];
        const b = params[1];
        const residuals = new Float64Array(xData.length);
        for (let i = 0; i < xData.length; i++) {
          residuals[i] = a * xData[i] + b - yData[i];
        }
        return residuals;
      };

      const result = levenbergMarquardt(new Float64Array([0.0, 0.0]), linearResidual, {
        useNumericJacobian: true,
        maxIterations: 100,
        tolGradient: 1e-6
      });

      expect(result.converged).toBe(true);
      expect(Math.abs(result.finalParameters[0] - 2.0)).toBeLessThan(LINEAR_SLOPE_TOLERANCE);
      expect(result.finalCost).toBeLessThan(LINEAR_MAX_COST);
    });
  });

  describe('Constrained Rosenbrock via constrained Levenberg-Marquardt', () => {
    const rosenbrockResidual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      const a = 1.0 - p[0];
      const b = x[0] - p[0] * p[0];
      return new Float64Array([a, 10.0 * b]);
    };

    const circleConstraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] * p[0] + x[0] * x[0] - 2.0]);
    };

    it('should reduce cost while staying near the constraint manifold', () => {
      const initialParameters = new Float64Array([1.0]);
      const initialStates = new Float64Array([1.0]);
      const initialResidual = rosenbrockResidual(initialParameters, initialStates);
      const initialCost =
        0.5 * (initialResidual[0] ** 2 + initialResidual[1] ** 2);

      const result = constrainedLevenbergMarquardt(
        initialParameters,
        initialStates,
        rosenbrockResidual,
        circleConstraint,
        {
          maxIterations: 500,
          tolGradient: 1e-4,
          tolStep: 1e-6,
          constraintTolerance: 1e-3,
          lambdaInitial: 1e-2
        }
      );

      const finalConstraint = circleConstraint(result.finalParameters, result.finalStates);
      expect(result.finalCost).toBeLessThanOrEqual(initialCost);
      expect(result.finalCost).toBeLessThan(CONSTRAINED_MAX_COST);
      expect(vectorNorm(finalConstraint)).toBeLessThan(CONSTRAINT_NORM_TOLERANCE);
    });
  });
});

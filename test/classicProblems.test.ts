/**
 * Lean classic-problem integration suite.
 * Keeps a few high-signal end-to-end cases that unit tests for toy quadratics do not cover.
 */

import { gradientDescent } from '../src/core/gradientDescent';
import { levenbergMarquardt } from '../src/core/levenbergMarquardt';
import { constrainedGaussNewton } from '../src/core/constrainedGaussNewton';
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

const EXPONENTIAL_A_TOLERANCE = 0.2;
const EXPONENTIAL_B_TOLERANCE = 0.2;
const EXPONENTIAL_MAX_COST = 0.1;
const EXPECTED_EXPONENTIAL_A = 1.0;
const EXPECTED_EXPONENTIAL_B = 1.0;

const LINEAR_SLOPE_TOLERANCE = 0.1;
const LINEAR_MAX_COST = 0.1;
const EXPECTED_LINEAR_SLOPE = 2.0;

const CONSTRAINT_NORM_TOLERANCE = 1e-4;
const CONSTRAINED_PARAMETER_TOLERANCE = 1e-2;

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
    it('should recover approximate exponential parameters', () => {
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
      expect(result.finalCost).toBeLessThan(EXPONENTIAL_MAX_COST);
      expect(Math.abs(result.finalParameters[0] - EXPECTED_EXPONENTIAL_A)).toBeLessThan(EXPONENTIAL_A_TOLERANCE);
      expect(Math.abs(result.finalParameters[1] - EXPECTED_EXPONENTIAL_B)).toBeLessThan(EXPONENTIAL_B_TOLERANCE);
    });

    it('should recover approximate linear slope', () => {
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
      expect(Math.abs(result.finalParameters[0] - EXPECTED_LINEAR_SLOPE)).toBeLessThan(LINEAR_SLOPE_TOLERANCE);
      expect(result.finalCost).toBeLessThan(LINEAR_MAX_COST);
    });
  });

  describe('Equality-constrained least squares via constrained Gauss-Newton', () => {
    const residual: ConstrainedResidualFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] - 0.5, x[0] - 0.5]);
    };

    const constraint: ConstraintFn = (p: Float64Array, x: Float64Array) => {
      return new Float64Array([p[0] + x[0] - 1.0]);
    };

    it('should converge to the constrained optimum', () => {
      const result = constrainedGaussNewton(
        new Float64Array([0.0]),
        new Float64Array([1.0]),
        residual,
        constraint,
        {
          maxIterations: 100,
          tolerance: 1e-6,
          constraintTolerance: 1e-6
        }
      );

      expect(result.converged).toBe(true);
      expect(Math.abs(result.finalParameters[0] - 0.5)).toBeLessThan(CONSTRAINED_PARAMETER_TOLERANCE);
      expect(Math.abs(result.finalStates[0] - 0.5)).toBeLessThan(CONSTRAINED_PARAMETER_TOLERANCE);
      expect(vectorNorm(constraint(result.finalParameters, result.finalStates))).toBeLessThan(
        CONSTRAINT_NORM_TOLERANCE
      );
    });
  });
});

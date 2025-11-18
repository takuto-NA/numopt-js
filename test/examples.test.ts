import { describe, it, expect } from 'vitest';
import { gradientDescent, levenbergMarquardt } from '../src/index';
import type { CostFn, GradientFn, ResidualFn } from '../src/core/types';

describe('Example Problems', () => {
  describe('Gradient Descent Examples', () => {
    /**
     * Rosenbrock function: f(x, y) = (1-x)^2 + 100*(y-x^2)^2
     * Minimum at (1, 1)
     * This is a classic test function for optimization algorithms
     */
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

    it('should minimize Rosenbrock function', () => {
      const initialParams = new Float64Array([-1.0, 1.0]);
      const result = gradientDescent(initialParams, rosenbrockCost, rosenbrockGradient, {
        maxIterations: 10000,
        tolerance: 1e-4,
        useLineSearch: true
      });

      expect(result.converged).toBe(true);
      expect(Math.abs(result.parameters[0] - 1.0)).toBeLessThan(0.1);
      expect(Math.abs(result.parameters[1] - 1.0)).toBeLessThan(0.1);
    });
  });

  describe('Levenberg-Marquardt Examples', () => {
    /**
     * Curve fitting example: fit y = a * exp(b * x) to data points
     * Residual: r_i = a * exp(b * x_i) - y_i
     */
    const xData = new Float64Array([0, 1, 2, 3, 4]);
    const yData = new Float64Array([1.0, 2.7, 7.4, 20.1, 54.6]); // Approximate exponential

    const exponentialResidual: ResidualFn = (params: Float64Array) => {
      const a = params[0];
      const b = params[1];
      const residuals = new Float64Array(xData.length);

      for (let i = 0; i < xData.length; i++) {
        const predicted = a * Math.exp(b * xData[i]);
        residuals[i] = predicted - yData[i];
      }

      return residuals;
    };

    it('should fit exponential curve', () => {
      const initialParams = new Float64Array([1.0, 1.0]);
      const result = levenbergMarquardt(initialParams, exponentialResidual, {
        useNumericJacobian: true,
        maxIterations: 100,
        tolGradient: 1e-6
      });

      expect(result.converged).toBe(true);
      expect(result.finalCost).toBeLessThan(1.0); // Should fit reasonably well
    });

    /**
     * Linear regression: fit y = a * x + b
     * Residual: r_i = a * x_i + b - y_i
     */
    const linearXData = new Float64Array([1, 2, 3, 4, 5]);
    const linearYData = new Float64Array([2.1, 3.9, 6.1, 8.0, 9.9]);

    const linearResidual: ResidualFn = (params: Float64Array) => {
      const a = params[0];
      const b = params[1];
      const residuals = new Float64Array(linearXData.length);

      for (let i = 0; i < linearXData.length; i++) {
        const predicted = a * linearXData[i] + b;
        residuals[i] = predicted - linearYData[i];
      }

      return residuals;
    };

    it('should fit linear regression', () => {
      const initialParams = new Float64Array([0.0, 0.0]);
      const result = levenbergMarquardt(initialParams, linearResidual, {
        useNumericJacobian: true,
        maxIterations: 100,
        tolGradient: 1e-6
      });

      expect(result.converged).toBe(true);
      // Should find approximately a ≈ 2, b ≈ 0
      expect(Math.abs(result.parameters[0] - 2.0)).toBeLessThan(0.1);
      expect(result.finalCost).toBeLessThan(0.1);
    });
  });
});

